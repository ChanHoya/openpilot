"""Driving quality trip aggregator and local persistence for comma 3/3X parking reports."""

from __future__ import annotations

import datetime
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_REPORTS_DIR = "/data/drive_reports"
MAX_STORED_REPORTS = 100

MIN_TRIP_DISTANCE_M = 300.0
MIN_TRIP_DURATION_S = 60.0

HARD_BRAKE_THRESHOLD = -2.8  # m/s^2
HARD_BRAKE_MIN_DURATION_S = 0.2
HARD_BRAKE_COOLDOWN_S = 3.0
HARD_BRAKE_MIN_ONSET_SPEED_MPS = 1.5  # 5.4 km/h

MOVING_SPEED_MPS = 0.5
CHANNEL_RELEASE_DEBOUNCE_S = 0.5
MAX_SAMPLE_DT_S = 1.0
MAX_STORED_EVENTS = 20

_REPORT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,64}$")

_GEAR_INT_MAP = {
  0: "unknown",
  1: "park",
  2: "drive",
  3: "neutral",
  4: "reverse",
  5: "low",
  6: "eco",
  7: "sport",
  8: "manumatic",
  9: "brake",
}


def resolve_reports_dir(base_dir: str | Path | None = None) -> Path:
  if base_dir is not None:
    return Path(base_dir)
  env_dir = os.environ.get("DRIVE_REPORTS_DIR", "").strip()
  if env_dir:
    return Path(env_dir)
  return Path(DEFAULT_REPORTS_DIR)


def format_duration_ko(seconds: float) -> str:
  total_s = max(0, int(round(float(seconds or 0.0))))
  hours, rem = divmod(total_s, 3600)
  mins, secs = divmod(rem, 60)
  if hours > 0:
    return f"{hours}시간 {mins:02d}분 {secs:02d}초"
  if mins > 0:
    return f"{mins}분 {secs:02d}초"
  return f"{secs}초"


def format_duration_count_ko(seconds: float, count: int) -> str:
  return f"{format_duration_ko(seconds)} ({max(0, int(count))}회)"


def format_duration_compact_ko(seconds: float) -> str:
  total_s = max(0, int(round(float(seconds or 0.0))))
  mins, secs = divmod(total_s, 60)
  return f"{mins:02d}분{secs:02d}초"


def normalize_gear_shifter(gear: Any) -> str:
  if gear is None:
    return "unknown"
  raw = getattr(gear, "raw", gear)
  if isinstance(raw, int) and not isinstance(raw, bool):
    return _GEAR_INT_MAP.get(raw, "unknown")
  s = str(gear).strip().lower()
  if "." in s:
    s = s.rsplit(".", 1)[-1]
  return s or "unknown"


def _clamp(value: float, lo: float, hi: float) -> float:
  return max(lo, min(hi, value))


def _allocate_percentages(values: list[float]) -> list[int]:
  total = sum(max(0.0, float(v)) for v in values)
  if total <= 1e-9:
    return [0 for _ in values]
  exact = [(max(0.0, float(v)) / total) * 100.0 for v in values]
  floors = [int(math.floor(x)) for x in exact]
  rem = 100 - sum(floors)
  order = sorted(range(len(values)), key=lambda idx: exact[idx] - floors[idx], reverse=True)
  for idx in order[:max(0, rem)]:
    floors[idx] += 1
  return floors


def _grade_from_score(score: int) -> str:
  if score >= 92:
    return "S"
  if score >= 82:
    return "A"
  if score >= 70:
    return "B"
  return "C"


class _ChannelEdgeTracker:
  """Tracks accumulated active time and debounced rising-edge counts."""

  def __init__(self, release_debounce_s: float = CHANNEL_RELEASE_DEBOUNCE_S):
    self.release_debounce_s = float(release_debounce_s)
    self.seconds = 0.0
    self.count = 0
    self._active = False
    self._last_inactive_t: float | None = None
    self._inactive_elapsed = self.release_debounce_s

  def reset(self) -> None:
    self.seconds = 0.0
    self.count = 0
    self._active = False
    self._last_inactive_t = None
    self._inactive_elapsed = self.release_debounce_s

  def update(self, active: bool, dt: float) -> None:
    if active:
      self.seconds += dt
      if not self._active:
        if self.count == 0 or self._inactive_elapsed >= self.release_debounce_s:
          self.count += 1
        self._active = True
      self._inactive_elapsed = 0.0
    else:
      self._active = False
      self._inactive_elapsed += dt


class DriveReportTracker:
  """Aggregates per-frame driving metrics and emits a report when shifting into P."""

  def __init__(
    self,
    vehicle_name: str = "",
    min_distance_m: float = MIN_TRIP_DISTANCE_M,
    min_duration_s: float = MIN_TRIP_DURATION_S,
    hard_brake_threshold: float = HARD_BRAKE_THRESHOLD,
    hard_brake_min_duration_s: float = HARD_BRAKE_MIN_DURATION_S,
    hard_brake_cooldown_s: float = HARD_BRAKE_COOLDOWN_S,
  ):
    self.vehicle_name = str(vehicle_name or "").strip()
    self.min_distance_m = float(min_distance_m)
    self.min_duration_s = float(min_duration_s)
    self.hard_brake_threshold = float(hard_brake_threshold)
    self.hard_brake_min_duration_s = float(hard_brake_min_duration_s)
    self.hard_brake_cooldown_s = float(hard_brake_cooldown_s)

    self._prev_gear = "unknown"
    self._reset_trip()

  def set_vehicle_name(self, name: str) -> None:
    clean = str(name or "").strip()
    if clean:
      self.vehicle_name = clean

  def _reset_trip(self) -> None:
    self._trip_started = False
    self._has_driven_out_of_park = False
    self._prev_mono_t: float | None = None
    self._start_wall_t: float | None = None
    self._last_wall_t: float | None = None

    self._distance_m = 0.0
    self._duration_s = 0.0
    self._moving_s = 0.0
    self._idle_s = 0.0

    self._ratio_auto_s = 0.0
    self._ratio_steer_s = 0.0
    self._ratio_pedal_s = 0.0

    self._auto_ch = _ChannelEdgeTracker()
    self._steer_ch = _ChannelEdgeTracker()
    self._gas_ch = _ChannelEdgeTracker()
    self._brake_ch = _ChannelEdgeTracker()

    self._live_auto = False
    self._live_steer = False
    self._live_gas = False
    self._live_brake = False

    self._moving_speed_sum = 0.0
    self._all_speed_sum = 0.0
    self._max_speed_kph = 0.0

    self._prev_a_ego: float | None = None
    self._a_filt: float | None = None
    self._jerk_prev_a: float | None = None
    self._jerk_dt_accum: float = 0.0
    self._jerk_sq_sum = 0.0
    self._jerk_weight_s = 0.0

    self._lat_sq_sum = 0.0
    self._lat_weight_s = 0.0
    self._lat_accel_peak = 0.0

    # Hard brake candidate state
    self._hb_active = False
    self._hb_duration = 0.0
    self._hb_start_speed_kph = 0.0
    self._hb_min_speed_kph = 0.0
    self._hb_peak_decel = 0.0
    self._hb_manual_votes = 0
    self._hb_auto_votes = 0
    self._hb_wall_t: float | None = None
    self._hb_cooldown_until = -1e9

    self._auto_hb_count = 0
    self._manual_hb_count = 0
    self._auto_peak_decel = 0.0
    self._manual_peak_decel = 0.0
    self._hb_events: list[dict[str, Any]] = []

  def _close_hard_brake_candidate(self, mono_t: float) -> None:
    if not self._hb_active:
      return
    duration = self._hb_duration
    peak_decel = self._hb_peak_decel
    start_kph = self._hb_start_speed_kph
    end_kph = self._hb_min_speed_kph
    source = "manual" if self._hb_manual_votes > 0 else "auto"
    wall_t = self._hb_wall_t or self._last_wall_t or time.time()

    self._hb_active = False
    self._hb_duration = 0.0
    self._hb_manual_votes = 0
    self._hb_auto_votes = 0

    if duration < self.hard_brake_min_duration_s or peak_decel > self.hard_brake_threshold:
      return
    if start_kph < HARD_BRAKE_MIN_ONSET_SPEED_MPS * 3.6:
      return

    self._hb_cooldown_until = mono_t + self.hard_brake_cooldown_s
    if source == "manual":
      self._manual_hb_count += 1
      self._manual_peak_decel = min(self._manual_peak_decel, peak_decel)
    else:
      self._auto_hb_count += 1
      self._auto_peak_decel = min(self._auto_peak_decel, peak_decel)

    if len(self._hb_events) < MAX_STORED_EVENTS:
      dt_local = datetime.datetime.fromtimestamp(wall_t)
      self._hb_events.append({
        "timestamp": round(wall_t, 3),
        "time_str": dt_local.strftime("%H:%M"),
        "source": source,
        "source_label": "수동" if source == "manual" else "자동주행",
        "start_speed_kph": int(round(start_kph)),
        "end_speed_kph": int(round(end_kph)),
        "peak_decel": round(peak_decel, 2),
      })

  def _update_live_flags(self, car_state: Any, car_control: Any, selfdrive_state: Any) -> None:
    steering_pressed = bool(getattr(car_state, "steeringPressed", False))
    gas_pressed = bool(getattr(car_state, "gasPressed", False))
    brake_pressed = bool(getattr(car_state, "brakePressed", False))
    pedal_pressed = gas_pressed or brake_pressed
    sd_enabled = bool(getattr(selfdrive_state, "enabled", False)) if selfdrive_state is not None else False
    cc_enabled = bool(getattr(car_control, "enabled", False)) if car_control is not None else False
    cc_lat = bool(getattr(car_control, "latActive", False)) if car_control is not None else False
    cc_long = bool(getattr(car_control, "longActive", False)) if car_control is not None else False
    op_engaged = sd_enabled or cc_enabled or cc_lat or cc_long
    self._live_auto = bool(op_engaged and not steering_pressed and not pedal_pressed)
    self._live_steer = steering_pressed
    self._live_gas = gas_pressed
    self._live_brake = brake_pressed

  def get_live_stats(self) -> dict[str, Any]:
    auto_s = float(self._auto_ch.seconds)
    steer_s = float(self._steer_ch.seconds)
    gas_s = float(self._gas_ch.seconds)
    brake_s = float(self._brake_ch.seconds)
    denom = auto_s + steer_s + gas_s + brake_s
    entries = (
      ("auto", "자동주행", auto_s, self._live_auto),
      ("steer", "수동조향", steer_s, self._live_steer),
      ("gas", "수동가속", gas_s, self._live_gas),
      ("brake", "수동감속", brake_s, self._live_brake),
    )
    return {
      "trip_started": bool(self._trip_started),
      "gear": self._prev_gear,
      "items": [
        {
          "key": key,
          "label": label,
          "seconds": round(sec, 1),
          "time_text": format_duration_compact_ko(sec),
          "pct": round((sec / denom) * 100.0, 1) if denom > 1e-6 else 0.0,
          "active": bool(active),
        }
        for key, label, sec, active in entries
      ],
    }

  def is_eligible(self) -> bool:
    return (
      self._trip_started
      and self._has_driven_out_of_park
      and (self._distance_m >= self.min_distance_m or self._duration_s >= self.min_duration_s)
    )

  def _build_report(self, end_wall_t: float | None = None) -> dict[str, Any]:
    wall_end = float(end_wall_t or self._last_wall_t or time.time())
    wall_start = float(self._start_wall_t or max(0.0, wall_end - self._duration_s))
    if wall_end < wall_start:
      wall_end = wall_start + self._duration_s

    dt_start = datetime.datetime.fromtimestamp(wall_start)
    dt_end = datetime.datetime.fromtimestamp(wall_end)
    report_id = dt_end.strftime("%Y%m%d_%H%M%S")

    auto_pct, steer_pct, pedal_pct = _allocate_percentages([
      self._ratio_auto_s,
      self._ratio_steer_s,
      self._ratio_pedal_s,
    ])

    if self._moving_s > 1e-3:
      avg_speed_kph = (self._moving_speed_sum / self._moving_s) * 3.6
    elif self._duration_s > 1e-3:
      avg_speed_kph = (self._all_speed_sum / self._duration_s) * 3.6
    else:
      avg_speed_kph = 0.0

    long_jerk_rms = math.sqrt(self._jerk_sq_sum / self._jerk_weight_s) if self._jerk_weight_s > 1e-3 else 0.0
    lat_accel_rms = math.sqrt(self._lat_sq_sum / self._lat_weight_s) if self._lat_weight_s > 1e-3 else 0.0

    long_score = int(round(_clamp(100.0 - max(0.0, long_jerk_rms - 0.15) * 20.0, 35.0, 100.0)))
    lat_score = int(round(_clamp(100.0 - max(0.0, self._lat_accel_peak - 0.8) * 10.0 - max(0.0, lat_accel_rms - 0.4) * 12.0, 30.0, 100.0)))

    deductions: list[dict[str, Any]] = []
    penalty = 0.0

    if self._auto_hb_count > 0:
      pts = min(24, self._auto_hb_count * 6)
      penalty += pts
      deductions.append({"reason": f"자동주행 급정거 {self._auto_hb_count}회", "points": -pts})
    if self._manual_hb_count > 0:
      pts = min(32, self._manual_hb_count * 8)
      penalty += pts
      deductions.append({"reason": f"수동 급정거 {self._manual_hb_count}회", "points": -pts})

    long_pen = min(15, int(round(max(0.0, (100 - long_score) * 0.4))))
    if long_pen > 0:
      penalty += long_pen
      deductions.append({"reason": "종방향 가감속 변화", "points": -long_pen})

    lat_pen = min(10, int(round(max(0.0, (100 - lat_score) * 0.3))))
    if lat_pen > 0:
      penalty += lat_pen
      deductions.append({"reason": "횡방향 쏠림 가속도", "points": -lat_pen})

    if self._duration_s >= 60.0 and (auto_pct + steer_pct + pedal_pct) > 0 and auto_pct < 50:
      manual_pen = min(5, int(round((50 - auto_pct) * 0.1)))
      if manual_pen > 0:
        penalty += manual_pen
        deductions.append({"reason": "자동주행 비율 50% 미만", "points": -manual_pen})

    score = int(_clamp(round(100.0 - penalty), 0.0, 100.0))
    grade = _grade_from_score(score)

    total_hb = self._auto_hb_count + self._manual_hb_count
    distance_km = round(self._distance_m / 1000.0, 2)

    return {
      "schema_version": 1,
      "report_id": report_id,
      "vehicle_name": self.vehicle_name or "CarrotPilot",
      "started_at": round(wall_start, 3),
      "ended_at": round(wall_end, 3),
      "date_label": dt_start.strftime("%Y-%m-%d"),
      "time_range_label": f"{dt_start.strftime('%Y-%m-%d %H:%M')} ~ {dt_end.strftime('%H:%M')}",
      "distance_m": round(self._distance_m, 1),
      "distance_km": distance_km,
      "duration_s": round(self._duration_s, 1),
      "duration_text": format_duration_ko(self._duration_s),
      "moving_s": round(self._moving_s, 1),
      "moving_text": format_duration_ko(self._moving_s),
      "idle_s": round(self._idle_s, 1),
      "idle_text": format_duration_ko(self._idle_s),
      "score": score,
      "grade": grade,
      "deductions": deductions,
      "ratios": {
        "auto_pct": auto_pct,
        "steer_pct": steer_pct,
        "pedal_pct": pedal_pct,
      },
      "controls": {
        "auto": {
          "label": "자동주행",
          "seconds": round(self._auto_ch.seconds, 1),
          "count": self._auto_ch.count,
          "text": format_duration_count_ko(self._auto_ch.seconds, self._auto_ch.count),
        },
        "steer": {
          "label": "수동조향",
          "seconds": round(self._steer_ch.seconds, 1),
          "count": self._steer_ch.count,
          "text": format_duration_count_ko(self._steer_ch.seconds, self._steer_ch.count),
        },
        "gas": {
          "label": "수동페달 (가속)",
          "seconds": round(self._gas_ch.seconds, 1),
          "count": self._gas_ch.count,
          "text": format_duration_count_ko(self._gas_ch.seconds, self._gas_ch.count),
        },
        "brake": {
          "label": "수동페달 (브레이크)",
          "seconds": round(self._brake_ch.seconds, 1),
          "count": self._brake_ch.count,
          "text": format_duration_count_ko(self._brake_ch.seconds, self._brake_ch.count),
        },
      },
      "speed": {
        "avg_kph": round(avg_speed_kph, 1),
        "max_kph": round(self._max_speed_kph, 1),
        "long_jerk_rms": round(long_jerk_rms, 2),
        "long_score": long_score,
        "lat_accel_peak": round(self._lat_accel_peak, 2),
        "lat_accel_rms": round(lat_accel_rms, 2),
        "lat_score": lat_score,
      },
      "hard_brakes": {
        "threshold": self.hard_brake_threshold,
        "total_count": total_hb,
        "auto_count": self._auto_hb_count,
        "manual_count": self._manual_hb_count,
        "auto_peak_decel": round(self._auto_peak_decel, 2),
        "manual_peak_decel": round(self._manual_peak_decel, 2),
        "events": list(self._hb_events),
      },
      "file_path": "",
    }

  def finalize_if_eligible(self, wall_t: float | None = None) -> dict[str, Any] | None:
    if self._prev_mono_t is not None:
      self._close_hard_brake_candidate(self._prev_mono_t)
    if not self.is_eligible():
      self._reset_trip()
      return None
    report = self._build_report(wall_t)
    self._reset_trip()
    return report

  def update(
    self,
    mono_t: float,
    car_state: Any,
    car_control: Any = None,
    selfdrive_state: Any = None,
    controls_state: Any = None,
    wall_t: float | None = None,
  ) -> dict[str, Any] | None:
    if car_state is None:
      return None

    mono_t = float(mono_t)
    now_wall = float(time.time() if wall_t is None else wall_t)
    gear = normalize_gear_shifter(getattr(car_state, "gearShifter", None))
    v_ego = max(0.0, float(getattr(car_state, "vEgo", 0.0) or 0.0))
    if not math.isfinite(v_ego) or v_ego > 90.0:
      v_ego = 0.0
    a_ego = float(getattr(car_state, "aEgo", 0.0) or 0.0)
    if not math.isfinite(a_ego):
      a_ego = 0.0

    # Entering P after driving finalizes and resets trip metrics.
    if gear == "park":
      self._live_auto = False
      self._live_steer = False
      self._live_gas = False
      self._live_brake = False
      if self._prev_gear != "park" and self._trip_started:
        report = self.finalize_if_eligible(now_wall)
        self._prev_gear = "park"
        return report
      self._prev_gear = "park"
      if self._trip_started and not self._has_driven_out_of_park:
        self._reset_trip()
      return None

    # Ignore unknown gear right at boot unless already in an active trip or moving.
    if gear == "unknown" and not self._trip_started and v_ego < MOVING_SPEED_MPS:
      self._prev_gear = gear
      return None

    self._prev_gear = gear

    if not self._trip_started:
      self._trip_started = True
      self._prev_mono_t = mono_t
      self._start_wall_t = now_wall
      self._last_wall_t = now_wall
      self._prev_a_ego = a_ego
      self._update_live_flags(car_state, car_control, selfdrive_state)
      if v_ego >= MOVING_SPEED_MPS or gear in ("drive", "reverse", "low", "sport", "manumatic"):
        if v_ego >= MOVING_SPEED_MPS:
          self._has_driven_out_of_park = True
      return None

    if self._prev_mono_t is None or mono_t < self._prev_mono_t:
      self._prev_mono_t = mono_t
      self._prev_a_ego = a_ego
      self._last_wall_t = now_wall
      self._update_live_flags(car_state, car_control, selfdrive_state)
      return None

    dt = min(mono_t - self._prev_mono_t, MAX_SAMPLE_DT_S)
    self._prev_mono_t = mono_t
    self._last_wall_t = now_wall

    if dt <= 0.0:
      self._update_live_flags(car_state, car_control, selfdrive_state)
      return None

    if v_ego >= MOVING_SPEED_MPS or self._distance_m >= 5.0:
      self._has_driven_out_of_park = True

    self._duration_s += dt
    self._distance_m += v_ego * dt

    is_moving = v_ego >= MOVING_SPEED_MPS
    if is_moving:
      self._moving_s += dt
      self._moving_speed_sum += v_ego * dt
    else:
      self._idle_s += dt

    self._all_speed_sum += v_ego * dt
    speed_kph = v_ego * 3.6
    if speed_kph > self._max_speed_kph:
      self._max_speed_kph = speed_kph

    steering_pressed = bool(getattr(car_state, "steeringPressed", False))
    gas_pressed = bool(getattr(car_state, "gasPressed", False))
    brake_pressed = bool(getattr(car_state, "brakePressed", False))
    pedal_pressed = gas_pressed or brake_pressed

    sd_enabled = bool(getattr(selfdrive_state, "enabled", False)) if selfdrive_state is not None else False
    cc_enabled = bool(getattr(car_control, "enabled", False)) if car_control is not None else False
    cc_lat = bool(getattr(car_control, "latActive", False)) if car_control is not None else False
    cc_long = bool(getattr(car_control, "longActive", False)) if car_control is not None else False
    op_engaged = sd_enabled or cc_enabled or cc_lat or cc_long

    # Mutually-exclusive ratio buckets + consistent channel counters (moving time only; idle stop time excluded)
    auto_pure = op_engaged and not steering_pressed and not pedal_pressed
    if is_moving:
      if pedal_pressed:
        self._ratio_pedal_s += dt
      elif steering_pressed:
        self._ratio_steer_s += dt
      elif op_engaged:
        self._ratio_auto_s += dt

    self._auto_ch.update(auto_pure and is_moving, dt)
    self._steer_ch.update(steering_pressed and is_moving, dt)
    self._gas_ch.update(gas_pressed and is_moving, dt)
    self._brake_ch.update(brake_pressed and is_moving, dt)
    self._live_auto = bool(auto_pure)
    self._live_steer = bool(steering_pressed)
    self._live_gas = bool(gas_pressed)
    self._live_brake = bool(brake_pressed)

    # Smoothness metrics during movement (low-pass filtered acceleration + >= 0.15s window
    # removes CAN quantization noise and keeps 20Hz UI vs 100Hz rlog comparable).
    if self._a_filt is None:
      self._a_filt = a_ego
    else:
      alpha = dt / (0.20 + dt)
      self._a_filt += alpha * (a_ego - self._a_filt)

    if v_ego >= MOVING_SPEED_MPS:
      if self._jerk_prev_a is None:
        self._jerk_prev_a = self._a_filt
        self._jerk_dt_accum = 0.0
      else:
        self._jerk_dt_accum += dt
        if self._jerk_dt_accum >= 0.15:
          jerk = _clamp((self._a_filt - self._jerk_prev_a) / self._jerk_dt_accum, -15.0, 15.0)
          self._jerk_sq_sum += (jerk * jerk) * self._jerk_dt_accum
          self._jerk_weight_s += self._jerk_dt_accum
          self._jerk_prev_a = self._a_filt
          self._jerk_dt_accum = 0.0

      curvature = float(getattr(controls_state, "curvature", 0.0) or 0.0) if controls_state is not None else 0.0
      if not math.isfinite(curvature):
        curvature = 0.0
      lat_accel = abs(curvature) * (v_ego * v_ego)
      if lat_accel <= 15.0:
        self._lat_sq_sum += (lat_accel * lat_accel) * dt
        self._lat_weight_s += dt
        if lat_accel > self._lat_accel_peak:
          self._lat_accel_peak = lat_accel
    else:
      self._jerk_prev_a = None
      self._jerk_dt_accum = 0.0

    self._prev_a_ego = a_ego

    # Track non-event overall decels only up to candidate events / peak decel tracking
    if a_ego <= self.hard_brake_threshold and mono_t >= self._hb_cooldown_until:
      if not self._hb_active:
        if v_ego >= HARD_BRAKE_MIN_ONSET_SPEED_MPS:
          self._hb_active = True
          self._hb_duration = dt
          self._hb_start_speed_kph = speed_kph
          self._hb_min_speed_kph = speed_kph
          self._hb_peak_decel = a_ego
          self._hb_manual_votes = 1 if brake_pressed else 0
          self._hb_auto_votes = 0 if brake_pressed else 1
          self._hb_wall_t = now_wall
      else:
        self._hb_duration += dt
        self._hb_min_speed_kph = min(self._hb_min_speed_kph, speed_kph)
        self._hb_peak_decel = min(self._hb_peak_decel, a_ego)
        if brake_pressed:
          self._hb_manual_votes += 1
        else:
          self._hb_auto_votes += 1
    elif self._hb_active:
      self._close_hard_brake_candidate(mono_t)

    # Track baseline peak decel per mode even when mild
    if a_ego < 0.0 and v_ego >= MOVING_SPEED_MPS:
      if brake_pressed:
        if a_ego < self._manual_peak_decel:
          self._manual_peak_decel = a_ego
      elif op_engaged:
        if a_ego < self._auto_peak_decel:
          self._auto_peak_decel = a_ego

    return None


class DriveReportStore:
  """Atomic JSON store + bounded index under /data/drive_reports (or DRIVE_REPORTS_DIR)."""

  def __init__(self, base_dir: str | Path | None = None, max_reports: int = MAX_STORED_REPORTS):
    self.base_dir = resolve_reports_dir(base_dir)
    self.max_reports = max(1, int(max_reports))
    self._lock = threading.RLock()

  @property
  def index_path(self) -> Path:
    return self.base_dir / "index.json"

  def _ensure_dir(self) -> None:
    self.base_dir.mkdir(parents=True, exist_ok=True)

  def _atomic_write_json(self, path: Path, payload: Any) -> None:
    self._ensure_dir()
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    with open(tmp_path, "w", encoding="utf-8") as f:
      f.write(text)
      f.flush()
      os.fsync(f.fileno())
    os.replace(tmp_path, path)

  def _sanitize_report_id(self, report_id: str) -> str | None:
    clean = str(report_id or "").strip()
    if not _REPORT_ID_RE.fullmatch(clean):
      return None
    return clean

  def _summary_entry(self, report: dict[str, Any]) -> dict[str, Any]:
    return {
      "report_id": report["report_id"],
      "date_label": report.get("date_label", ""),
      "time_range_label": report.get("time_range_label", ""),
      "vehicle_name": report.get("vehicle_name", ""),
      "distance_km": report.get("distance_km", 0.0),
      "duration_s": report.get("duration_s", 0.0),
      "duration_text": report.get("duration_text", ""),
      "score": report.get("score", 0),
      "grade": report.get("grade", ""),
      "ended_at": report.get("ended_at", 0.0),
      "file_path": report.get("file_path", ""),
    }

  def _load_index_locked(self) -> list[dict[str, Any]]:
    if not self.index_path.exists():
      return []
    try:
      data = json.loads(self.index_path.read_text(encoding="utf-8"))
      if isinstance(data, list):
        return [item for item in data if isinstance(item, dict) and "report_id" in item]
    except Exception:
      pass
    return []

  def save_report(self, report: dict[str, Any]) -> dict[str, Any]:
    with self._lock:
      self._ensure_dir()
      payload = dict(report)
      base_id = self._sanitize_report_id(payload.get("report_id", "")) or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
      report_id = base_id
      suffix = 1
      while (self.base_dir / f"{report_id}.json").exists():
        report_id = f"{base_id}_{suffix:02d}"
        suffix += 1
      payload["report_id"] = report_id

      report_path = self.base_dir / f"{report_id}.json"
      payload["file_path"] = str(report_path)
      self._atomic_write_json(report_path, payload)

      entries = [item for item in self._load_index_locked() if item.get("report_id") != report_id]
      entries.insert(0, self._summary_entry(payload))
      entries.sort(key=lambda item: float(item.get("ended_at", 0.0) or 0.0), reverse=True)

      removed = entries[self.max_reports:]
      entries = entries[:self.max_reports]
      for old in removed:
        old_id = self._sanitize_report_id(old.get("report_id", ""))
        if old_id:
          try:
            (self.base_dir / f"{old_id}.json").unlink(missing_ok=True)
          except Exception:
            pass

      self._atomic_write_json(self.index_path, entries)
      return payload

  def get_report(self, report_id: str) -> dict[str, Any] | None:
    clean_id = self._sanitize_report_id(report_id)
    if not clean_id:
      return None
    with self._lock:
      path = self.base_dir / f"{clean_id}.json"
      if not path.exists():
        return None
      try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
      except Exception:
        return None

  def list_reports(self, limit: int = 50, include_full: bool = False) -> list[dict[str, Any]]:
    with self._lock:
      entries = self._load_index_locked()
      if not entries and self.base_dir.exists():
        # Rebuild index from JSON files on disk if index.json is missing.
        for path in sorted(self.base_dir.glob("*.json"), reverse=True):
          if path.name == "index.json":
            continue
          try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "report_id" in data:
              entries.append(self._summary_entry(data))
          except Exception:
            continue
        entries.sort(key=lambda item: float(item.get("ended_at", 0.0) or 0.0), reverse=True)

      selected = entries[:max(1, int(limit))]
      if not include_full:
        return [dict(item) for item in selected]

      results: list[dict[str, Any]] = []
      for item in selected:
        full = self.get_report(item["report_id"])
        results.append(full if full is not None else dict(item))
      return results

  def delete_report(self, report_id: str) -> bool:
    clean_id = self._sanitize_report_id(report_id)
    if not clean_id:
      return False
    with self._lock:
      path = self.base_dir / f"{clean_id}.json"
      existed = path.exists()
      path.unlink(missing_ok=True)
      entries = self._load_index_locked()
      filtered = [item for item in entries if item.get("report_id") != clean_id]
      if len(filtered) != len(entries):
        existed = True
        self._atomic_write_json(self.index_path, filtered)
      return existed

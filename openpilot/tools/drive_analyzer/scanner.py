"""First-pass deterministic anomaly scanner across downloaded rlog segments and drive reports."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any, Sequence

_SEG_INDEX_RE = re.compile(r"--(\d+)(?:_|--|$)")
_IMPORTANT_PARAMS = (
  "LongitudinalPersonality",
  "MyDrivingMode",
  "CruiseEcoControl",
  "StopDistanceCarrot",
  "TFollowGap1",
  "TFollowGap2",
  "TFollowGap3",
  "TFollowGap4",
  "UseLaneLineSpeed",
  "LongTuningKpV",
  "LongTuningKiV",
  "LongTuningKf",
  "LongActuatorDelay",
  "CurveInnerBias",
  "LanelessCenterAdjust",
  "AutoCurveSpeedCtrlUse",
  "AutoCurveSpeedAggressiveness",
  "AutoCurveSpeedFactor",
  "AutoCurveSpeedFactorIn",
  "MapTurnSpeedFactor",
  "TurnSpeedControlMode",
  "SteerActuatorDelay",
  "LatAccelFactor",
  "Friction",
)


@dataclass(frozen=True)
class FrameSample:
  t: float
  v_kph: float
  a_ego: float
  long_active: bool
  gas_pressed: bool
  brake_pressed: bool
  lead_status: bool
  lead_radar: bool
  lead_d: float
  lead_v: float
  lead_v_rel: float
  lead_a: float
  lead_prob: float
  a_base: float
  a_target: float
  mpc_source: str
  should_stop: bool
  fcw: bool
  v_target_now: float = 0.0
  j_target_now: float = 0.0
  t_follow: float = 1.3
  lat_active: bool = False
  steering_pressed: bool = False
  blinker_active: bool = False
  lane_changing: bool = False
  steer_angle_deg: float = 0.0
  steer_angle_desired_deg: float = 0.0
  steer_rate_deg: float = 0.0
  steer_torque: float = 0.0
  steer_saturated: bool = False
  curvature: float = 0.0
  desired_curvature: float = 0.0
  lat_accel: float = 0.0
  desired_lat_accel: float = 0.0
  lane_offset_m: float | None = None
  lane_width_m: float = 0.0
  model_t: tuple[float, ...] = ()
  model_yaws: tuple[float, ...] = ()
  model_yaw_rates: tuple[float, ...] = ()
  lane_y_left: float | None = None
  lane_y_right: float | None = None
  lane_prob_left: float = 0.0
  lane_prob_right: float = 0.0


@dataclass
class AnomalyEvent:
  category: str
  severity: str
  t_start: float
  t_end: float
  segment: str
  route: str
  segment_path: str
  v_kph: float
  summary: str
  metrics: dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass
class SegmentSummary:
  segment: str
  route: str
  segment_path: str
  commit: str = ""
  branch: str = ""
  dirty: bool = False
  duration_s: float = 0.0
  frames: int = 0
  active_frames: int = 0
  lat_active_frames: int = 0
  v_min_kph: float = 0.0
  v_max_kph: float = 0.0
  a_ego_min: float = 0.0
  a_ego_max: float = 0.0
  lat_accel_max: float = 0.0
  steer_sat_frames: int = 0
  curve_steer_rate_rms: float = 0.0
  params: dict[str, str] = field(default_factory=dict)
  anomalies: list[AnomalyEvent] = field(default_factory=list)
  error: str = ""
  started_at: str = ""
  ended_at: str = ""
  started_at_ts: float = 0.0
  t_offset_s: float = 0.0
  telemetry: dict[str, list[Any]] = field(default_factory=dict)

  def to_dict(self) -> dict[str, Any]:
    d = asdict(self)
    d["anomalies"] = [a.to_dict() for a in self.anomalies]
    return d


@dataclass
class ScanReport:
  segments: list[SegmentSummary]
  anomalies: list[AnomalyEvent]
  drive_reports: list[dict[str, Any]] = field(default_factory=list)

  def to_dict(self) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for a in self.anomalies:
      counts[a.category] = counts.get(a.category, 0) + 1
    return {
      "segment_count": len(self.segments),
      "anomaly_count": len(self.anomalies),
      "anomaly_counts": counts,
      "segments": [s.to_dict() for s in self.segments],
      "anomalies": [a.to_dict() for a in self.anomalies],
      "drive_reports_summary": [
        {
          "report_id": r.get("report_id"),
          "score": r.get("score"),
          "grade": r.get("grade"),
          "distance_km": r.get("distance_km"),
          "duration_s": r.get("duration_s"),
          "hard_brake_count": r.get("hard_brake_count"),
          "auto_hard_brake_count": r.get("auto_hard_brake_count"),
          "manual_hard_brake_count": r.get("manual_hard_brake_count"),
          "time_range_label": r.get("time_range_label"),
          "started_at": r.get("started_at"),
          "ended_at": r.get("ended_at"),
          "deductions": r.get("deductions"),
          "ratios": r.get("ratios"),
          "controls": r.get("controls"),
          "speed": r.get("speed"),
          "hard_brakes": r.get("hard_brakes"),
        }
        for r in self.drive_reports[:10]
      ],
    }


def _natural_sort_key(path: Path) -> tuple[str, int, str]:
  name = path.name
  m = _SEG_INDEX_RE.search(name)
  idx = int(m.group(1)) if m else -1
  prefix = name[: m.start()] if m else name
  return (str(path.parent), idx, prefix)


def _infer_segment_and_route(path: Path) -> tuple[str, str]:
  name = path.name
  if name in ("rlog.zst", "rlog.bz2", "rlog"):
    seg = path.parent.name or path.stem
  else:
    seg = re.sub(r"_rlog(\.(?:zst|bz2))?$|--rlog(\.(?:zst|bz2))?$", "", name)
  route = re.sub(r"--\d+$", "", seg)
  return seg, route


def discover_rlogs(paths: Sequence[Path | str]) -> list[Path]:
  found: list[Path] = []
  seen: set[Path] = set()
  for raw in paths:
    p = Path(raw).expanduser().resolve()
    if p.is_file():
      if p not in seen and ("rlog" in p.name or p.suffix in (".zst", ".bz2")):
        seen.add(p)
        found.append(p)
    elif p.is_dir():
      for child in p.rglob("*"):
        if not child.is_file():
          continue
        cname = child.name
        if cname.endswith("_rlog.zst") or cname.endswith("_rlog.bz2") or cname in ("rlog.zst", "rlog.bz2") or cname.endswith("--rlog.zst"):
          resolved = child.resolve()
          if resolved not in seen:
            seen.add(resolved)
            found.append(resolved)
  found.sort(key=_natural_sort_key)
  return found


def load_segment_frames(path: Path) -> tuple[dict[str, Any], list[FrameSample]]:
  from openpilot.tools.lib.logreader import LogReader

  lr = LogReader(str(path))
  meta: dict[str, Any] = {
    "commit": "",
    "branch": "",
    "dirty": False,
    "params": {},
    "stop_dist": 5.0,
    "curve_inner_bias": 40.0,
    "started_at": "",
    "ended_at": "",
    "started_at_ts": 0.0,
  }
  frames: list[FrameSample] = []

  cs = rs = cc = m = cs_ctrl = lp_lat = None
  t0: float | None = None
  route_start_epoch: float | None = None
  init_wall_epoch: float | None = None
  gps_route_epoch: float | None = None

  for msg in lr:
    w = msg.which()
    if t0 is None:
      t0 = msg.logMonoTime * 1e-9
    t = msg.logMonoTime * 1e-9 - t0

    if w == "clocks" and route_start_epoch is None:
      wt = getattr(msg.clocks, "wallTimeNanos", 0)
      if wt and wt > 1e17:
        route_start_epoch = wt * 1e-9 - t
    elif w in ("gpsLocationExternal", "gpsLocation") and gps_route_epoch is None:
      try:
        gps_ms = getattr(getattr(msg, w), "unixTimestampMillis", 0)
        if gps_ms and gps_ms > 1e11:
          gps_route_epoch = gps_ms * 1e-3 - t
      except Exception:
        pass
    elif w == "initData" and not meta["commit"]:
      init = msg.initData
      wt = getattr(init, "wallTimeNanos", 0)
      if wt and wt > 1e17 and init_wall_epoch is None:
        init_wall_epoch = wt * 1e-9
      meta["commit"] = str(getattr(init, "gitCommit", ""))[:12]
      meta["branch"] = str(getattr(init, "gitBranch", ""))
      meta["dirty"] = bool(getattr(init, "dirty", False))
      params = getattr(init, "params", None)
      if params is not None and hasattr(params, "entries"):
        for entry in params.entries:
          if entry.key in _IMPORTANT_PARAMS:
            try:
              val = entry.value.decode("utf-8", errors="ignore")
            except Exception:
              val = repr(entry.value)
            meta["params"][entry.key] = val
            if entry.key == "StopDistanceCarrot":
              try:
                meta["stop_dist"] = max(2.0, min(12.0, float(val) / 100.0))
              except ValueError:
                pass
            elif entry.key == "CurveInnerBias":
              try:
                meta["curve_inner_bias"] = max(0.0, min(100.0, float(val)))
              except ValueError:
                pass
    elif w == "carState":
      cs = msg.carState
    elif w == "radarState":
      rs = msg.radarState
    elif w == "carControl":
      cc = msg.carControl
    elif w == "controlsState":
      cs_ctrl = msg.controlsState
    elif w == "lateralPlan":
      lp_lat = msg.lateralPlan
    elif w == "modelV2":
      m = msg.modelV2
    elif w == "longitudinalPlan":
      lp = msg.longitudinalPlan
      if cs is None or rs is None:
        continue
      l1 = rs.leadOne
      has_lead = bool(l1.status)
      lead_radar = bool(getattr(l1, "radar", False)) if has_lead else False
      lead_d = float(l1.dRel) if has_lead else 0.0
      lead_v = float(l1.vLead) if has_lead else 0.0
      lead_v_rel = float(l1.vRel) if has_lead else 0.0
      lead_a = float(l1.aLeadK) if has_lead else 0.0
      lead_prob = float(getattr(l1, "modelProb", 0.0)) if has_lead else 0.0
      if has_lead and lead_prob <= 0.0 and m is not None and len(getattr(m, "leadsV3", ())) > 0:
        lead_prob = float(getattr(m.leadsV3[0], "prob", 0.0))

      v_ego = float(cs.vEgo)
      a_target = float(lp.aTarget)
      a_base = float(getattr(lp, "aTargetBase", a_target))
      t_follow = float(getattr(lp, "tFollow", 1.3)) or 1.3

      lat_active = bool(cc.latActive) if cc is not None else False
      blinker_active = bool(getattr(cs, "leftBlinker", False) or getattr(cs, "rightBlinker", False))
      lane_changing = bool(lp_lat is not None and str(getattr(lp_lat, "laneChangeState", "off")) != "off")

      curvature = float(getattr(cs_ctrl, "curvature", 0.0)) if cs_ctrl is not None else 0.0
      desired_curvature = float(getattr(cs_ctrl, "desiredCurvature", 0.0)) if cs_ctrl is not None else 0.0
      steer_angle_desired_deg = 0.0
      steer_saturated = False
      if cs_ctrl is not None and hasattr(cs_ctrl, "lateralControlState"):
        ls = cs_ctrl.lateralControlState
        try:
          sub = getattr(ls, ls.which())
          steer_saturated = bool(getattr(sub, "saturated", False))
          steer_angle_desired_deg = float(getattr(sub, "steeringAngleDesiredDeg", 0.0))
        except Exception:
          pass
      if steer_angle_desired_deg == 0.0 and cc is not None and hasattr(cc, "actuators"):
        steer_angle_desired_deg = float(getattr(cc.actuators, "steeringAngleDeg", 0.0))

      lane_offset_m: float | None = None
      lane_y_left: float | None = None
      lane_y_right: float | None = None
      lane_prob_left = 0.0
      lane_prob_right = 0.0
      model_t: tuple[float, ...] = ()
      model_yaws: tuple[float, ...] = ()
      model_yaw_rates: tuple[float, ...] = ()
      if m is not None:
        llines = getattr(m, "laneLines", ())
        lprobs = getattr(m, "laneLineProbs", ())
        if len(llines) >= 3 and len(lprobs) >= 3:
          try:
            lane_prob_left = float(lprobs[1])
            lane_prob_right = float(lprobs[2])
            if len(llines[1].y) > 0 and len(llines[2].y) > 0:
              lane_y_left = float(llines[1].y[0])
              lane_y_right = float(llines[2].y[0])
              width = abs(lane_y_right - lane_y_left)
              if lane_prob_left >= 0.55 and lane_prob_right >= 0.55 and 2.5 <= width <= 4.8:
                lane_offset_m = 0.5 * (lane_y_left + lane_y_right)
          except Exception:
            pass
        try:
          pos = getattr(m, "position", None)
          ori = getattr(m, "orientation", None)
          orate = getattr(m, "orientationRate", None)
          if pos is not None and ori is not None and orate is not None:
            t_vals = getattr(pos, "t", ())
            z_vals = getattr(ori, "z", ())
            zr_vals = getattr(orate, "z", ())
            if len(t_vals) == len(z_vals) == len(zr_vals) and len(t_vals) >= 10:
              model_t = tuple(float(v) for v in t_vals)
              model_yaws = tuple(float(v) for v in z_vals)
              model_yaw_rates = tuple(float(v) for v in zr_vals)
        except Exception:
          pass

      frames.append(
        FrameSample(
          t=t,
          v_kph=v_ego * 3.6,
          a_ego=float(cs.aEgo),
          long_active=bool(cc.longActive) if cc is not None else False,
          gas_pressed=bool(cs.gasPressed),
          brake_pressed=bool(cs.brakePressed),
          lead_status=has_lead,
          lead_radar=lead_radar,
          lead_d=lead_d,
          lead_v=lead_v,
          lead_v_rel=lead_v_rel,
          lead_a=lead_a,
          lead_prob=lead_prob,
          a_base=a_base,
          a_target=a_target,
          mpc_source=str(lp.longitudinalPlanSource),
          should_stop=bool(lp.shouldStop),
          fcw=bool(lp.fcw),
          v_target_now=float(getattr(lp, "vTargetNow", v_ego)),
          j_target_now=float(getattr(lp, "jTargetNow", 0.0)),
          t_follow=t_follow,
          lat_active=lat_active,
          steering_pressed=bool(getattr(cs, "steeringPressed", False)),
          blinker_active=blinker_active,
          lane_changing=lane_changing,
          steer_angle_deg=float(getattr(cs, "steeringAngleDeg", 0.0)),
          steer_angle_desired_deg=steer_angle_desired_deg,
          steer_rate_deg=float(getattr(cs, "steeringRateDeg", 0.0)),
          steer_torque=float(getattr(cs, "steeringTorque", 0.0)),
          steer_saturated=steer_saturated,
          curvature=curvature,
          desired_curvature=desired_curvature,
          lat_accel=curvature * (v_ego ** 2),
          desired_lat_accel=desired_curvature * (v_ego ** 2),
          lane_offset_m=lane_offset_m,
          lane_width_m=float(getattr(lp_lat, "laneWidth", 0.0)) if lp_lat is not None else 0.0,
          model_t=model_t,
          model_yaws=model_yaws,
          model_yaw_rates=model_yaw_rates,
          lane_y_left=lane_y_left,
          lane_y_right=lane_y_right,
          lane_prob_left=lane_prob_left,
          lane_prob_right=lane_prob_right,
        )
      )

  base_epoch = route_start_epoch if route_start_epoch is not None else gps_route_epoch
  if frames:
    if base_epoch is not None:
      seg_start_ts = base_epoch + frames[0].t
      seg_end_ts = base_epoch + frames[-1].t
    elif init_wall_epoch is not None:
      seg_start_ts = init_wall_epoch
      seg_end_ts = init_wall_epoch + max(0.0, frames[-1].t - frames[0].t)
    else:
      seg_start_ts = 0.0
      seg_end_ts = 0.0
    if seg_start_ts > 1e9:
      meta["started_at_ts"] = round(seg_start_ts, 2)
      meta["started_at"] = datetime.fromtimestamp(seg_start_ts).strftime("%Y-%m-%d %H:%M:%S")
      meta["ended_at"] = datetime.fromtimestamp(seg_end_ts).strftime("%H:%M:%S")
  elif base_epoch is not None and base_epoch > 1e9:
    meta["started_at_ts"] = round(base_epoch, 2)
    meta["started_at"] = datetime.fromtimestamp(base_epoch).strftime("%Y-%m-%d %H:%M:%S")

  return meta, frames


def _merge_hits(
  hits: list[dict[str, Any]],
  *,
  category: str,
  segment: str,
  route: str,
  segment_path: str,
  cooldown_s: float = 2.0,
) -> list[AnomalyEvent]:
  if not hits:
    return []
  groups: list[list[dict[str, Any]]] = [[hits[0]]]
  for item in hits[1:]:
    if item["t"] - groups[-1][-1]["t"] <= cooldown_s:
      groups[-1].append(item)
    else:
      groups.append([item])

  events: list[AnomalyEvent] = []
  for grp in groups:
    best = max(grp, key=lambda x: x["score"])
    events.append(
      AnomalyEvent(
        category=category,
        severity=best["severity"],
        t_start=round(grp[0]["t"], 2),
        t_end=round(grp[-1]["t"], 2),
        segment=segment,
        route=route,
        segment_path=segment_path,
        v_kph=round(best["v_kph"], 1),
        summary=best["summary"],
        metrics=best["metrics"],
      )
    )
  return events


def _detect_surge_events(
  frames: Sequence[FrameSample],
  *,
  category: str,
  v_lo: float,
  v_hi: float,
  segment: str,
  route: str,
  segment_path: str,
) -> list[AnomalyEvent]:
  hits: list[dict[str, Any]] = []
  signs: list[tuple[float, int, float]] = []
  for i, f in enumerate(frames):
    active = f.long_active and not f.gas_pressed and not f.brake_pressed
    if not active or not (v_lo <= f.v_kph < v_hi):
      signs.clear()
      continue
    s = 1 if f.a_target > 0.25 else (-1 if f.a_target < -0.25 else 0)
    if s != 0:
      if not signs or signs[-1][1] != s:
        signs.append((f.t, s, f.a_target))
      while signs and f.t - signs[0][0] > 8.0:
        signs.pop(0)
      flips = max(0, len(signs) - 1)
      if flips >= 3:
        hits.append({
          "t": f.t,
          "v_kph": f.v_kph,
          "score": flips * 1.0,
          "severity": "high" if flips >= 4 else "medium",
          "summary": f"{v_lo:.0f}~{v_hi:.0f}km/h 가감속 반전 {flips}회 (8초 내 aTarget 진동)",
          "metrics": {"flips_8s": flips, "a_target": round(f.a_target, 2), "a_ego": round(f.a_ego, 2), "lead_d": round(f.lead_d, 1)},
        })

    # Abrupt mid-speed snap without matching base drop
    if v_lo >= 30.0 and i >= 2:
      prev2 = frames[i - 2]
      if prev2.long_active and not prev2.gas_pressed and not prev2.brake_pressed and (f.t - prev2.t) <= 0.20:
        da = f.a_target - prev2.a_target
        if da <= -0.40 and f.a_base >= -0.05:
          hits.append({
            "t": f.t,
            "v_kph": f.v_kph,
            "score": abs(da) + 2.0,
            "severity": "high" if da <= -0.60 else "medium",
            "summary": f"중속 목표 가속도 급감 ({prev2.a_target:+.2f}->{f.a_target:+.2f}, aBase={f.a_base:+.2f})",
            "metrics": {"da_target": round(da, 2), "a_base": round(f.a_base, 2), "a_target": round(f.a_target, 2), "lead_d": round(f.lead_d, 1)},
          })
          continue

    # Short-window physical aEgo oscillation with sign flip in low speed
    if v_lo < 30.0 and i >= 20:
      win = [frames[j] for j in range(max(0, i - 40), i + 1)
             if frames[j].long_active and not frames[j].gas_pressed and not frames[j].brake_pressed and (f.t - frames[j].t) <= 2.0]
      if len(win) >= 15:
        amin = min(x.a_ego for x in win)
        amax = max(x.a_ego for x in win)
        if (amax - amin) >= 1.8 and amax > 0.45 and amin < -0.45:
          hits.append({
            "t": f.t,
            "v_kph": f.v_kph,
            "score": amax - amin,
            "severity": "high" if (amax - amin) >= 2.2 else "medium",
            "summary": f"저속 실가속도 출렁임 ({amin:+.2f} ~ {amax:+.2f} m/s²)",
            "metrics": {"a_ego_min": round(amin, 2), "a_ego_max": round(amax, 2), "span": round(amax - amin, 2), "lead_d": round(f.lead_d, 1)},
          })

  return _merge_hits(hits, category=category, segment=segment, route=route, segment_path=segment_path, cooldown_s=3.0)


def analyze_frames(
  segment_name: str,
  frames: Sequence[FrameSample],
  *,
  segment_path: str = "",
  route: str = "",
  metadata: dict[str, Any] | None = None,
) -> SegmentSummary:
  meta = metadata or {}
  started_at = str(meta.get("started_at", ""))
  ended_at = str(meta.get("ended_at", ""))
  started_at_ts = float(meta.get("started_at_ts", 0.0) or 0.0)
  if frames and started_at_ts > 1e9 and not started_at:
    dur = max(0.0, frames[-1].t - frames[0].t)
    started_at = datetime.fromtimestamp(started_at_ts).strftime("%Y-%m-%d %H:%M:%S")
    ended_at = datetime.fromtimestamp(started_at_ts + dur).strftime("%H:%M:%S")
  summary = SegmentSummary(
    segment=segment_name,
    route=route or re.sub(r"--\d+$", "", segment_name),
    segment_path=segment_path,
    commit=str(meta.get("commit", "")),
    branch=str(meta.get("branch", "")),
    dirty=bool(meta.get("dirty", False)),
    frames=len(frames),
    params=dict(meta.get("params") or {}),
    started_at=started_at,
    ended_at=ended_at,
    started_at_ts=started_at_ts,
  )
  if not frames:
    return summary

  summary.duration_s = round(max(0.0, frames[-1].t - frames[0].t), 1)
  summary.active_frames = sum(1 for f in frames if f.long_active and not f.gas_pressed and not f.brake_pressed)
  summary.lat_active_frames = sum(1 for f in frames if f.lat_active and not f.steering_pressed)
  summary.v_min_kph = round(min(f.v_kph for f in frames), 1)
  summary.v_max_kph = round(max(f.v_kph for f in frames), 1)
  summary.a_ego_min = round(min(f.a_ego for f in frames), 2)
  summary.a_ego_max = round(max(f.a_ego for f in frames), 2)
  lat_mags = [abs(f.lat_accel) for f in frames if f.lat_active and not f.blinker_active and not f.lane_changing]
  summary.lat_accel_max = round(max(lat_mags), 2) if lat_mags else 0.0
  summary.steer_sat_frames = sum(
    1 for f in frames if f.lat_active and not f.steering_pressed and f.steer_saturated and f.v_kph >= 15.0
  )
  curve_rates = [
    f.steer_rate_deg ** 2
    for f in frames
    if f.lat_active and not f.steering_pressed and not f.blinker_active and not f.lane_changing
    and f.v_kph >= 20.0 and abs(f.desired_lat_accel) >= 0.7
  ]
  summary.curve_steer_rate_rms = round(math.sqrt(sum(curve_rates) / len(curve_rates)), 1) if curve_rates else 0.0

  # 1) auto_hard_brake
  hard_hits: list[dict[str, Any]] = []
  for i, f in enumerate(frames):
    active = f.long_active and not f.gas_pressed and not f.brake_pressed
    if not active or f.v_kph < 3.0:
      continue
    da = (f.a_target - frames[i - 2].a_target) if (i >= 2 and frames[i - 2].long_active and (f.t - frames[i - 2].t) <= 0.20) else 0.0
    hard_decel = f.a_ego <= -2.3 or (f.a_target <= -2.3 and f.a_ego <= -1.8)
    sharp_drop = da <= -0.60 and f.a_target <= -0.45
    if hard_decel or sharp_drop:
      peak_neg = max(-f.a_ego, -f.a_target, abs(da) * 2.0)
      sev = "high" if (f.a_ego <= -2.8 or f.a_target <= -2.8) else "medium"
      hard_hits.append({
        "t": f.t,
        "v_kph": f.v_kph,
        "score": peak_neg,
        "severity": sev,
        "summary": (
          f"자동주행 급감속 (aEgo={f.a_ego:+.2f}, aTarget={f.a_target:+.2f}, "
          f"d={f.lead_d:.1f}m, vr={f.lead_v_rel:+.1f}m/s)"
        ),
        "metrics": {
          "a_ego": round(f.a_ego, 2),
          "a_target": round(f.a_target, 2),
          "a_base": round(f.a_base, 2),
          "da_target": round(da, 2),
          "lead_status": f.lead_status,
          "lead_radar": f.lead_radar,
          "lead_d": round(f.lead_d, 1),
          "lead_v_rel": round(f.lead_v_rel, 1),
          "lead_prob": round(f.lead_prob, 2),
        },
      })

  # 2) manual_brake_override
  manual_hits: list[dict[str, Any]] = []
  last_active_idx: int | None = None
  for i, f in enumerate(frames):
    if f.long_active and not f.brake_pressed:
      last_active_idx = i
    if f.brake_pressed and last_active_idx is not None and (f.t - frames[last_active_idx].t) <= 0.50 and f.v_kph >= 10.0:
      pre_idx = max(0, i - 20)
      pre = frames[pre_idx]
      # Find min aEgo over the next 2s while braking
      end_j = i
      peak_a = f.a_ego
      while end_j < len(frames) and (frames[end_j].t - f.t) <= 2.5 and frames[end_j].brake_pressed:
        peak_a = min(peak_a, frames[end_j].a_ego)
        end_j += 1
      manual_hits.append({
        "t": f.t,
        "v_kph": f.v_kph,
        "score": max(0.5, -peak_a),
        "severity": "high" if peak_a <= -2.5 else "medium",
        "summary": (
          f"자동주행 중 브레이크 개입 (peak aEgo={peak_a:+.2f}, 개입시 aTarget={pre.a_target:+.2f}, "
          f"선행차={'R' if pre.lead_radar else ('V' if pre.lead_status else '-')} {pre.lead_d:.1f}m)"
        ),
        "metrics": {
          "peak_manual_a_ego": round(peak_a, 2),
          "pre_a_target": round(pre.a_target, 2),
          "pre_a_base": round(pre.a_base, 2),
          "pre_lead_status": pre.lead_status,
          "pre_lead_radar": pre.lead_radar,
          "pre_lead_d": round(pre.lead_d, 1),
          "pre_lead_v_rel": round(pre.lead_v_rel, 1),
          "pre_lead_prob": round(pre.lead_prob, 2),
        },
      })
      last_active_idx = None

  # 3 & 4) low / mid speed surges
  low_surges = _detect_surge_events(
    frames, category="low_speed_surge", v_lo=1.0, v_hi=30.0,
    segment=segment_name, route=summary.route, segment_path=segment_path,
  )
  mid_surges = _detect_surge_events(
    frames, category="mid_speed_surge", v_lo=30.0, v_hi=65.0,
    segment=segment_name, route=summary.route, segment_path=segment_path,
  )

  # 5) far_lead_late_decel: sustained positive accel toward confirmed slower far lead
  far_hits: list[dict[str, Any]] = []
  streak_start: int | None = None
  for i, f in enumerate(frames):
    active = f.long_active and not f.gas_pressed and not f.brake_pressed
    lead_ok = f.lead_status and (f.lead_radar or f.lead_prob >= 0.55)
    far_accel_cond = (
      active
      and f.v_kph >= 50.0
      and lead_ok
      and f.lead_d >= 40.0
      and f.lead_v_rel <= -2.2
      and f.a_target >= 0.15
    )
    if far_accel_cond:
      if streak_start is None:
        streak_start = i
      dur = f.t - frames[streak_start].t
      if dur >= 0.45:
        far_hits.append({
          "t": f.t,
          "v_kph": f.v_kph,
          "score": abs(f.lead_v_rel) * f.a_target + 1.0,
          "severity": "high" if (f.a_target >= 0.45 and f.lead_v_rel <= -3.5) else "medium",
          "summary": (
            f"원거리 느린 선행차({'레이더' if f.lead_radar else f'비전 p={f.lead_prob:.2f}'}, "
            f"d={f.lead_d:.1f}m, vr={f.lead_v_rel:+.1f}m/s) 인지 중 가속 지속 (aTarget={f.a_target:+.2f})"
          ),
          "metrics": {
            "lead_radar": f.lead_radar,
            "lead_prob": round(f.lead_prob, 2),
            "lead_d": round(f.lead_d, 1),
            "lead_v_rel": round(f.lead_v_rel, 1),
            "a_target": round(f.a_target, 2),
            "mpc_source": f.mpc_source,
            "streak_s": round(dur, 2),
          },
        })
    else:
      streak_start = None

  # 6) manual_steer_override: driver overrides steering shortly after active auto steer
  steer_manual_hits: list[dict[str, Any]] = []
  last_lat_idx: int | None = None
  for i, f in enumerate(frames):
    if f.lat_active and not f.steering_pressed and not f.blinker_active and not f.lane_changing:
      last_lat_idx = i
    if (f.steering_pressed and not f.blinker_active and not f.lane_changing
        and last_lat_idx is not None and (f.t - frames[last_lat_idx].t) <= 0.50 and f.v_kph >= 15.0):
      pre = frames[max(0, i - 15)]
      end_j = i
      peak_torque = abs(f.steer_torque)
      while end_j < len(frames) and (frames[end_j].t - f.t) <= 2.5 and frames[end_j].steering_pressed:
        peak_torque = max(peak_torque, abs(frames[end_j].steer_torque))
        end_j += 1
      angle_err = pre.steer_angle_desired_deg - pre.steer_angle_deg
      off = pre.lane_offset_m
      sev = (
        "high"
        if (abs(pre.lat_accel) >= 2.0 or pre.steer_saturated or (off is not None and abs(off) >= 0.45))
        else "medium"
      )
      steer_manual_hits.append({
        "t": f.t,
        "v_kph": f.v_kph,
        "score": abs(pre.lat_accel) + (1.0 if pre.steer_saturated else 0.0) + (abs(off) if off is not None else 0.0),
        "severity": sev,
        "summary": (
          f"수동 조향 개입 (횡가속={pre.lat_accel:+.2f}m/s², 조향오차={angle_err:+.1f}°, "
          f"토크포화={int(pre.steer_saturated)}, 차선편차={f'{off:+.2f}m' if off is not None else '-'})"
        ),
        "metrics": {
          "pre_lat_accel": round(pre.lat_accel, 2),
          "pre_desired_lat_accel": round(pre.desired_lat_accel, 2),
          "pre_angle_err_deg": round(angle_err, 2),
          "pre_saturated": pre.steer_saturated,
          "pre_lane_offset_m": round(off, 2) if off is not None else None,
          "peak_driver_torque": round(peak_torque, 1),
        },
      })
      last_lat_idx = None

  # 7) steer_pingpong: rapid steering-rate or tracking-error reversals ("와리가리" / oscillation)
  pingpong_hits: list[dict[str, Any]] = []
  rate_signs: list[tuple[float, int, float]] = []
  err_signs: list[tuple[float, int, float]] = []
  for i, f in enumerate(frames):
    lat_clean = f.lat_active and not f.steering_pressed and not f.blinker_active and not f.lane_changing
    if not lat_clean or f.v_kph < 20.0:
      rate_signs.clear()
      err_signs.clear()
      continue
    is_curve = abs(f.desired_lat_accel) >= 0.7 or abs(f.lat_accel) >= 0.7
    rate_thr = 10.0 if is_curve else 6.0
    win_s = 4.5 if is_curve else 4.0
    rsign = 1 if f.steer_rate_deg > rate_thr else (-1 if f.steer_rate_deg < -rate_thr else 0)
    if rsign != 0:
      if not rate_signs or rate_signs[-1][1] != rsign:
        rate_signs.append((f.t, rsign, f.steer_angle_deg))
      while rate_signs and (f.t - rate_signs[0][0]) > win_s:
        rate_signs.pop(0)
      flips = max(0, len(rate_signs) - 1)
      swing = max(x[2] for x in rate_signs) - min(x[2] for x in rate_signs) if rate_signs else 0.0
      if flips >= 4 and swing >= (5.0 if is_curve else 3.0):
        pingpong_hits.append({
          "t": f.t,
          "v_kph": f.v_kph,
          "score": flips + swing * 0.25,
          "severity": "high" if (flips >= 5 or swing >= 10.0) else "medium",
          "summary": (
            f"{'코너' if is_curve else '직선'} 조향 와리가리 "
            f"(조향속도 반전 {flips}회/{win_s:.1f}s, 조향각 진폭 {swing:.1f}°)"
          ),
          "metrics": {
            "mode": "curve" if is_curve else "straight",
            "rate_flips": flips,
            "angle_swing_deg": round(swing, 1),
            "lat_accel": round(f.lat_accel, 2),
            "desired_lat_accel": round(f.desired_lat_accel, 2),
          },
        })

    if is_curve:
      angle_err = f.steer_angle_desired_deg - f.steer_angle_deg
      esign = 1 if angle_err > 2.0 else (-1 if angle_err < -2.0 else 0)
      if esign != 0:
        if not err_signs or err_signs[-1][1] != esign:
          err_signs.append((f.t, esign, angle_err))
        while err_signs and (f.t - err_signs[0][0]) > 4.5:
          err_signs.pop(0)
        eflips = max(0, len(err_signs) - 1)
        if eflips >= 3:
          pingpong_hits.append({
            "t": f.t,
            "v_kph": f.v_kph,
            "score": eflips + 1.5,
            "severity": "high" if eflips >= 4 else "medium",
            "summary": f"코너 조향 추종오차 반전 {eflips}회 (목표-실제 조향각 진동, lat={f.lat_accel:+.2f}m/s²)",
            "metrics": {
              "mode": "curve_error",
              "err_flips": eflips,
              "angle_err_deg": round(angle_err, 2),
              "lat_accel": round(f.lat_accel, 2),
            },
          })

  # 8) curve_lat_overshoot: high lateral accel, sustained torque saturation, or corner lane drift
  curve_over_hits: list[dict[str, Any]] = []
  sat_start: float | None = None
  for f in frames:
    lat_clean = f.lat_active and not f.steering_pressed and not f.blinker_active and not f.lane_changing
    if not lat_clean or f.v_kph < 25.0:
      sat_start = None
      continue
    if f.steer_saturated and abs(f.desired_lat_accel) >= 1.5:
      if sat_start is None:
        sat_start = f.t
    else:
      sat_start = None

    sat_dur = (f.t - sat_start) if sat_start is not None else 0.0
    high_lat = abs(f.lat_accel) >= 2.6
    sat_long = sat_dur >= 0.50
    lane_drift = abs(f.desired_lat_accel) >= 1.2 and f.lane_offset_m is not None and abs(f.lane_offset_m) >= 0.45
    if high_lat or sat_long or lane_drift:
      off_val = f.lane_offset_m
      curve_over_hits.append({
        "t": f.t,
        "v_kph": f.v_kph,
        "score": abs(f.lat_accel) + (1.0 if sat_long else 0.0) + (abs(off_val) if off_val is not None else 0.0),
        "severity": "high" if (abs(f.lat_accel) >= 3.1 or (sat_long and abs(f.lat_accel) >= 2.2)) else "medium",
        "summary": (
          f"코너 횡가속 과다/포화 (lat={f.lat_accel:+.2f}m/s², 목표={f.desired_lat_accel:+.2f}, "
          f"포화={sat_dur:.1f}s, 편차={f'{off_val:+.2f}m' if off_val is not None else '-'})"
        ),
        "metrics": {
          "lat_accel": round(f.lat_accel, 2),
          "desired_lat_accel": round(f.desired_lat_accel, 2),
          "sat_dur_s": round(sat_dur, 2),
          "lane_offset_m": round(off_val, 2) if off_val is not None else None,
        },
      })

  # 9) lane_center_bias: sustained one-sided lane offset on straight/mild roads
  bias_hits: list[dict[str, Any]] = []
  bias_start_t: float | None = None
  bias_sign = 0
  bias_vals: list[float] = []
  for f in frames:
    lat_clean = f.lat_active and not f.steering_pressed and not f.blinker_active and not f.lane_changing
    straight_ok = (
      lat_clean
      and f.v_kph >= 30.0
      and abs(f.curvature) < 0.0025
      and abs(f.lat_accel) < 0.8
      and f.lane_offset_m is not None
      and abs(f.lane_offset_m) >= 0.32
    )
    if straight_ok and f.lane_offset_m is not None:
      s = 1 if f.lane_offset_m > 0 else -1
      if bias_start_t is None or s != bias_sign:
        bias_start_t = f.t
        bias_sign = s
        bias_vals = [f.lane_offset_m]
      else:
        bias_vals.append(f.lane_offset_m)
      dur = f.t - bias_start_t
      if dur >= 2.5:
        mean_off = sum(bias_vals) / len(bias_vals)
        bias_hits.append({
          "t": f.t,
          "v_kph": f.v_kph,
          "score": abs(mean_off) * 3.0 + dur * 0.2,
          "severity": "high" if abs(mean_off) >= 0.45 else "medium",
          "summary": f"직선 차선 편향 ({'좌측' if mean_off > 0 else '우측'} 평균 {abs(mean_off):.2f}m 치우침, {dur:.1f}s 지속)",
          "metrics": {
            "mean_offset_m": round(mean_off, 2),
            "duration_s": round(dur, 1),
          },
        })
    else:
      bias_start_t = None
      bias_sign = 0
      bias_vals = []

  anomalies = (
    _merge_hits(hard_hits, category="auto_hard_brake", segment=segment_name, route=summary.route, segment_path=segment_path)
    + _merge_hits(manual_hits, category="manual_brake_override", segment=segment_name, route=summary.route, segment_path=segment_path, cooldown_s=3.0)
    + low_surges
    + mid_surges
    + _merge_hits(far_hits, category="far_lead_late_decel", segment=segment_name, route=summary.route, segment_path=segment_path, cooldown_s=3.0)
    + _merge_hits(steer_manual_hits, category="manual_steer_override", segment=segment_name, route=summary.route, segment_path=segment_path, cooldown_s=3.0)
    + _merge_hits(pingpong_hits, category="steer_pingpong", segment=segment_name, route=summary.route, segment_path=segment_path, cooldown_s=3.5)
    + _merge_hits(curve_over_hits, category="curve_lat_overshoot", segment=segment_name, route=summary.route, segment_path=segment_path, cooldown_s=3.0)
    + _merge_hits(bias_hits, category="lane_center_bias", segment=segment_name, route=summary.route, segment_path=segment_path, cooldown_s=4.0)
  )
  anomalies.sort(key=lambda e: (0 if e.severity == "high" else 1, e.t_start))
  summary.anomalies = anomalies
  summary.t_offset_s = round(frames[0].t, 2)
  summary.telemetry = _extract_compact_telemetry(frames, anomalies)
  return summary


def _extract_compact_telemetry(
  frames: Sequence[FrameSample],
  anomalies: Sequence[AnomalyEvent],
  *,
  stride: int = 10,
) -> dict[str, list[Any]]:
  """Downsample 20Hz FrameSamples to ~2Hz while preserving anomaly bounds and extrema."""
  n = len(frames)
  if n == 0:
    return {}
  keep: set[int] = set(range(0, n, max(1, stride)))
  keep.add(n - 1)

  # Preserve global longitudinal/lateral extrema so spikes are never smoothed away
  keep.add(min(range(n), key=lambda i: frames[i].a_ego))
  keep.add(max(range(n), key=lambda i: frames[i].a_ego))
  keep.add(max(range(n), key=lambda i: abs(frames[i].lat_accel)))

  # Preserve closest samples to each anomaly start/end
  t_arr = [f.t for f in frames]
  import bisect
  for a in anomalies:
    for target_t in (a.t_start, a.t_end):
      pos = bisect.bisect_left(t_arr, target_t)
      if pos < n:
        keep.add(pos)
      if pos > 0:
        keep.add(pos - 1)

  indices = sorted(keep)
  out: dict[str, list[Any]] = {
    "t": [], "v": [], "ae": [], "ab": [], "at": [],
    "ld": [], "lv": [], "la": [], "dl": [],
    "sa": [], "ds": [], "lo": [], "fl": [],
  }
  for i in indices:
    f = frames[i]
    fl = (
      (1 if f.long_active else 0)
      | (2 if f.lat_active else 0)
      | (4 if f.gas_pressed else 0)
      | (8 if f.brake_pressed else 0)
      | (16 if f.steering_pressed else 0)
      | (32 if f.steer_saturated else 0)
      | (64 if (f.blinker_active or f.lane_changing) else 0)
    )
    out["t"].append(round(f.t, 2))
    out["v"].append(round(f.v_kph, 1))
    out["ae"].append(round(f.a_ego, 2))
    out["ab"].append(round(f.a_base, 2))
    out["at"].append(round(f.a_target, 2))
    out["ld"].append(round(f.lead_d, 1) if (f.lead_status and f.lead_d > 0) else None)
    out["lv"].append(round(f.lead_v * 3.6, 1) if f.lead_status else None)
    out["la"].append(round(f.lat_accel, 2))
    out["dl"].append(round(f.desired_lat_accel, 2))
    out["sa"].append(round(f.steer_angle_deg, 1))
    out["ds"].append(round(f.steer_angle_desired_deg, 1))
    out["lo"].append(round(f.lane_offset_m, 3) if f.lane_offset_m is not None else None)
    out["fl"].append(fl)
  return out


def _scan_one_file(path_str: str) -> SegmentSummary:
  p = Path(path_str)
  seg, route = _infer_segment_and_route(p)
  try:
    meta, frames = load_segment_frames(p)
    return analyze_frames(seg, frames, segment_path=str(p), route=route, metadata=meta)
  except Exception as e:
    return SegmentSummary(segment=seg, route=route, segment_path=str(p), error=str(e))


def load_drive_reports(reports_dir: Path | None) -> list[dict[str, Any]]:
  if reports_dir is None or not reports_dir.is_dir():
    return []
  reports: list[dict[str, Any]] = []
  for p in sorted(reports_dir.glob("*.json"), reverse=True):
    if p.name == "index.json":
      continue
    try:
      data = json.loads(p.read_text(encoding="utf-8"))
      if isinstance(data, dict) and "report_id" in data:
        reports.append(data)
    except Exception:
      continue
  reports.sort(key=lambda r: float(r.get("ended_at", 0.0) or 0.0), reverse=True)
  return reports


def _match_reports_for_segments(
  segments: Sequence[SegmentSummary],
  drive_reports: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
  """Keep only drive reports overlapping the scanned segments' time window when timestamps exist."""
  ts_ranges = [
    (s.started_at_ts, s.started_at_ts + max(0.0, s.duration_s))
    for s in segments
    if not s.error and s.started_at_ts > 1e9
  ]
  if not ts_ranges or not drive_reports:
    return list(drive_reports)
  seg_min = min(r[0] for r in ts_ranges)
  seg_max = max(r[1] for r in ts_ranges)
  matched: list[tuple[float, dict[str, Any]]] = []
  for r in drive_reports:
    r_end = float(r.get("ended_at") or 0.0)
    r_start = float(r.get("started_at") or r_end)
    if r_end <= 1e9:
      continue
    if r_start <= seg_max + 360.0 and r_end >= seg_min - 360.0:
      matched.append((abs(r_end - seg_max), r))
  matched.sort(key=lambda item: item[0])
  return [item[1] for item in matched]


def render_scan_markdown(report: ScanReport) -> str:
  lines: list[str] = ["# 1차 주행 로그 이상치 스캔 리포트", ""]
  total_frames = sum(s.frames for s in report.segments)
  total_active = sum(s.active_frames for s in report.segments)
  total_lat_active = sum(s.lat_active_frames for s in report.segments)
  lat_cats = {"manual_steer_override", "steer_pingpong", "curve_lat_overshoot", "lane_center_bias"}
  lat_count = sum(1 for a in report.anomalies if a.category in lat_cats)
  long_count = len(report.anomalies) - lat_count
  lines.append(
    f"- 분석 세그먼트: **{len(report.segments)}개** "
    f"(총 {total_frames} 프레임 / 종방향 활성 {total_active} / 조향 활성 {total_lat_active})"
  )
  lines.append(
    f"- 검출된 이상치: **{len(report.anomalies)}건** "
    f"(높음 {sum(1 for a in report.anomalies if a.severity == 'high')}건 | 가감속 {long_count}건 / 조향 {lat_count}건)"
  )
  lines.append("")

  if report.drive_reports:
    lines.append("## 최근 저장 주행리포트 요약")
    lines.append("| Report ID | 점수/등급 | 거리 | 급정거(자동/수동) |")
    lines.append("|---|---:|---:|---:|")
    for r in report.drive_reports[:5]:
      lines.append(
        f"| `{r.get('report_id')}` | {r.get('score')}점 ({r.get('grade')}) | "
        f"{r.get('distance_km', 0):.1f} km | {r.get('auto_hard_brake_count', r.get('hard_brake_count', 0))} / {r.get('manual_hard_brake_count', 0)} |"
      )
    lines.append("")

  lines.append("## 세그먼트별 요약")
  lines.append("| 세그먼트 | 발생일시 | 커밋 | 속도(km/h) | aEgo(m/s²) | 최대횡가속 | 코너조향RMS | 토크포화 | 이상치 |")
  lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
  for s in report.segments:
    dt_str = f"{s.started_at}~{s.ended_at}" if (s.started_at and s.ended_at) else (s.started_at or "-")
    if s.error:
      lines.append(f"| `{s.segment}` | {dt_str} | - | - | - | - | - | - | 오류: {s.error} |")
      continue
    dirty_tag = "*" if s.dirty else ""
    lines.append(
      f"| `{s.segment}` | {dt_str} | `{s.commit[:8]}{dirty_tag}` | {s.v_min_kph:.0f}..{s.v_max_kph:.0f} | "
      f"{s.a_ego_min:+.2f}..{s.a_ego_max:+.2f} | {s.lat_accel_max:.2f} | {s.curve_steer_rate_rms:.1f}°/s | {s.steer_sat_frames} | {len(s.anomalies)} |"
    )
  lines.append("")

  lines.append("## 주요 이상치 상세 (우선순위 정렬)")
  if not report.anomalies:
    lines.append("- 검출된 주요 이상치가 없습니다.")
  else:
    for idx, a in enumerate(report.anomalies[:30], 1):
      lines.append(
        f"{idx}. **[{a.severity.upper()}] `{a.category}`** — `{a.segment}` "
        f"`t={a.t_start:.1f}~{a.t_end:.1f}s` ({a.v_kph:.1f} km/h): {a.summary}"
      )
      if a.segment_path:
        lines.append(f"   - 파일: `{a.segment_path}`")
  lines.append("")
  return "\n".join(lines)


def scan_paths(
  paths: Sequence[Path | str],
  *,
  reports_dir: Path | None = None,
  out_dir: Path | None = None,
  workers: int = 4,
  custom_notes: dict[str, Any] | None = None,
  render_html: bool = True,
  route_groups: list[dict[str, Any]] | None = None,
  active_route: str = "",
  device_label: str = "",
  device_error: str = "",
) -> ScanReport:
  rlogs = discover_rlogs(paths)
  summaries: list[SegmentSummary] = []

  if len(rlogs) <= 1 or workers <= 1:
    for p in rlogs:
      summaries.append(_scan_one_file(str(p)))
  else:
    by_path: dict[str, SegmentSummary] = {}
    with ProcessPoolExecutor(max_workers=min(int(workers), len(rlogs))) as pool:
      futs = {pool.submit(_scan_one_file, str(p)): str(p) for p in rlogs}
      for fut in as_completed(futs):
        ps = futs[fut]
        by_path[ps] = fut.result()
    summaries = [by_path[str(p)] for p in rlogs]

  all_anomalies: list[AnomalyEvent] = []
  for s in summaries:
    all_anomalies.extend(s.anomalies)
  all_anomalies.sort(key=lambda a: (0 if a.severity == "high" else 1, a.category, a.segment, a.t_start))

  drive_reports = _match_reports_for_segments(summaries, load_drive_reports(reports_dir))
  report = ScanReport(segments=summaries, anomalies=all_anomalies, drive_reports=drive_reports)

  if out_dir is not None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "scan_summary.json").write_text(
      json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
      encoding="utf-8",
    )
    (out_path / "scan_report.md").write_text(render_scan_markdown(report), encoding="utf-8")
    if render_html:
      from .html_report import render_html_report
      groups_payload = route_groups
      if groups_payload is None:
        try:
          from .sync import list_route_groups
          local_groups, dev_lbl, dev_e = list_route_groups(limit=15, local_only=True)
          groups_payload = [g.to_dict() for g in local_groups]
          if not device_label:
            device_label = dev_lbl
          if not device_error:
            device_error = dev_e
        except Exception:
          groups_payload = []
      inferred_route = active_route or (summaries[0].route if len({s.route for s in summaries if s.route}) == 1 else "")
      (out_path / "analysis_report.html").write_text(
        render_html_report(
          report,
          custom_notes=custom_notes,
          route_groups=groups_payload,
          active_route=inferred_route,
          device_label=device_label,
          device_error=device_error,
        ),
        encoding="utf-8",
      )

  return report

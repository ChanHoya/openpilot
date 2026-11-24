"""Human-like longitudinal shaping: gap buffering, coast-first transitions, and smooth jerk control."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY, CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import read_curve_inner_bias
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import desired_follow_distance
from openpilot.selfdrive.modeld.constants import ModelConstants


CURVE_ENTRY_LOOKAHEAD_S = 6.5
CURVE_ENTRY_MIN_T_S = 0.45
CURVE_ENTRY_MAX_DIST_M = 180.0
CURVE_ENTRY_MIN_MODEL_SPEED = 3.0
CURVE_ENTRY_MIN_CURVATURE = 0.0018
CURVE_ENTRY_MIN_RAW_LAT_ACCEL = 0.60
CURVE_ENTRY_MAX_ROLL_RAD = 0.15
CURVE_ENTRY_RESPONSE_S = 1.1
CURVE_ENTRY_ONSET_JERK = 0.55
CURVE_ENTRY_SEVERE_ONSET_JERK = 1.35
CURVE_ENTRY_RELEASE_JERK = 0.65
CURVE_ENTRY_ENGAGE_DEADBAND = 0.08
CURVE_ENTRY_RELEASE_HOLD_S = 0.25


@dataclass(frozen=True)
class ShaperResult:
  a_target: float
  v_target_now: float
  j_target_now: float
  shaped: bool
  is_urgent: bool
  a_coast: float
  curve_decel: float = 0.0


def _curve_entry_thresholds(curve_inner_bias: float, curve_speed_factor: float) -> tuple[float, float, float, float]:
  """Compute roll-adjusted lateral onset threshold and progressive curve-entry deceleration caps."""
  factor = float(np.clip(curve_speed_factor, 0.5, 2.5)) if math.isfinite(curve_speed_factor) else 1.2
  bias = float(np.clip(curve_inner_bias, 0.0, 100.0)) if math.isfinite(curve_inner_bias) else 10.0
  bias_boost = float(np.interp(bias, [0.0, 10.0, 20.0], [0.0, 1.0, 1.25]))
  lat_start = float(np.clip((1.42 - 0.14 * bias_boost) / factor, 0.85, 1.48))
  mild_cap = float(np.interp(bias, [0.0, 10.0, 25.0], [0.38, 0.45, 0.52]))
  high_cap = float(np.interp(bias, [0.0, 10.0, 25.0], [0.85, 1.00, 1.15]))
  firm_cap = float(np.interp(bias, [0.0, 10.0, 25.0], [1.35, 1.55, 1.70]))
  return lat_start, mild_cap, high_cap, firm_cap


def compute_curve_entry_decel_target(
  model_msg,
  v_ego: float,
  *,
  roll: float = 0.0,
  current_curvature: float = 0.0,
  curve_inner_bias: float = 10.0,
  curve_speed_factor: float = 1.2,
  lower_limit_kph: float = 30.0,
  turn_speed_control_mode: int = 1,
  lane_change_active: bool = False,
  blinker_active: bool = False,
) -> tuple[float, float]:
  """Compute gentle roll-compensated corner-entry decel magnitude (m/s^2) and current lat accel."""
  if int(turn_speed_control_mode) == 0 or lane_change_active or blinker_active or model_msg is None:
    return 0.0, 0.0
  if not math.isfinite(v_ego) or v_ego < 8.0:
    return 0.0, 0.0

  v_kph = v_ego * CV.MS_TO_KPH
  floor_kph = float(np.clip(lower_limit_kph, 10.0, 120.0)) if math.isfinite(lower_limit_kph) else 30.0
  speed_gate = float(np.clip(np.interp(v_kph, [30.0, 42.0], [0.0, 1.0]), 0.0, 1.0))
  floor_gate = float(np.clip(np.interp(v_kph, [floor_kph, floor_kph + 6.0], [0.0, 1.0]), 0.0, 1.0))
  gate = speed_gate * floor_gate
  if gate <= 1e-4:
    return 0.0, 0.0

  try:
    values = (
      model_msg.position.x,
      model_msg.position.y,
      model_msg.position.z,
      model_msg.velocity.x,
      model_msg.orientationRate.z,
    )
    if any(len(v) != ModelConstants.IDX_N for v in values):
      return 0.0, 0.0
    end = int(np.searchsorted(ModelConstants.T_IDXS, CURVE_ENTRY_LOOKAHEAD_S, side="right"))
    if end < 3:
      return 0.0, 0.0
    t_arr = np.asarray(ModelConstants.T_IDXS[:end], dtype=float)
    pos = np.asarray(values[:3], dtype=float)[:, :end]
    vel = np.asarray(values[3], dtype=float)[:end]
    yaw_rate = np.asarray(values[4], dtype=float)[:end]
  except (AttributeError, TypeError, ValueError):
    return 0.0, 0.0

  if not np.all(np.isfinite(pos)):
    return 0.0, 0.0

  dist = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pos, axis=1), axis=0))]
  max_dist = min(CURVE_ENTRY_MAX_DIST_M, max(25.0, v_ego * CURVE_ENTRY_LOOKAHEAD_S))
  valid = np.isfinite(vel) & np.isfinite(yaw_rate) & (vel >= CURVE_ENTRY_MIN_MODEL_SPEED)
  curv = np.divide(yaw_rate, vel, out=np.zeros_like(yaw_rate), where=valid)

  if math.isfinite(current_curvature) and abs(current_curvature) > 1e-6:
    k_now = float(current_curvature)
  else:
    near_mask = (t_arr <= 0.4) & valid
    k_now = float(np.median(curv[near_mask])) if np.any(near_mask) else 0.0

  lat_start, mild_cap, high_cap, firm_cap = _curve_entry_thresholds(curve_inner_bias, curve_speed_factor)
  lat_mid = lat_start + 0.45
  lat_high = lat_start + 0.95
  lat_severe = lat_start + 1.60
  roll_safe = float(np.clip(roll, -CURVE_ENTRY_MAX_ROLL_RAD, CURVE_ENTRY_MAX_ROLL_RAD)) if math.isfinite(roll) else 0.0

  best_decel = 0.0
  peak_future_curv = 0.0
  peak_eff_lat = 0.0
  for i in range(end):
    if t_arr[i] < CURVE_ENTRY_MIN_T_S or dist[i] > max_dist:
      continue
    s = min(max(i - 1, 0), end - 3)
    nb = slice(s, s + 3)
    if not np.all(valid[nb]) or dist[s + 2] - dist[s] < 0.05:
      continue
    k_i = float(np.median(curv[nb]))
    if t_arr[i] <= 3.0:
      peak_future_curv = max(peak_future_curv, abs(k_i))
    if abs(k_i) < CURVE_ENTRY_MIN_CURVATURE or abs(k_i) * (v_ego ** 2) < CURVE_ENTRY_MIN_RAW_LAT_ACCEL:
      continue

    roll_w = 0.0 if (abs(k_now) > 0.0015 and (k_i * k_now) < 0.0) else float(np.interp(t_arr[i], [1.0, 3.0], [1.0, 0.35]))
    signed_eff = (v_ego ** 2) * k_i - ACCELERATION_DUE_TO_GRAVITY * roll_safe * roll_w
    a_lat_eff = max(0.0, math.copysign(1.0, k_i) * signed_eff)
    if a_lat_eff <= lat_start:
      continue

    peak_eff_lat = max(peak_eff_lat, a_lat_eff)
    eff_curv = a_lat_eff / max(v_ego ** 2, 1.0)
    v_target = max(floor_kph * CV.KPH_TO_MS, math.sqrt(lat_start / max(eff_curv, 1e-6)))
    if v_ego <= v_target:
      continue

    d_rem = max(4.0, float(dist[i]) - v_ego * CURVE_ENTRY_RESPONSE_S)
    a_req = (v_ego ** 2 - v_target ** 2) / (2.0 * d_rem)
    sev_cap = float(np.interp(a_lat_eff, [lat_start, lat_mid, lat_high, lat_severe], [0.0, mild_cap, high_cap, firm_cap]))
    best_decel = max(best_decel, min(a_req, sev_cap))

  # Past the apex, taper entry decel as upcoming curvature unwinds below current curvature.
  if best_decel > 0.0 and abs(k_now) > 0.0025 and peak_future_curv < 0.75 * abs(k_now) and peak_eff_lat < lat_mid:
    best_decel *= float(np.interp(peak_future_curv / max(abs(k_now), 1e-4), [0.35, 0.75], [0.0, 1.0]))

  return float(best_decel * gate), float(abs(k_now) * (v_ego ** 2))


class CurveEntryDecel:
  """Stateful jerk-limited curve-entry deceleration with mid-curve hold and apex release."""

  def __init__(self, dt: float = DT_MDL):
    self.dt = max(1e-3, float(dt))
    self.decel = 0.0
    self._hold_timer = 0.0

  def reset(self) -> None:
    self.decel = 0.0
    self._hold_timer = 0.0

  def update(
    self,
    model_msg,
    v_ego: float,
    *,
    roll: float = 0.0,
    current_curvature: float = 0.0,
    curve_inner_bias: float = 10.0,
    curve_speed_factor: float = 1.2,
    lower_limit_kph: float = 30.0,
    turn_speed_control_mode: int = 1,
    lane_change_active: bool = False,
    blinker_active: bool = False,
  ) -> float:
    raw_decel, current_lat = compute_curve_entry_decel_target(
      model_msg,
      v_ego,
      roll=roll,
      current_curvature=current_curvature,
      curve_inner_bias=curve_inner_bias,
      curve_speed_factor=curve_speed_factor,
      lower_limit_kph=lower_limit_kph,
      turn_speed_control_mode=turn_speed_control_mode,
      lane_change_active=lane_change_active,
      blinker_active=blinker_active,
    )
    lat_start = _curve_entry_thresholds(curve_inner_bias, curve_speed_factor)[0]

    if self.decel < 1e-3 and raw_decel < CURVE_ENTRY_ENGAGE_DEADBAND:
      desired = 0.0
      self._hold_timer = 0.0
    elif raw_decel > 0.0:
      desired = raw_decel
      self._hold_timer = CURVE_ENTRY_RELEASE_HOLD_S
    elif self.decel > 0.0 and self._hold_timer > 0.0 and current_lat >= 0.9 * lat_start:
      desired = self.decel
      self._hold_timer = max(0.0, self._hold_timer - self.dt)
    else:
      desired = 0.0
      self._hold_timer = 0.0

    if desired > self.decel:
      onset_jerk = float(np.interp(desired, [0.45, 1.40], [CURVE_ENTRY_ONSET_JERK, CURVE_ENTRY_SEVERE_ONSET_JERK]))
      self.decel = min(desired, self.decel + onset_jerk * self.dt)
    else:
      self.decel = max(desired, self.decel - CURVE_ENTRY_RELEASE_JERK * self.dt)
    return self.decel


def natural_coast_accel(v_ego: float, pitch: float = 0.0) -> float:
  """Return empirical speed- and slope-aware natural coasting deceleration (m/s^2)."""
  v_safe = max(0.0, float(v_ego)) if math.isfinite(v_ego) else 0.0
  v_kph = v_safe * CV.MS_TO_KPH
  base = float(np.interp(v_kph, [0.0, 20.0, 50.0, 80.0, 110.0], [-0.12, -0.16, -0.20, -0.26, -0.34]))
  slope = float(np.clip(-math.sin(pitch) * 5.0, -0.18, 0.18)) if math.isfinite(pitch) else 0.0
  return float(np.clip(base + slope, -0.40, -0.12))


def compute_accel_ramp_targets(
  v_ego: float,
  base_max_accel: float,
  has_lead: bool,
  *,
  d_rel: float = 0.0,
  v_rel: float = 0.0,
  v_lead: float = 0.0,
  a_lead_filt: float = 0.0,
  t_follow: float = 1.3,
  comfort_brake: float = 2.5,
  stop_dist: float = 5.0,
) -> tuple[float, float, float, float, float]:
  """Compute pre-MPC accel cap, S-curve ramp target, tau, headroom, and launch floor."""
  v_safe = max(0.0, float(v_ego)) if math.isfinite(v_ego) else 0.0
  v_ego_kph = v_safe * CV.MS_TO_KPH
  base_max = max(0.0, float(base_max_accel)) if math.isfinite(base_max_accel) else 1.0

  if not has_lead:
    low_speed_cap = float(np.interp(v_ego_kph, [0.0, 15.0, 35.0, 50.0], [0.80, 1.20, 1.40, base_max]))
    target_a = min(base_max, low_speed_cap)
    return float(target_a), float(base_max), 1.60, 0.35, float(min(0.75, target_a))

  d_safe = max(0.0, float(d_rel)) if math.isfinite(d_rel) else 0.0
  v_rel_safe = float(v_rel) if math.isfinite(v_rel) else 0.0
  v_lead_safe = max(0.0, float(v_lead)) if math.isfinite(v_lead) else v_safe
  a_lead_safe = float(a_lead_filt) if math.isfinite(a_lead_filt) else 0.0

  desired_dist = max(
    float(stop_dist),
    float(desired_follow_distance(v_safe, v_lead_safe, comfort_brake, stop_dist, t_follow)),
  )
  d_headroom = d_safe - desired_dist
  lead_departing = bool(v_rel_safe > 0.25 or a_lead_safe > 0.30)

  low_speed_cap = float(np.interp(v_ego_kph, [0.0, 15.0, 35.0, 50.0], [1.05, 1.38, 1.48, base_max]))
  depart_catchup_cap = float(np.interp(v_ego_kph, [0.0, 15.0, 25.0, 38.0], [1.62, 1.66, 1.42, base_max]))
  depart_gate = float(np.clip(
    max(
      np.interp(d_safe, [2.8, 6.2], [0.0, 1.0]),
      np.interp(v_lead_safe * CV.MS_TO_KPH, [2.5, 13.0], [0.0, 1.0]),
    ),
    0.0,
    1.0,
  ))

  if lead_departing:
    # Restrict responsive launch boost to <25~38 km/h so mid/high-speed cruising avoids kickdown surges.
    depart_cap = float(np.interp(depart_gate, [0.0, 1.0], [1.32, depart_catchup_cap]))
    cruise_fade = float(np.clip(np.interp(v_ego_kph, [25.0, 38.0], [1.0, 0.0]), 0.0, 1.0))
    depart_boost = base_max + cruise_fade * max(0.0, depart_cap - base_max)
    allowed_max_accel = max(base_max, depart_boost)
    raw_target = min(allowed_max_accel, max(low_speed_cap, depart_boost))
  else:
    allowed_max_accel = base_max
    raw_target = min(base_max, low_speed_cap)

  gap_scale = float(np.interp(d_headroom, [-1.5, 2.5, 7.5], [0.15, 0.55, 1.0]))
  rel_scale = float(np.interp(v_rel_safe, [-1.2, 0.0, 1.2], [0.12, 0.50, 1.0])) if not lead_departing else 1.0
  far_free_w = float(np.clip(np.interp(d_headroom, [14.0, 28.0], [0.0, 1.0]), 0.0, 1.0))
  eff_rel_scale = rel_scale * (1.0 - far_free_w) + 1.0 * far_free_w

  if v_ego_kph < 25.0 and lead_departing:
    lead_match_a = float(np.clip(a_lead_safe * 1.15 + max(0.0, v_rel_safe) * 0.85, 0.0, raw_target))
    approach_scale = max(gap_scale, float(np.interp(d_safe, [2.8, 5.5], [0.72, 1.0])))
  else:
    match_gain = 0.90 if lead_departing else 0.75
    match_vrel_gain = 0.45 if lead_departing else 0.35
    lead_match_a = float(np.clip(a_lead_safe * match_gain + max(0.0, v_rel_safe) * match_vrel_gain, 0.0, raw_target))
    approach_scale = gap_scale if lead_departing else min(gap_scale, eff_rel_scale)

  target_a = lead_match_a + (raw_target - lead_match_a) * approach_scale
  if v_rel_safe < -0.2 and d_headroom < 4.5:
    closing_feather = float(np.interp(v_rel_safe, [-1.8, -0.2], [0.12, 0.85]))
    target_a *= closing_feather

  if v_ego_kph < 30.0 and lead_departing and depart_gate > 0.08:
    tau, headroom = 0.55, 0.85
  elif d_headroom > 28.0:
    free_w = float(np.clip(np.interp(d_headroom, [28.0, 48.0], [0.0, 1.0]), 0.0, 1.0))
    tau = float(1.55 * (1.0 - free_w) + 0.90 * free_w)
    headroom = float(0.35 * (1.0 - free_w) + 0.65 * free_w)
  elif lead_departing and (v_rel_safe > 0.45 or a_lead_safe > 0.45):
    tau, headroom = 1.55, 0.58
  else:
    tau, headroom = 1.55, 0.35

  if lead_departing and v_ego_kph < 30.0 and depart_gate > 0.08:
    floor_cap = 1.25
  elif (lead_departing and (v_rel_safe > 0.45 or a_lead_safe > 0.45)) or d_headroom > 28.0:
    floor_cap = 1.15
  else:
    floor_cap = 0.85
  start_floor = min(floor_cap, target_a)
  return float(target_a), float(allowed_max_accel), float(tau), float(headroom), float(start_floor)


class HumanLongitudinalShaper:
  """Shapes planner output toward progressive throttle, buffered gaps, and coast-first deceleration."""

  def __init__(self, dt: float = DT_MDL):
    self.dt = max(1e-3, float(dt))
    self.a_lead_filt = FirstOrderFilter(0.0, 0.35, self.dt)
    self.curve_entry = CurveEntryDecel(dt=self.dt)
    self.curve_inner_bias = 10.0
    self.auto_curve_speed_factor = 1.2
    self.auto_curve_speed_lower_limit = 30.0
    self.turn_speed_control_mode = 1
    self._param_timer = 1.0
    self._reset_lead_state(0.0)
    self._has_reset_anchor = False

  def _reset_lead_state(self, a_init: float = 0.0) -> None:
    a_safe = float(np.clip(a_init, -2.0, 1.5)) if math.isfinite(a_init) else 0.0
    self.a_lead_filt.x = 0.0
    self.a_lead_filtered = 0.0
    self._lead_active = False
    self.track_id = -1
    self.lead_age = 0.0
    self._prev_d_rel = 1e3
    self.prev_a = a_safe
    self.last_pos_age = 10.0
    self.coast_dwell = 0.0
    self.initialized = False
    self._has_reset_anchor = True

  def reset(self, a_init: float = 0.0) -> None:
    self._reset_lead_state(a_init)
    self.curve_entry.reset()

  def update_params(self, params) -> None:
    if params is None:
      return
    self._param_timer += self.dt
    if self._param_timer < 1.0:
      return
    self._param_timer = 0.0
    try:
      self.curve_inner_bias = float(read_curve_inner_bias(params))
    except Exception:
      self.curve_inner_bias = 10.0
    try:
      raw_factor = float(params.get_int("AutoCurveSpeedFactor")) * 0.01
      if math.isfinite(raw_factor) and raw_factor > 0.0:
        self.auto_curve_speed_factor = float(np.clip(raw_factor, 0.5, 2.5))
    except Exception:
      self.auto_curve_speed_factor = 1.2
    try:
      raw_limit = float(params.get_int("AutoCurveSpeedLowerLimit"))
      if math.isfinite(raw_limit):
        self.auto_curve_speed_lower_limit = float(np.clip(raw_limit, 10.0, 120.0))
    except Exception:
      self.auto_curve_speed_lower_limit = 30.0
    try:
      self.turn_speed_control_mode = int(params.get_int("TurnSpeedControlMode"))
    except Exception:
      self.turn_speed_control_mode = 1

  def update_lead_filter(self, lead_status: bool, a_lead: float, track_id: int) -> float:
    if not lead_status or not math.isfinite(a_lead):
      self._lead_active = False
      self.track_id = -1
      self.lead_age = 0.0
      self._prev_d_rel = 1e3
      self.a_lead_filt.x = 0.0
      self.a_lead_filtered = 0.0
      return 0.0

    tid = int(track_id)
    if not self._lead_active or tid != self.track_id:
      self._lead_active = True
      self.track_id = tid
      self.lead_age = 0.0
      self.a_lead_filt.x = float(np.clip(a_lead, -3.5, 2.0))
    else:
      self.a_lead_filt.update(float(np.clip(a_lead, -6.0, 3.0)))
    self.a_lead_filtered = float(self.a_lead_filt.x)
    return self.a_lead_filtered

  def compute_ramp_targets(
    self,
    v_ego: float,
    base_max_accel: float,
    has_lead: bool,
    *,
    d_rel: float = 0.0,
    v_rel: float = 0.0,
    v_lead: float = 0.0,
    t_follow: float = 1.3,
    comfort_brake: float = 2.5,
    stop_dist: float = 5.0,
  ) -> tuple[float, float, float, float, float]:
    """Delegate pre-MPC ramp target computation using the shaper's filtered lead acceleration."""
    return compute_accel_ramp_targets(
      v_ego,
      base_max_accel,
      has_lead,
      d_rel=d_rel,
      v_rel=v_rel,
      v_lead=v_lead,
      a_lead_filt=self.a_lead_filtered,
      t_follow=t_follow,
      comfort_brake=comfort_brake,
      stop_dist=stop_dist,
    )

  def shape(
    self,
    *,
    a_mpc: float,
    v_target_mpc: float,
    j_target_mpc: float,
    v_ego: float,
    a_ego: float,
    pitch: float,
    action_t: float,
    lead_status: bool,
    d_rel: float,
    v_rel: float,
    v_lead: float,
    a_lead: float,
    t_follow: float,
    comfort_brake: float,
    stop_dist: float,
    should_stop: bool,
    fcw: bool,
    reset_state: bool,
    mpc_mode: str = 'acc',
    predicted_danger_margin: float = 1e3,
    mpc_source: str = 'lead0',
    model_msg=None,
    roll: float = 0.0,
    current_curvature: float = 0.0,
    lane_change_active: bool = False,
    blinker_active: bool = False,
    lead_radar: bool = True,
  ) -> ShaperResult:
    v_safe = max(0.0, float(v_ego)) if math.isfinite(v_ego) else 0.0
    pitch_safe = float(pitch) if math.isfinite(pitch) else 0.0
    a_coast = natural_coast_accel(v_safe, pitch_safe)
    if reset_state:
      self.reset(a_ego)
      return ShaperResult(a_mpc, v_target_mpc, j_target_mpc, False, bool(fcw or should_stop), a_coast, 0.0)

    if model_msg is not None and not (fcw or should_stop):
      curve_decel = self.curve_entry.update(
        model_msg,
        v_safe,
        roll=roll,
        current_curvature=current_curvature,
        curve_inner_bias=self.curve_inner_bias,
        curve_speed_factor=self.auto_curve_speed_factor,
        lower_limit_kph=self.auto_curve_speed_lower_limit,
        turn_speed_control_mode=self.turn_speed_control_mode,
        lane_change_active=lane_change_active,
        blinker_active=blinker_active,
      )
    else:
      self.curve_entry.reset()
      curve_decel = 0.0

    if mpc_mode != 'acc':
      self._reset_lead_state(a_ego)
      if curve_decel > 1e-3 and a_mpc > -curve_decel:
        out_a = -curve_decel
        out_v = min(float(v_target_mpc), v_safe + out_a * float(action_t))
        return ShaperResult(out_a, out_v, float(j_target_mpc), True, bool(fcw or should_stop), a_coast, curve_decel)
      return ShaperResult(a_mpc, v_target_mpc, j_target_mpc, False, bool(fcw or should_stop), a_coast, curve_decel)

    if not self.initialized:
      if not self._has_reset_anchor:
        self.prev_a = float(a_mpc)
      self.initialized = True
      self._has_reset_anchor = False

    prev_before = float(self.prev_a)
    a_lead_f = float(self.a_lead_filtered)
    v_kph = v_safe * CV.MS_TO_KPH

    if lead_status and math.isfinite(d_rel) and d_rel > 0.1:
      d_safe = float(d_rel)
      v_rel_safe = float(v_rel) if math.isfinite(v_rel) else 0.0
      v_lead_safe = max(0.0, float(v_lead)) if math.isfinite(v_lead) else v_safe
      a_lead_safe = float(a_lead) if math.isfinite(a_lead) else 0.0
      if not self._lead_active or (self._prev_d_rel - d_safe) > 6.0:
        self.lead_age = 0.0
      else:
        self.lead_age = min(10.0, self.lead_age + self.dt)
      self._prev_d_rel = d_safe

      des_d = max(float(stop_dist), float(desired_follow_distance(v_safe, v_lead_safe, comfort_brake, stop_dist, t_follow)))
      d_headroom = d_safe - des_d
      phys_ref = max(float(stop_dist) + 2.0, 0.78 * (v_safe * float(t_follow) + float(stop_dist)))
      phys_margin = d_safe - phys_ref
      danger_ok = math.isfinite(predicted_danger_margin) and predicted_danger_margin < 2.5

      closing_v = max(0.0, -v_rel_safe)
      ttc = d_safe / closing_v if closing_v > 0.1 else 99.0
      d_steady = max(float(stop_dist) + 2.0, v_lead_safe * float(t_follow) + float(stop_dist))
      d_gap = d_safe - d_steady
      a_kin_close = -(closing_v ** 2) / (2.0 * max(3.5, d_gap)) if closing_v > 0.1 else 0.0
      if a_lead_f < -0.5 and d_safe < max(45.0, v_safe * 2.2):
        a_kin_eff = a_kin_close + 0.45 * (a_lead_f + 0.5)
      elif a_lead_f > 0.5:
        a_kin_eff = a_kin_close * float(np.interp(a_lead_f, [0.5, 1.8], [0.85, 0.45]))
      else:
        a_kin_eff = a_kin_close

      close_lead = (d_safe < max(28.0, v_safe * 1.25)) or (phys_margin < 0.0)
      stopping_lead = (v_lead_safe <= 2.5 and d_safe < max(42.0, v_safe * 3.3))
      closing_braking_lead = (
        d_safe < max(45.0, v_safe * 2.8)
        and v_rel_safe < -2.2
        and (a_lead_f < -1.0 or a_lead_safe < -1.8)
      )
      is_urgent = bool(
        fcw or should_stop or danger_ok or stopping_lead or closing_braking_lead
        or ttc < 4.2 or a_kin_eff < -1.25
        or (phys_margin < -1.5 and v_rel_safe < -0.6)
        or (close_lead and (v_rel_safe < -2.4 or a_lead_f < -1.5 or a_lead_safe < -2.5))
        or (a_mpc < -1.50 and (ttc < 6.5 or close_lead or a_kin_eff < -0.85))
      )
      stopping_emergency = bool(
        stopping_lead and (ttc < 3.5 or a_kin_eff < -1.65 or d_safe < max(float(stop_dist) - 0.5, 4.0))
      )
      hard_emergency = bool(
        fcw or should_stop or danger_ok or stopping_emergency
        or ttc < 3.0 or a_kin_eff < -1.85 or a_mpc < -2.20
      )
    else:
      d_safe = 0.0
      v_rel_safe = 0.0
      v_lead_safe = 0.0
      d_headroom = 100.0
      ttc = 99.0
      d_gap = 100.0
      a_kin_eff = 0.0
      danger_ok = False
      self.lead_age = 0.0
      self._prev_d_rel = 1e3
      hard_emergency = bool(fcw or should_stop or a_mpc < -2.4)
      is_urgent = bool(fcw or should_stop or a_mpc < -1.85)

    target = float(a_mpc)
    shaped = False
    stop_flared = False

    # 1) Cruising following-gap buffer on the acceleration side (>25 km/h), fading out on open-gap highway catches
    if lead_status and d_safe > 0.1 and not is_urgent and v_kph > 25.0 and target > 0.0:
      buf_hi = float(np.interp(v_kph, [25.0, 50.0, 80.0], [2.5, 4.5, 6.5]))
      gap_cap = float(np.interp(d_headroom, [-1.5, 0.5, buf_hi, buf_hi + 6.0], [0.05, 0.22, 0.48, 1.05]))
      vrel_cap = float(np.interp(v_rel_safe, [-0.4, 0.0, 0.5, 1.3], [0.02, 0.18, 0.45, 1.05]))
      lead_away = (v_rel_safe > 0.35 or a_lead_f > 0.35)
      lead_match = max(0.0, a_lead_f * (0.90 if lead_away else 0.75) + max(0.0, v_rel_safe) * (0.45 if lead_away else 0.30))
      buf_cap = max(lead_match, max(gap_cap, vrel_cap) if v_rel_safe >= 0.45 else min(gap_cap, vrel_cap))
      cruise_w = float(np.clip(np.interp(v_kph, [25.0, 42.0], [0.0, 1.0]), 0.0, 1.0))
      buf_fade = float(np.clip(np.interp(d_headroom, [buf_hi + 2.0, buf_hi + 10.0], [1.0, 0.0]), 0.0, 1.0))
      eff_w = cruise_w * buf_fade
      eff_cap = target * (1.0 - eff_w) + min(target, buf_cap) * eff_w
      if eff_cap < target:
        target = eff_cap
        shaped = True

    # 2) Kinematic closing pre-decel for confirmed high-speed or rapid urban closing threats
    vision_closing_ok = (
      d_safe < max(72.0, min(115.0, v_safe * 3.6))
      and (v_rel_safe < -3.0 or a_kin_eff < -0.75)
    )
    high_speed_closing = (v_kph > 55.0 and (ttc < 7.5 or d_gap < 8.0) and a_kin_eff < -0.55)
    urban_fast_closing = (32.0 < v_kph <= 55.0 and (ttc < 6.8 or d_gap < 10.0)
                          and (v_rel_safe < -4.5 or a_kin_eff < -0.85) and a_kin_eff < -0.60)
    if (lead_status and d_safe > max(8.0, float(stop_dist) + 2.0)
        and not (blinker_active or lane_change_active)
        and (lead_radar or vision_closing_ok)
        and (high_speed_closing or urban_fast_closing)):
      if v_rel_safe < -1.8 or (v_rel_safe < -1.0 and a_lead_f < -1.0):
        sw = (float(np.clip(np.interp(v_kph, [55.0, 75.0], [0.0, 1.0]), 0.0, 1.0))
              if high_speed_closing
              else float(np.clip(np.interp(v_kph, [28.0, 42.0], [0.60, 1.0]), 0.0, 1.0)))
        early = float(np.clip(a_kin_eff * 1.15 * sw, -1.85, a_coast))
        if target > early:
          target = early
          shaped = True

    # 2b) Sudden lead acquisition and lane-change shock damper (non-urgent / non-stopping leads only)
    if lead_status and not is_urgent and not hard_emergency and target < -0.55 and v_safe > 8.0 and v_lead_safe > 3.0:
      if (blinker_active or lane_change_active) and ttc >= 4.0 and d_gap > 3.0 and a_kin_eff > -0.95:
        lc_floor = max(min(a_kin_eff * 1.20 - 0.15, a_coast), -1.25)
        if target < lc_floor:
          target = lc_floor
          shaped = True
      if self.lead_age < 0.7 and d_gap > 8.0 and ttc >= 5.0 and a_kin_eff > -0.85 and a_lead_f > -1.0 and v_rel_safe > -5.5:
        acq_floor = max(min(a_kin_eff * 1.25 - 0.20, a_coast), -1.25)
        if target < acq_floor:
          target = acq_floor
          shaped = True

    # 3) Human coast-first floor and accel-to-coast dwell (never lift cruise-driven or closing decel)
    if (lead_status and d_safe > 0.1 and not is_urgent and v_safe > 3.5 and v_lead_safe > 2.5
        and target < 0.0 and target >= -0.95 and a_kin_eff >= -0.55 and mpc_source != 'cruise' and curve_decel < 0.05):
      if d_headroom >= -1.5 and v_rel_safe >= -1.6 and a_lead_f >= -1.3:
        u_gap = float(np.interp(d_headroom, [-1.5, 0.5, 4.0], [1.0, 0.25, 0.0]))
        u_vr = float(np.interp(v_rel_safe, [-1.8, -1.1, -0.4], [1.0, 0.25, 0.0]))
        u_al = float(np.interp(a_lead_f, [-1.5, -0.95, -0.4], [1.0, 0.25, 0.0]))
        u = max(u_gap, u_vr, u_al)
        if self.last_pos_age < 1.8 and u < 0.70:
          u *= float(np.interp(self.coast_dwell, [0.0, 1.0], [0.25, 1.0]))
        gentle_coast = max(a_coast, -0.22) if u < 0.2 else a_coast
        coast_floor = float(np.interp(u, [0.0, 1.0], [gentle_coast, min(a_coast - 0.85, target)]))
        if target < coast_floor:
          target = coast_floor
          shaped = True

    # 3b) Curve-entry cornering deceleration cap (coordinates with CurveInnerBias; never stacks past MPC braking)
    if curve_decel > 1e-3 and not is_urgent:
      curve_cap = -curve_decel
      if target > curve_cap:
        target = curve_cap
        shaped = True

    # 3c) Flat/downhill low-speed final-stop comfort flare ("limousine stop")
    if (lead_status and not fcw and not danger_ok and not should_stop
        and 0.15 < v_safe < 3.5 and target < -0.18 and pitch_safe < 0.06):
      grade_w = float(np.interp(pitch_safe, [-0.03, 0.02, 0.06], [1.0, 0.85, 0.0]))
      d_stop_margin = d_safe - max(float(stop_dist) - 0.8, 3.3)
      if grade_w > 0.0 and d_stop_margin > 0.4 and v_rel_safe > -2.5:
        a_req_stop = -(v_safe ** 2) / (2.0 * max(0.4, d_stop_margin))
        if a_req_stop > target + 0.08:
          flare_target = max(target, min(a_req_stop - 0.12, -0.18))
          target = target + grade_w * (flare_target - target)
          shaped = True
          stop_flared = True

    # 4) Continuous asymmetric jerk shaping (safety downward path remains immediate)
    if not is_urgent:
      if target < prev_before:
        max_down_rate = 1.25 if (mpc_source == 'cruise' and target < -0.45) else (0.90 if prev_before > 0.0 else 0.85)
        if lead_status:
          threat_scale = max(
            float(np.interp(-v_rel_safe, [1.2, 2.5], [0.0, 1.0])),
            float(np.interp(-a_kin_eff, [0.40, 0.75], [0.0, 1.0])),
            float(np.interp(-a_lead_f, [0.85, 1.50], [0.0, 1.0])),
          )
          brake_need = max(0.0, -a_kin_eff, -target * threat_scale)
          max_down_rate = max(max_down_rate, float(np.interp(brake_need, [0.45, 0.90, 1.50], [0.85, 1.55, 2.80])))
          if prev_before > 0.05 and v_rel_safe < -1.0:
            lift_rate = float(np.interp(-v_rel_safe, [1.0, 4.0], [1.20, 3.00]))
            max_down_rate = max(max_down_rate, lift_rate)
        if curve_decel > 0.55:
          max_down_rate = max(max_down_rate, float(np.interp(curve_decel, [0.55, 1.50], [0.90, 1.45])))
        if prev_before <= 0.05 and lead_status and self.lead_age < 0.7 and d_gap > 8.0 and ttc >= 5.0 and a_kin_eff > -0.85 and a_lead_f > -1.0:
          max_down_rate = min(max_down_rate, 1.20)
        new_t = max(target, prev_before - max_down_rate * self.dt)
        if new_t != target:
          target = new_t
          shaped = True
      elif target > prev_before:
        free_accel = (not lead_status) or (d_headroom > 12.0) or (v_rel_safe > 0.45) or (a_lead_f > 0.45)
        if free_accel:
          base_up = float(np.interp(v_kph, [10.0, 28.0, 50.0], [2.10, 1.45, 1.20]))
        else:
          base_up = float(np.interp(v_kph, [10.0, 28.0, 50.0], [1.80, 0.95, 0.65]))
        if prev_before < a_coast - 0.10:
          base_up = max(base_up, 2.40 if not lead_status else 1.80)
        elif prev_before <= 0.05 and v_kph > 25.0 and not free_accel and target < 0.50:
          base_up = min(base_up, 0.55)
        if pitch_safe < -0.02 and target < 0.0:
          base_up = min(base_up, 1.20)
        new_t = min(target, prev_before + base_up * self.dt)
        if new_t != target:
          target = new_t
          shaped = True
    else:
      if target < prev_before:
        urg_down = 8.0 if hard_emergency else (1.6 if stop_flared else 4.2)
        new_t = max(target, prev_before - urg_down * self.dt)
        if new_t != target:
          target = new_t
          shaped = True
      elif target > prev_before:
        urg_up = 1.4 if (stop_flared and not hard_emergency) else (2.4 if prev_before < a_coast - 0.10 else 1.5)
        new_t = min(target, prev_before + urg_up * self.dt)
        if new_t != target:
          target = new_t
          shaped = True

    if target > 0.10:
      self.last_pos_age = 0.0
      self.coast_dwell = 0.0
    else:
      self.last_pos_age = min(10.0, self.last_pos_age + self.dt)
      if a_coast - 0.08 <= target <= 0.05:
        self.coast_dwell = min(5.0, self.coast_dwell + self.dt)
      elif target < a_coast - 0.08:
        self.coast_dwell = min(5.0, self.coast_dwell + self.dt * 0.5)

    self.prev_a = target

    # 5) Synchronize vTargetNow & jTargetNow for LongControl and SCC jerk limits
    v_t = float(v_target_mpc)
    j_t = float(j_target_mpc)
    if not is_urgent or stop_flared:
      expected_v_t = v_safe + target * float(action_t)
      if stop_flared:
        v_t = max(v_t, expected_v_t)
        j_t = float(np.clip((target - prev_before) / self.dt, -2.5, 2.5))
      else:
        if target < a_mpc - 0.02:
          v_t = min(v_t, expected_v_t)
        elif target >= a_coast - 0.05:
          v_t = float(np.clip(v_t, expected_v_t - 0.04, expected_v_t + 0.10))
        elif target > a_mpc + 0.02:
          v_t = max(v_t, expected_v_t)
        if shaped:
          j_t = float(np.clip((target - prev_before) / self.dt, -2.5, 2.5))

    return ShaperResult(target, v_t, j_t, shaped, is_urgent, a_coast, curve_decel)

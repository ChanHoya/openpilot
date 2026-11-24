import os
import math
import numpy as np
from openpilot.cereal import log
from opendbc.car.vehicle_model import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from opendbc.car.volkswagen.values import VolkswagenFlags


def is_volkswagen_meb(CP) -> bool:
  # VW MEB(ID.4/ID.5) 판별 단일 소스. 플랫폼 플래그 기반이라 향후 다른 VW angle 제어
  # 차종이 추가되어도 MEB 전용 경로(곡률 폐루프, 리드선택, FCW 게이트)가 오활성되지 않음.
  return CP.brand == "volkswagen" and bool(CP.flags & VolkswagenFlags.MEB)

MIN_SPEED = 1.0
CONTROL_N = 17
CAR_ROTATION_RADIUS = 0.0
# This is a turn radius smaller than most cars can achieve
MAX_CURVATURE = 0.2
MAX_VEL_ERR = 5.0  # m/s

# EU guidelines
MAX_LATERAL_JERK = 5.0  # m/s^3
MAX_LATERAL_ACCEL_NO_ROLL = 3.0  # m/s^2
MAX_LATERAL_ACCEL_NO_ROLL_LOW_SPEED = 4.5  # m/s^2

def apply_deadzone(error, deadzone):
  if error > deadzone:
    error -= deadzone
  elif error < - deadzone:
    error += deadzone
  else:
    error = 0.
  return error

def get_lag_adjusted_curvature(CP, v_ego, psis, curvatures, steer_actuator_delay, distances):
  if len(psis) != CONTROL_N:
    psis = [0.0]*CONTROL_N
    curvatures = [0.0]*CONTROL_N
    distances = [0.0] * CONTROL_N
  v_ego = max(MIN_SPEED, v_ego)

  # TODO this needs more thought, use .2s extra for now to estimate other delays
  delay = max(0.01, steer_actuator_delay)

  # MPC can plan to turn the wheel and turn back before t_delay. This means
  # in high delay cases some corrections never even get commanded. So just use
  # psi to calculate a simple linearization of desired curvature
  current_curvature_desired = curvatures[0]
  delayed_curvature_desired = np.interp(delay, ModelConstants.T_IDXS[:CONTROL_N], curvatures)
  future_curvature_desired = np.interp(1.2, ModelConstants.T_IDXS[:CONTROL_N], curvatures)

  psi = np.interp(delay, ModelConstants.T_IDXS[:CONTROL_N], psis)

  distance = max(np.interp(delay, ModelConstants.T_IDXS[:CONTROL_N], distances), 0.001)
  #average_curvature_desired = psi / (v_ego * delay)

  # curve -> straight or reverse curve
  if v_ego > 5 and abs(current_curvature_desired) > 0.002 and \
     (abs(future_curvature_desired) < 0.001 or np.sign(current_curvature_desired) != np.sign(future_curvature_desired)):
    psis_damping = 0.2
  else:
    psis_damping = 1.0
  #psi *= psis_damping


  average_curvature_desired = psi / distance
  desired_curvature = 2 * average_curvature_desired - current_curvature_desired

  # This is the "desired rate of the setpoint" not an actual desired rate
  max_curvature_rate = MAX_LATERAL_JERK / (v_ego**2) # inexact calculation, check https://github.com/commaai/openpilot/pull/24755
  safe_desired_curvature = np.clip(desired_curvature,
                                current_curvature_desired - max_curvature_rate * DT_MDL,
                                current_curvature_desired + max_curvature_rate * DT_MDL)
  return safe_desired_curvature

def clamp(val, min_val, max_val):
  clamped_val = float(np.clip(val, min_val, max_val))
  return clamped_val, clamped_val != val

def smooth_value(val, prev_val, tau, dt=DT_MDL):
  alpha = 1 - np.exp(-dt/tau) if tau > 0 else 1
  return alpha * val + (1 - alpha) * prev_val

def clip_curvature(v_ego, prev_curvature, new_curvature, roll, max_lateral_jerk: float = MAX_LATERAL_JERK):
  # This function respects ISO lateral jerk and acceleration limits + a max curvature
  v_ego = max(v_ego, MIN_SPEED)
  max_curvature_rate = max_lateral_jerk / (v_ego ** 2)  # inexact calculation, check https://github.com/commaai/openpilot/pull/24755
  new_curvature = np.clip(new_curvature,
                          prev_curvature - max_curvature_rate * DT_CTRL,
                          prev_curvature + max_curvature_rate * DT_CTRL)

  roll_compensation = roll * ACCELERATION_DUE_TO_GRAVITY
  # Avoid an abrupt authority drop at 100 km/h. Gradually transition from
  # the low-speed allowance to the high-speed EU-guideline limit.
  max_lateral_accel_no_roll = float(np.interp(
    v_ego,
    [80 / 3.6, 120 / 3.6],
    [MAX_LATERAL_ACCEL_NO_ROLL_LOW_SPEED, MAX_LATERAL_ACCEL_NO_ROLL],
  ))
  max_lat_accel = max_lateral_accel_no_roll + roll_compensation
  min_lat_accel = -max_lateral_accel_no_roll + roll_compensation
  new_curvature, limited_accel = clamp(new_curvature, min_lat_accel / v_ego ** 2, max_lat_accel / v_ego ** 2)

  new_curvature, limited_max_curv = clamp(new_curvature, -MAX_CURVATURE, MAX_CURVATURE)
  return float(new_curvature), limited_accel or limited_max_curv


def get_accel_from_plan(speeds, accels, t_idxs, action_t=DT_MDL, vEgoStopping=0.05):
  if len(speeds) == len(t_idxs):
    v_now = speeds[0]
    a_now = accels[0]
    v_target = np.interp(action_t, t_idxs, speeds)
    a_target = 2 * (v_target - v_now) / (action_t) - a_now
    v_target_1sec = np.interp(action_t + 1.0, t_idxs, speeds)
    v_max = np.max(speeds)
  else:
    v_now = 0.0
    a_now = 0.0
    v_target = 0.0
    v_target_1sec = 0.0
    a_target = 0.0
    v_max = 0.0
  should_stop = (v_target < vEgoStopping and
                 v_target_1sec < vEgoStopping)
  return a_target, should_stop, v_now, v_max #v_target #v_now

def curv_from_psis(psi_target, psi_rate, vego, action_t):
  vego = np.clip(vego, MIN_SPEED, np.inf)
  curv_from_psi = psi_target / (vego * action_t)
  return 2*curv_from_psi - psi_rate / vego

def get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego, action_t):
  psi_target = np.interp(action_t, t_idxs, yaws)
  psi_rate = yaw_rates[0]
  return curv_from_psis(psi_target, psi_rate, vego, action_t)


CURVE_INNER_BIAS_KEY = "CurveInnerBias"
CURVE_INNER_BIAS_DEFAULT = 10.0


def read_curve_inner_bias(params, default: float = CURVE_INNER_BIAS_DEFAULT) -> float:
  """Read CurveInnerBias (0..100%), with backing-file fallback before params_pyx rebuild."""
  if params is None:
    return float(np.clip(default, 0.0, 100.0))

  raw_value = None
  try:
    if hasattr(params, "get_int"):
      raw_value = params.get_int(CURVE_INNER_BIAS_KEY)
    elif hasattr(params, "get_float"):
      raw_value = params.get_float(CURVE_INNER_BIAS_KEY)
    elif hasattr(params, "get"):
      raw_value = params.get(CURVE_INNER_BIAS_KEY)
  except Exception:
    raw_value = None

  if raw_value is None and hasattr(params, "get_param_path"):
    try:
      param_dir = params.get_param_path()
      param_file = os.path.join(param_dir, CURVE_INNER_BIAS_KEY)
      if os.path.isfile(param_file):
        with open(param_file, "r", encoding="utf-8") as f:
          raw_value = f.read().strip()
    except Exception:
      raw_value = None

  if raw_value is None or raw_value == "":
    return float(np.clip(default, 0.0, 100.0))

  try:
    if isinstance(raw_value, bytes):
      raw_value = raw_value.decode("utf-8", errors="ignore").strip()
    val = float(raw_value)
  except (TypeError, ValueError):
    return float(np.clip(default, 0.0, 100.0))

  if not math.isfinite(val):
    return float(np.clip(default, 0.0, 100.0))
  return float(np.clip(val, 0.0, 100.0))


LANELESS_CENTER_ADJUST_KEY = "LanelessCenterAdjust"
LANELESS_CENTER_ADJUST_DEFAULT = 10.0


def read_laneless_center_adjust(params, default: float = LANELESS_CENTER_ADJUST_DEFAULT) -> float:
  """Read LanelessCenterAdjust (0..30%), with backing-file fallback before params_pyx rebuild."""
  if params is None:
    return float(np.clip(default, 0.0, 100.0))

  raw_value = None
  try:
    if hasattr(params, "get_int"):
      raw_value = params.get_int(LANELESS_CENTER_ADJUST_KEY)
    elif hasattr(params, "get_float"):
      raw_value = params.get_float(LANELESS_CENTER_ADJUST_KEY)
    elif hasattr(params, "get"):
      raw_value = params.get(LANELESS_CENTER_ADJUST_KEY)
  except Exception:
    raw_value = None

  if raw_value is None and hasattr(params, "get_param_path"):
    try:
      param_dir = params.get_param_path()
      param_file = os.path.join(param_dir, LANELESS_CENTER_ADJUST_KEY)
      if os.path.isfile(param_file):
        with open(param_file, "r", encoding="utf-8") as f:
          raw_value = f.read().strip()
    except Exception:
      raw_value = None

  if raw_value is None or raw_value == "":
    return float(np.clip(default, 0.0, 100.0))

  try:
    if isinstance(raw_value, bytes):
      raw_value = raw_value.decode("utf-8", errors="ignore").strip()
    val = float(raw_value)
  except (TypeError, ValueError):
    return float(np.clip(default, 0.0, 100.0))

  if not math.isfinite(val):
    return float(np.clip(default, 0.0, 100.0))
  return float(np.clip(val, 0.0, 100.0))



def _preview_curvature_from_plan(yaws, yaw_rates, t_idxs, vego) -> float:
  """Compute preview road curvature ahead over t in [0.6, 1.2] s."""
  t_arr = np.asarray(t_idxs, dtype=float)
  yaws_arr = np.asarray(yaws, dtype=float)
  rates_arr = np.asarray(yaw_rates, dtype=float)
  if len(t_arr) == 0 or len(yaws_arr) != len(t_arr) or len(rates_arr) != len(t_arr):
    return 0.0

  v_safe = max(float(vego), MIN_SPEED)
  mask = (t_arr >= 0.6) & (t_arr <= 1.2) & np.isfinite(rates_arr) & np.isfinite(yaws_arr)
  if np.any(mask):
    rate_curv = float(np.median(rates_arr[mask])) / v_safe
    t_lo = max(0.6, float(t_arr[0]))
    t_hi = min(1.2, float(t_arr[-1]))
    if t_hi - t_lo >= 0.15:
      chord_curv = float(np.interp(t_hi, t_arr, yaws_arr) - np.interp(t_lo, t_arr, yaws_arr)) / (v_safe * (t_hi - t_lo))
      preview_curv = 0.65 * rate_curv + 0.35 * chord_curv
    else:
      preview_curv = rate_curv
  else:
    preview_curv = float(rates_arr[min(10, len(rates_arr) - 1)]) / v_safe

  return preview_curv if math.isfinite(preview_curv) else 0.0


def _near_lane_y(lane_line) -> float | None:
  """Extract near-field lateral coordinate (x <= 0.75m, +y right) from array or capnp laneLine."""
  pts = getattr(lane_line, "y", lane_line)
  arr = np.asarray(pts, dtype=float)
  if arr.size == 0 or not np.isfinite(arr[0]):
    return None
  if arr.size >= 3 and np.all(np.isfinite(arr[:3])):
    return float(0.5 * arr[0] + 0.3 * arr[1] + 0.2 * arr[2])
  return float(arr[0])


def _lane_center_and_width(lane_lines, lane_line_probs) -> tuple[float, float, float] | None:
  """Estimate current lane center (+y right), lane width, and confidence in [0, 1]."""
  if lane_lines is None or lane_line_probs is None:
    return None
  try:
    if len(lane_lines) < 3 or len(lane_line_probs) < 3:
      return None
    l_prob = float(lane_line_probs[1])
    r_prob = float(lane_line_probs[2])
    y_left = _near_lane_y(lane_lines[1])
    y_right = _near_lane_y(lane_lines[2])
  except (TypeError, IndexError, ValueError):
    return None

  if not (math.isfinite(l_prob) and math.isfinite(r_prob)):
    return None

  lane_conf = float(np.interp(max(l_prob, r_prob), [0.25, 0.60], [0.0, 1.0]))
  if lane_conf <= 0.0:
    return None

  left_valid = l_prob > 0.35 and y_left is not None and -4.0 <= y_left <= 1.5
  right_valid = r_prob > 0.35 and y_right is not None and -1.5 <= y_right <= 4.0

  if left_valid and right_valid:
    raw_width = y_right - y_left
    width_mod = float(np.interp(raw_width, [2.0, 2.5], [0.0, 1.0]) * np.interp(raw_width, [4.5, 5.5], [1.0, 0.0]))
    lane_conf *= width_mod
    if lane_conf <= 0.0:
      return None
    y_center = 0.5 * (y_left + y_right)
    lane_width = float(np.clip(raw_width, 2.5, 4.0))
  elif left_valid and -3.5 <= y_left <= -0.3:
    y_center = y_left + 1.60
    lane_width = 3.2
  elif right_valid and 0.3 <= y_right <= 3.5:
    y_center = y_right - 1.60
    lane_width = 3.2
  else:
    return None

  lane_conf *= float(np.interp(abs(y_center), [1.65, 2.25], [1.0, 0.0]))
  if lane_conf <= 0.0:
    return None
  return y_center, lane_width, lane_conf


def compute_curve_entry_smooth_factor(yaws, yaw_rates, t_idxs, v_ego, action_t, bias_percent: float) -> float:
  """Reduce smoothing lag during curve entry to prevent outer-lane drift while avoiding torque jitter."""
  bias = float(np.clip(bias_percent, 0.0, 100.0))
  if bias <= 0.0 or v_ego < 5.0:
    return 1.0

  t_arr = np.asarray(t_idxs, dtype=float)
  yaws_arr = np.asarray(yaws, dtype=float)
  rates_arr = np.asarray(yaw_rates, dtype=float)
  if len(t_arr) == 0 or len(yaws_arr) != len(t_arr) or len(rates_arr) != len(t_arr):
    return 1.0

  k_now = float(get_curvature_from_plan(yaws_arr, rates_arr, t_arr, v_ego, action_t))
  k_preview = _preview_curvature_from_plan(yaws_arr, rates_arr, t_arr, v_ego)
  lat_accel_preview = abs(k_preview) * (v_ego ** 2)
  if lat_accel_preview <= 0.45 or abs(k_preview) <= abs(k_now):
    return 1.0
  if abs(k_now) > 0.001 and (k_preview * k_now) < 0.0:
    return 1.0

  entry_delta = abs(k_preview) - abs(k_now)
  entry_ratio = entry_delta / max(abs(k_preview), 0.002)
  accel_ramp = float(np.interp(lat_accel_preview, [0.45, 1.35], [0.0, 1.0]))
  entry_gain = float(np.interp(entry_ratio, [0.05, 0.45], [0.0, 1.0])) * accel_ramp
  bias_strength = bias * 0.01
  factor = 1.0 - (0.08 * min(1.0, bias / 10.0) + 0.0875 * bias_strength) * entry_gain
  return float(np.clip(factor, 0.85, 1.0))


def compute_curve_inner_curvature_assist(base_curvature, yaws, yaw_rates, t_idxs, v_ego, action_t,
                                         bias_percent: float, lane_lines=None, lane_line_probs=None,
                                         desire=0, lane_change_prob: float = 0.0) -> float:
  """Guide cornering toward the inner lane at entry while suppressing outer drift and inner overshoot."""
  base = float(base_curvature)
  if not math.isfinite(base):
    return 0.0
  bias = float(np.clip(bias_percent, 0.0, 100.0))
  if bias <= 0.0 or v_ego < 5.0 or int(desire) != 0 or float(lane_change_prob) > 0.15:
    return base

  speed_scale = float(np.interp(v_ego, [5.0, 8.5], [0.0, 1.0]))
  preview_curvature = _preview_curvature_from_plan(yaws, yaw_rates, t_idxs, v_ego)
  curv_mag = max(abs(base), abs(preview_curvature))
  lat_accel_mag = curv_mag * (v_ego ** 2)
  curv_gate = float(np.interp(curv_mag, [0.0020, 0.0075], [0.0, 1.0]))
  accel_gate = float(np.interp(lat_accel_mag, [0.45, 1.35], [0.0, 1.0]))
  curve_activity = curv_gate * accel_gate * speed_scale
  if curve_activity <= 1e-4:
    return base

  bias_strength = bias * 0.01
  ramp = min(1.0, bias_strength / 0.10)

  # S-curve inflection detection: if current and preview curvature have opposite signs,
  # we are at an S-curve inflection point. Suppress assist to avoid cutting inside or oscillating.
  s_curve_factor = 1.0
  if abs(base) > 0.001 and abs(preview_curvature) > 0.001 and (base * preview_curvature) < 0.0:
    s_curve_factor = 0.0

  # Apex exit damping: as curvature drops ahead, taper off inner guidance so the car doesn't hug inside on exit.
  apex_factor = 1.0
  if abs(preview_curvature) < abs(base):
    apex_factor = float(np.interp(abs(preview_curvature) / max(abs(base), 1e-4), [0.35, 0.85], [0.25, 0.85]))

  # High-speed damping: reduce inner target shift at high speeds (v > 50 km/h) to prevent hugging too tight
  speed_damping = float(np.interp(v_ego * 3.6, [45.0, 75.0, 100.0], [1.0, 0.65, 0.40]))
  # High-curvature tapering: avoid stacking extra inward bias on already tight curves (R < 125m)
  high_curv_taper = float(np.interp(abs(base), [0.008, 0.018], [1.0, 0.20]))

  # Dominant curve sign: prioritize current established curvature over noisy preview in S-curves
  if abs(base) >= 0.001:
    curve_sign = 1.0 if base >= 0.0 else -1.0
  else:
    curve_sign = 1.0 if preview_curvature >= 0.0 else -1.0

  lane_est = _lane_center_and_width(lane_lines, lane_line_probs)
  already_inside = False
  if lane_est is not None:
    y_center, lane_width, lane_conf = lane_est
    drift_metric = curve_sign * y_center
    max_inner_shift = min(0.24 * bias_strength * curve_activity * speed_damping * high_curv_taper, max(0.0, 0.5 * lane_width - 1.25))
    inside_clearance = 0.5 * lane_width + drift_metric
    try:
      if lane_lines is not None and lane_line_probs is not None and len(lane_lines) >= 3 and len(lane_line_probs) >= 3:
        if curve_sign > 0.0 and float(lane_line_probs[2]) > 0.40:
          y_in = _near_lane_y(lane_lines[2])
          if y_in is not None and -1.5 <= y_in <= 4.0:
            inside_clearance = y_in
        elif curve_sign < 0.0 and float(lane_line_probs[1]) > 0.40:
          y_in = _near_lane_y(lane_lines[1])
          if y_in is not None and -4.0 <= y_in <= 1.5:
            inside_clearance = -y_in
    except (TypeError, IndexError, ValueError):
      pass
    already_inside = (drift_metric < -max_inner_shift) or (inside_clearance < 1.25)

  # 1. Curve-entry preview lead: advance steering as curvature builds ahead in same direction
  delta_kappa_lead = 0.0
  if not already_inside and (preview_curvature * curve_sign) > 0.0 and abs(preview_curvature) > abs(base):
    delta_kappa_lead = 0.35 * bias_strength * curve_activity * s_curve_factor * high_curv_taper * (preview_curvature - base)

  # 2. Lane-bounded inner-curve bias & asymmetric outer-drift guard
  delta_kappa_lane = 0.0
  is_outer_drift = False
  intrusion_blend = 0.0
  if lane_est is not None:
    target_y = y_center + curve_sign * max_inner_shift
    lookahead_dist = float(np.clip(v_ego * 0.95, 9.0, 24.0))
    pure_pursuit_curv = (2.0 * target_y) / (lookahead_dist ** 2)

    inner_overshoot = max(0.0, -drift_metric - max_inner_shift)
    line_intrusion = max(0.0, 1.20 - inside_clearance)
    if drift_metric > 0.02 and line_intrusion <= 0.0:
      # Drifting toward outside of curve: corrective recovery (tapered on tight high-curvature turns)
      is_outer_drift = True
      outer_boost = float(np.interp(abs(base), [0.008, 0.018], [1.45, 1.05]))
      delta_kappa_lane = lane_conf * ramp * s_curve_factor * apex_factor * (0.60 + 0.40 * bias_strength) * outer_boost * pure_pursuit_curv
    elif drift_metric < -max_inner_shift or line_intrusion > 0.0:
      # Already hugging or crossing inside lane line: smoothly escalate outward centering to stay inside lane
      intrusion_blend = max(
        float(np.interp(line_intrusion, [0.0, 0.45], [0.0, 1.0])),
        float(np.interp(inner_overshoot, [0.25, 0.75], [0.0, 1.0])),
      )
      inner_gain = float(np.interp(intrusion_blend, [0.0, 1.0], [0.25, 1.15]))
      delta_kappa_lane = lane_conf * max(ramp, intrusion_blend) * inner_gain * pure_pursuit_curv
    else:
      # Centered or mild inside: gentle guidance only at entry
      delta_kappa_lane = lane_conf * ramp * s_curve_factor * apex_factor * high_curv_taper * 0.45 * pure_pursuit_curv

  outer_headroom = 1.30 if is_outer_drift else float(np.interp(intrusion_blend, [0.0, 1.0], [1.0, 2.60]))
  max_delta = (0.28 + 0.42 * bias_strength) * max(ramp, intrusion_blend) * outer_headroom / max(v_ego ** 2, 25.0)
  raw_delta = delta_kappa_lead + delta_kappa_lane
  # Never let outward inside-line correction reverse the curve direction
  if intrusion_blend > 0.0 and (raw_delta * curve_sign) < 0.0 and abs(base) > 0.004:
    max_delta = min(max_delta, 0.45 * abs(base))
  delta_kappa = float(np.clip(raw_delta, -max_delta, max_delta))
  return float(base + delta_kappa)


def compute_laneless_center_assist(base_curvature: float, v_ego: float, adjust_percent: float,
                                   lane_lines=None, lane_line_probs=None,
                                   desire: int = 0, lane_change_prob: float = 0.0) -> float:
  """Gently nudge vehicle back to lane center on straight/mild roads in Laneless mode when drifting past deadzone."""
  base = float(base_curvature)
  if not math.isfinite(base):
    return 0.0
  adjust = float(np.clip(adjust_percent, 0.0, 100.0))
  if adjust <= 0.0 or v_ego < 5.0 or int(desire) != 0 or float(lane_change_prob) > 0.15:
    return base

  curv_mag = abs(base)
  # Only apply on straight or mild curves (radius >= 400m); yield to curve control on sharp curves
  curve_gate = float(np.interp(curv_mag, [0.0010, 0.0025], [1.0, 0.0]))
  if curve_gate <= 1e-4:
    return base

  lane_est = _lane_center_and_width(lane_lines, lane_line_probs)
  if lane_est is None:
    return base

  y_center, _lane_width, lane_conf = lane_est
  # Deadzone: 12cm (0.12m). If drift is within 12cm of lane center, do not intervene to prevent hunting.
  deadzone = 0.12
  if abs(y_center) <= deadzone:
    return base

  excess_drift = math.copysign(min(abs(y_center) - deadzone, 0.35), y_center)
  adjust_strength = adjust * 0.01

  # Pure pursuit lookahead distance (12m to 28m)
  lookahead_dist = float(np.clip(v_ego * 1.1, 12.0, 28.0))
  pure_pursuit_curv = (2.0 * excess_drift) / (lookahead_dist ** 2)

  speed_damping = float(np.interp(v_ego * 3.6, [40.0, 80.0, 120.0], [1.0, 0.85, 0.65]))
  delta_kappa_target = pure_pursuit_curv * adjust_strength * lane_conf * curve_gate * speed_damping

  # Bound maximum restorative curvature rate to maintain gentle feel (<= ~0.08 m/s^2 lat accel)
  max_delta = 0.08 * (adjust_strength / 0.10) / max(v_ego ** 2, 25.0)
  max_delta = min(max_delta, 0.0012)
  delta_kappa = float(np.clip(delta_kappa_target, -max_delta, max_delta))

  return float(base + delta_kappa)


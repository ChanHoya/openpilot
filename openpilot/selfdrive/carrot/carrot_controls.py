from openpilot.common.realtime import DT_CTRL
from openpilot.common.params import Params

def _safe_enum_int(val, default: int = 0) -> int:
  if val is None:
    return default
  raw = getattr(val, "raw", val)
  try:
    return int(raw)
  except (TypeError, ValueError):
    pass
  if isinstance(raw, str):
    mapping = {
      "off": 0,
      "preLaneChange": 1,
      "laneChangeStarting": 2,
      "laneChangeFinishing": 3,
      "none": 0,
      "left": 1,
      "right": 2,
    }
    return mapping.get(raw, default)
  return default


class CarrotControls:
  def __init__(self, CP):
    self.CP = CP
    self.params = Params()
    self.lat_suspend_active = False
    self.lat_suspend_enter_t = 0.0
    self.lat_suspend_hold_t = 0.0

    # Lane change smoothing state
    self.prev_lane_change_state = 0  # 0: off, 1: preLaneChange, 2: laneChangeStarting, 3: laneChangeFinishing
    self.lane_change_timer = 0.0
    self.lane_change_finish_timer = 0.0
    self.lane_change_base_curvature = 0.0
    self.lane_change_finish_base_curvature = 0.0
    self.is_auto_lane_change = False
    self.finish_smoothing_active = False
    self.lane_change_ramp = 1.0
    self.lane_change_smooth_sec = 1.0
    self._params_read_frame = 0
    self.update_params()

  def update_params(self):
    try:
      smooth_param = self.params.get_int("AutoLaneChangeSmooth")
    except Exception:
      smooth_param = 10

    # default in params_keys is "10" (1.0s). 0 disables it.
    if smooth_param <= 0:
      self.lane_change_smooth_sec = 0.0
    else:
      self.lane_change_smooth_sec = float(smooth_param) * 0.1

  def reset_lane_change(self):
    self.prev_lane_change_state = 0
    self.lane_change_timer = 0.0
    self.lane_change_finish_timer = 0.0
    self.lane_change_base_curvature = 0.0
    self.lane_change_finish_base_curvature = 0.0
    self.is_auto_lane_change = False
    self.finish_smoothing_active = False
    self.lane_change_ramp = 1.0

  def smooth_lane_change(self, CS, model_v2, raw_desired_curvature, current_desired_curvature):
    self._params_read_frame += 1
    if self._params_read_frame % 50 == 0:
      self.update_params()

    if self.lane_change_smooth_sec <= 0.0 or model_v2 is None:
      self.lane_change_ramp = 1.0
      return raw_desired_curvature, 1.0, 5.0

    meta = getattr(model_v2, "meta", None)
    lc_state = _safe_enum_int(getattr(meta, "laneChangeState", 0)) if meta is not None else 0
    lc_dir = _safe_enum_int(getattr(meta, "laneChangeDirection", 0)) if meta is not None else 0

    blinker_left = (lc_dir == 1)
    # Check if driver is currently applying nudge steering torque in the direction of the lane change
    is_nudge = CS.steeringPressed and (
      (CS.steeringTorque > 0 if blinker_left else CS.steeringTorque < 0)
    )

    # 1) State transitions:
    if lc_state == 2 and self.prev_lane_change_state != 2:
      # Just entered laneChangeStarting
      if is_nudge:
        self.is_auto_lane_change = False
        self.lane_change_ramp = 1.0
      else:
        self.is_auto_lane_change = True
        self.lane_change_timer = 0.0
        self.lane_change_base_curvature = current_desired_curvature
        self.finish_smoothing_active = False
        self.lane_change_ramp = 0.0

    elif lc_state == 3 and self.prev_lane_change_state != 3:
      # Just entered laneChangeFinishing
      self.lane_change_finish_base_curvature = current_desired_curvature
      self.lane_change_finish_timer = 0.0
      self.finish_smoothing_active = True
      self.is_auto_lane_change = False

    elif lc_state not in (2, 3) and self.prev_lane_change_state == 3:
      # Exiting laneChangeFinishing into off/preLaneChange
      if not is_nudge and not CS.steeringPressed:
        self.lane_change_finish_base_curvature = current_desired_curvature
        self.lane_change_finish_timer = 0.0
        self.finish_smoothing_active = True
      else:
        self.finish_smoothing_active = False
      self.is_auto_lane_change = False

    self.prev_lane_change_state = lc_state

    # 2) Calculate smoothed target curvature and lateral jerk limit
    target_curvature = raw_desired_curvature
    max_lateral_jerk = 5.0
    ramp_duration = max(0.1, self.lane_change_smooth_sec)

    if lc_state == 2 and self.is_auto_lane_change:
      # Driver manual steering override check: if driver presses the wheel, immediately grant 100% authority
      if CS.steeringPressed:
        self.is_auto_lane_change = False
        self.lane_change_ramp = 1.0
      else:
        self.lane_change_timer += DT_CTRL
        p = min(1.0, max(0.0, self.lane_change_timer / ramp_duration))
        # Smoothstep (cubic Hermite) S-curve: 3p^2 - 2p^3
        self.lane_change_ramp = 3.0 * (p ** 2) - 2.0 * (p ** 3)
        target_curvature = self.lane_change_base_curvature + self.lane_change_ramp * (
          raw_desired_curvature - self.lane_change_base_curvature
        )
        # Bounded lateral jerk during ease-in phase
        max_lateral_jerk = 1.5 if self.lane_change_ramp < 0.95 else 5.0

    elif (lc_state == 3 or self.finish_smoothing_active) and not CS.steeringPressed:
      v_ego = float(getattr(CS, "vEgo", 0.0))
      base_mag = abs(self.lane_change_finish_base_curvature)
      raw_mag = abs(raw_desired_curvature)
      entering_curve = (
        (raw_mag > max(base_mag + 0.0025, 0.0055))
        or (raw_mag * (v_ego ** 2) > 0.65 and raw_mag > base_mag + 0.0015)
        or ((raw_desired_curvature * self.lane_change_finish_base_curvature) < 0.0 and raw_mag > 0.0030)
      )
      if entering_curve:
        self.finish_smoothing_active = False
        self.lane_change_ramp = 1.0
      else:
        self.lane_change_finish_timer += DT_CTRL
        finish_duration = min(ramp_duration, 1.0 if lc_state == 3 else 0.5)
        if self.lane_change_finish_timer < finish_duration:
          p = min(1.0, max(0.0, self.lane_change_finish_timer / finish_duration))
          ramp = 3.0 * (p ** 2) - 2.0 * (p ** 3)
          target_curvature = (1.0 - ramp) * self.lane_change_finish_base_curvature + ramp * raw_desired_curvature
          max_lateral_jerk = 1.8 if ramp < 0.95 else 5.0
          self.lane_change_ramp = 1.0
        else:
          self.finish_smoothing_active = False
          self.lane_change_ramp = 1.0
    else:
      self.finish_smoothing_active = False
      self.lane_change_ramp = 1.0

    return target_curvature, self.lane_change_ramp, max_lateral_jerk

  def lat_suspend_control(self, CS, latActive, model_v2=None):
    resume_angle  = 15
    delay_sec     = 1.0
    hold_sec      = 0.5

    blinker_on = bool(getattr(CS, "leftBlinker", False) or getattr(CS, "rightBlinker", False))
    lane_probs = getattr(model_v2, "laneLineProbs", None) or []
    # laneLineProbs: [0: outer left, 1: inner left, 2: inner right, 3: outer right]
    lane_visible = False
    if len(lane_probs) >= 3:
      lane_visible = bool(lane_probs[1] >= 0.3 or lane_probs[2] >= 0.3)

    # 1) enter condition timer
    if not self.lat_suspend_active:
      enter_cond = CS.steeringPressed and abs(CS.steeringAngleDeg) > float(self.params.get_int("LatSuspendAngleDeg"))
      # 커브길에서 차선이 30% 이상 인식되고 깜빡이를 켜지 않은 경우 수동 조향 중에도 자동조향 유지
      if lane_visible and not blinker_on:
        enter_cond = False

      if enter_cond:
        self.lat_suspend_enter_t += DT_CTRL
        if self.lat_suspend_enter_t >= delay_sec:
          self.lat_suspend_active = True
          self.lat_suspend_hold_t = 0.0
      else:
        self.lat_suspend_enter_t = 0.0

    # 2) while suspended: enforce minimum hold time + hysteresis exit
    if self.lat_suspend_active:
      self.lat_suspend_hold_t += DT_CTRL

      # 수동 조향을 멈췄을 때:
      # - 스티어링 각도가 15도 미만으로 돌아왔거나
      # - 또는 차선이 인식되고 깜빡이가 꺼져 있다면 커브 중간이어도 즉시 복귀
      angle_cleared = abs(CS.steeringAngleDeg) < resume_angle
      lane_ready = lane_visible and not blinker_on
      exit_cond = (not CS.steeringPressed) and (angle_cleared or lane_ready)

      if (self.lat_suspend_hold_t >= hold_sec) and exit_cond:
        self.lat_suspend_active = False
        self.lat_suspend_enter_t = 0.0

    if self.lat_suspend_active:
      latActive = False
    return latActive

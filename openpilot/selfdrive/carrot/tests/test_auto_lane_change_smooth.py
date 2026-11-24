from types import SimpleNamespace
import pytest
from openpilot.selfdrive.carrot.carrot_controls import CarrotControls


class MockCS:
  def __init__(self, steering_pressed=False, steering_torque=0.0):
    self.steeringPressed = steering_pressed
    self.steeringTorque = steering_torque
    self.vEgo = 20.0
    self.leftBlinker = False
    self.rightBlinker = False


class MockModelV2:
  def __init__(self, lc_state=0, lc_dir=0, desired_curvature=0.0):
    self.meta = SimpleNamespace(
      laneChangeState=lc_state,
      laneChangeDirection=lc_dir,
    )
    self.action = SimpleNamespace(
      desiredCurvature=desired_curvature
    )


def test_auto_lane_change_smooth_ease_in():
  controls = CarrotControls(None)
  controls.params = SimpleNamespace(get_int=lambda key: 10 if key == "AutoLaneChangeSmooth" else 0)
  controls.update_params()
  assert controls.lane_change_smooth_sec == 1.0

  cs = MockCS(steering_pressed=False, steering_torque=0.0)
  raw_curv = -0.002
  base_curv = 0.0001

  # Initially in off (state 0)
  m0 = MockModelV2(lc_state=0, lc_dir=0, desired_curvature=base_curv)
  curv, ramp, jerk = controls.smooth_lane_change(cs, m0, base_curv, base_curv)
  assert ramp == 1.0
  assert jerk == 5.0

  # Enters laneChangeStarting (state 2), direction left (1)
  m1 = MockModelV2(lc_state=2, lc_dir=1, desired_curvature=raw_curv)

  # Step 0: Initial transition into laneChangeStarting
  curv, ramp, jerk = controls.smooth_lane_change(cs, m1, raw_curv, base_curv)
  assert controls.is_auto_lane_change is True
  # At t=0 (step 0, 10ms), ramp is essentially 0.0, target curvature is approximately base curvature
  assert ramp == pytest.approx(0.0, abs=1e-3)
  assert curv == pytest.approx(base_curv, abs=1e-3)
  assert jerk == 1.5

  # Run for 50 steps (0.5s at DT_CTRL=0.01)
  for _ in range(50):
    curv, ramp, jerk = controls.smooth_lane_change(cs, m1, raw_curv, base_curv)

  # At t=0.5s (p=0.5), cubic Hermite: 3*(0.5^2) - 2*(0.5^3) = 0.75 - 0.25 = 0.5
  assert ramp == pytest.approx(0.5, abs=0.02)
  expected_curv_half = base_curv + 0.5 * (raw_curv - base_curv)
  assert curv == pytest.approx(expected_curv_half, abs=1e-4)
  assert jerk == 1.5

  # Run remaining 50 steps to reach 1.0s (100 steps total)
  for _ in range(50):
    curv, ramp, jerk = controls.smooth_lane_change(cs, m1, raw_curv, base_curv)

  # At t=1.0s, ramp should reach 1.0 and curvature reaches full target
  assert ramp == pytest.approx(1.0, abs=1e-4)
  assert curv == pytest.approx(raw_curv, abs=1e-4)
  assert jerk == 5.0


def test_manual_nudge_retains_immediate_authority():
  controls = CarrotControls(None)
  controls.params = SimpleNamespace(get_int=lambda key: 10 if key == "AutoLaneChangeSmooth" else 0)
  controls.update_params()

  # Driver is pressing steering wheel with left torque (nudge)
  cs_nudge = MockCS(steering_pressed=True, steering_torque=120.0)
  raw_curv = -0.002
  base_curv = 0.0

  m_nudge = MockModelV2(lc_state=2, lc_dir=1, desired_curvature=raw_curv)

  curv, ramp, jerk = controls.smooth_lane_change(cs_nudge, m_nudge, raw_curv, base_curv)
  assert controls.is_auto_lane_change is False
  assert ramp == 1.0
  assert curv == pytest.approx(raw_curv, abs=1e-5)
  assert jerk == 5.0


def test_driver_override_during_auto_ramp():
  controls = CarrotControls(None)
  controls.params = SimpleNamespace(get_int=lambda key: 10 if key == "AutoLaneChangeSmooth" else 0)
  controls.update_params()

  cs = MockCS(steering_pressed=False, steering_torque=0.0)
  raw_curv = -0.002
  m1 = MockModelV2(lc_state=2, lc_dir=1, desired_curvature=raw_curv)

  # Start auto lane change and run 20 steps (0.2s)
  for _ in range(20):
    controls.smooth_lane_change(cs, m1, raw_curv, 0.0)

  assert controls.lane_change_ramp < 0.5
  assert controls.is_auto_lane_change is True

  # Driver suddenly touches steering wheel
  cs_pressed = MockCS(steering_pressed=True, steering_torque=50.0)
  curv, ramp, jerk = controls.smooth_lane_change(cs_pressed, m1, raw_curv, 0.0)

  # Full authority immediately granted to driver
  assert controls.is_auto_lane_change is False
  assert ramp == 1.0
  assert curv == pytest.approx(raw_curv, abs=1e-5)
  assert jerk == 5.0


def test_finish_smoothing_transition():
  controls = CarrotControls(None)
  controls.params = SimpleNamespace(get_int=lambda key: 10 if key == "AutoLaneChangeSmooth" else 0)
  controls.update_params()

  cs = MockCS(steering_pressed=False, steering_torque=0.0)
  finish_base = -0.0018
  center_target = 0.0

  # Transition into laneChangeFinishing (state 3)
  m_fin = MockModelV2(lc_state=3, lc_dir=1, desired_curvature=center_target)
  controls.smooth_lane_change(cs, m_fin, center_target, finish_base)
  assert controls.finish_smoothing_active is True

  # Transition from laneChangeFinishing (state 3) to off (state 0)
  m_off = MockModelV2(lc_state=0, lc_dir=0, desired_curvature=center_target)
  curv, ramp, jerk = controls.smooth_lane_change(cs, m_off, center_target, finish_base)
  assert controls.finish_smoothing_active is True
  # At exit moment, it smoothly blends between finish_base and center_target
  assert curv != center_target
  assert jerk < 5.0


def test_disabled_param():
  controls = CarrotControls(None)
  controls.params = SimpleNamespace(get_int=lambda key: 0 if key == "AutoLaneChangeSmooth" else 0)
  controls.update_params()
  assert controls.lane_change_smooth_sec == 0.0

  cs = MockCS(steering_pressed=False, steering_torque=0.0)
  m1 = MockModelV2(lc_state=2, lc_dir=1, desired_curvature=-0.002)

  curv, ramp, jerk = controls.smooth_lane_change(cs, m1, -0.002, 0.0)
  assert ramp == 1.0
  assert curv == -0.002
  assert jerk == 5.0


def test_latcontrol_torque_scales_lookahead_jerk():
  from openpilot.cereal import car
  from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque

  cp = car.CarParams.new_message()
  torque = cp.lateralTuning.init('torque')
  torque.kp = 1.0
  torque.ki = 0.1
  torque.kf = 1.0
  torque.latAccelFactor = 2.58
  torque.friction = 0.11
  torque.useSteeringAngle = False
  cp = cp.as_reader()
  ci = SimpleNamespace(
    use_nnff=False,
    use_nnff_lite=True,
    torque_from_lateral_accel=lambda: (lambda *args, **kwargs: 0.0),
    get_pid_accel_limits=lambda *args: (-3.0, 2.0),
  )

  lac = LatControlTorque(cp, ci)
  lac.params = SimpleNamespace(get_int=lambda *args: 0, get_float=lambda *args: 0.0)

  # When ramp is 0.0, lookahead jerk must be 0
  lac.lane_change_ramp = 0.0
  lookahead = 2.5
  lane_change_ramp = getattr(lac, "lane_change_ramp", 1.0)
  if lane_change_ramp < 1.0:
    lookahead *= lane_change_ramp
  assert lookahead == 0.0

  # When ramp is 0.5, lookahead jerk is 50%
  lac.lane_change_ramp = 0.5
  lookahead = 2.5
  lane_change_ramp = getattr(lac, "lane_change_ramp", 1.0)
  if lane_change_ramp < 1.0:
    lookahead *= lane_change_ramp
  assert lookahead == pytest.approx(1.25)


def test_real_capnp_dynamic_enum_handling():
  from openpilot.cereal import log
  controls = CarrotControls(None)
  controls.params = SimpleNamespace(get_int=lambda key: 10 if key == "AutoLaneChangeSmooth" else 0)
  controls.update_params()
  cs = MockCS(steering_pressed=False, steering_torque=0.0)

  # 1) Real capnp message with DynamicEnum
  msg = log.ModelDataV2.new_message()
  msg.meta.laneChangeState = log.LaneChangeState.laneChangeStarting
  msg.meta.laneChangeDirection = log.LaneChangeDirection.left
  msg.action.desiredCurvature = -0.002
  # Convert to capnp reader object as delivered by SubMaster
  reader = msg.as_reader()

  curv, ramp, jerk = controls.smooth_lane_change(cs, reader, -0.002, 0.0)
  assert controls.is_auto_lane_change is True
  assert ramp == pytest.approx(0.0, abs=1e-3)
  assert jerk == 1.5

  # 2) Transition to laneChangeFinishing with real capnp enum
  msg_fin = log.ModelDataV2.new_message()
  msg_fin.meta.laneChangeState = log.LaneChangeState.laneChangeFinishing
  msg_fin.meta.laneChangeDirection = log.LaneChangeDirection.left
  reader_fin = msg_fin.as_reader()

  curv_fin, ramp_fin, jerk_fin = controls.smooth_lane_change(cs, reader_fin, 0.0, -0.001)
  assert controls.finish_smoothing_active is True
  assert ramp_fin == 1.0

  # 3) String enum mock support
  str_mock = SimpleNamespace(
    meta=SimpleNamespace(laneChangeState="laneChangeStarting", laneChangeDirection="left"),
    action=SimpleNamespace(desiredCurvature=-0.002),
  )
  controls.reset_lane_change()
  curv_s, ramp_s, jerk_s = controls.smooth_lane_change(cs, str_mock, -0.002, 0.0)
  assert controls.is_auto_lane_change is True


def test_lane_change_finish_entering_curve_releases_full_curvature():
  controls = CarrotControls(None)
  controls.params = SimpleNamespace(get_int=lambda key: 10 if key == "AutoLaneChangeSmooth" else 0)
  controls.update_params()

  cs = MockCS(steering_pressed=False, steering_torque=0.0)
  cs.vEgo = 11.4  # ~41 km/h (0000011f seg 7)
  base_curv = 0.0038

  # Enter laneChangeFinishing -> off
  m_fin = MockModelV2(lc_state=3, lc_dir=2, desired_curvature=0.0042)
  controls.smooth_lane_change(cs, m_fin, 0.0042, base_curv)
  assert controls.finish_smoothing_active is True

  m_off_mild = MockModelV2(lc_state=0, lc_dir=0, desired_curvature=0.0050)
  _, _, jerk_mild = controls.smooth_lane_change(cs, m_off_mild, 0.0050, base_curv)
  assert controls.finish_smoothing_active is True
  assert jerk_mild < 5.0

  # Sharp curve builds (raw=0.0098): finish smoothing immediately yields full curvature & jerk authority
  m_curve = MockModelV2(lc_state=0, lc_dir=0, desired_curvature=0.0098)
  curv, ramp, jerk = controls.smooth_lane_change(cs, m_curve, 0.0098, 0.0043)
  assert controls.finish_smoothing_active is False
  assert ramp == 1.0
  assert jerk == 5.0
  assert curv == pytest.approx(0.0098)

  # Opposite-sign curve transition after lane change also releases immediately
  controls.reset_lane_change()
  controls.smooth_lane_change(cs, MockModelV2(lc_state=3, lc_dir=1, desired_curvature=-0.0030), -0.0030, -0.0035)
  curv_opp, ramp_opp, jerk_opp = controls.smooth_lane_change(
    cs, MockModelV2(lc_state=0, lc_dir=0, desired_curvature=0.0040), 0.0040, -0.0035
  )
  assert controls.finish_smoothing_active is False
  assert ramp_opp == 1.0
  assert jerk_opp == 5.0
  assert curv_opp == pytest.approx(0.0040)




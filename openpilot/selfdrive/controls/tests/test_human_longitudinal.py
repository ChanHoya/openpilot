import math
from types import SimpleNamespace

import numpy as np
import pytest
from openpilot.cereal import car
from openpilot.selfdrive.controls.lib.human_longitudinal import (
  CURVE_ENTRY_ONSET_JERK,
  CURVE_ENTRY_SEVERE_ONSET_JERK,
  CurveEntryDecel,
  HumanLongitudinalShaper,
  compute_accel_ramp_targets,
  compute_curve_entry_decel_target,
  natural_coast_accel,
)
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.modeld.constants import ModelConstants


def _step_shaper(
  shaper: HumanLongitudinalShaper,
  *,
  a_mpc: float,
  v_ego: float,
  a_ego: float = 0.0,
  pitch: float = 0.0,
  action_t: float = 0.25,
  lead_status: bool = True,
  d_rel: float = 32.0,
  v_rel: float = 0.4,
  v_lead: float | None = None,
  a_lead: float = 0.0,
  track_id: int = 7,
  t_follow: float = 0.90,
  comfort_brake: float = 2.5,
  stop_dist: float = 5.0,
  should_stop: bool = False,
  fcw: bool = False,
  reset_state: bool = False,
  mpc_mode: str = "acc",
  predicted_danger_margin: float = 100.0,
  v_target_mpc: float | None = None,
):
  if v_lead is None:
    v_lead = max(0.0, v_ego + v_rel)
  if v_target_mpc is None:
    v_target_mpc = v_ego + a_mpc * action_t
  shaper.update_lead_filter(lead_status, a_lead, track_id)
  return shaper.shape(
    a_mpc=a_mpc,
    v_target_mpc=v_target_mpc,
    j_target_mpc=0.0,
    v_ego=v_ego,
    a_ego=a_ego,
    pitch=pitch,
    action_t=action_t,
    lead_status=lead_status,
    d_rel=d_rel,
    v_rel=v_rel,
    v_lead=v_lead,
    a_lead=a_lead,
    t_follow=t_follow,
    comfort_brake=comfort_brake,
    stop_dist=stop_dist,
    should_stop=should_stop,
    fcw=fcw,
    reset_state=reset_state,
    mpc_mode=mpc_mode,
    predicted_danger_margin=predicted_danger_margin,
  )


def test_receding_lead_radar_jitter_never_triggers_false_brake_spikes():
  """Replays seg 25 t=29.97s and seg 26 t=3.66s / 54.70s receding-lead radar aLeadK dips."""
  shaper = HumanLongitudinalShaper(dt=0.05)

  # Scenario A (seg 25 t=29.97s): 64.6 km/h, lead pulling away (+0.4 m/s) at d=32.1m, single radar blip aLeadK=-0.46
  for _ in range(5):
    _step_shaper(shaper, a_mpc=0.90, v_ego=17.9, a_ego=0.90, d_rel=32.1, v_rel=0.4, a_lead=0.50)
  res = _step_shaper(shaper, a_mpc=0.85, v_ego=17.95, a_ego=0.95, d_rel=32.1, v_rel=0.4, a_lead=-0.46)
  assert res.a_target > 0.0
  assert not res.is_urgent

  # Scenario B (seg 26 t=54.70s): 68.2 km/h, d=19.5m, lead pulling away fast (+3.4 m/s), brief aLeadK=-0.37
  shaper.reset(0.55)
  for _ in range(4):
    _step_shaper(shaper, a_mpc=0.58, v_ego=18.9, a_ego=0.30, d_rel=19.3, v_rel=3.4, a_lead=0.13)
  res_b = _step_shaper(shaper, a_mpc=0.58, v_ego=18.95, a_ego=0.30, d_rel=19.5, v_rel=3.4, a_lead=-0.37)
  assert res_b.a_target >= 0.35
  assert abs(res_b.a_target - 0.58) < 0.25


def test_accel_to_coast_dwell_uses_natural_coast_and_syncs_hyundai_longcontrol():
  """Transitioning from acceleration to mild closing dwells in natural coasting without hydraulic brake PID windup."""
  CP = car.CarParams.new_message()
  CP.brand = "hyundai"
  CP.openpilotLongitudinalControl = True
  CP.vEgoStarting = 0.1
  CP.vEgoStopping = 0.5
  CP.longitudinalTuning.kpBP = [0.0]
  CP.longitudinalTuning.kpV = [1.0]
  CP.longitudinalTuning.kiBP = [0.0]
  CP.longitudinalTuning.kiV = [0.0]
  CP.longitudinalTuning.kf = 1.0
  lc = LongControl(CP)

  shaper = HumanLongitudinalShaper(dt=0.05)
  v_ego = 16.0  # 57.6 km/h
  t_follow = 1.0
  stop_dist = 5.0

  # First accelerate positively for 0.5s
  for _ in range(10):
    _step_shaper(shaper, a_mpc=0.55, v_ego=v_ego, a_ego=0.50, d_rel=25.0, v_rel=0.2, a_lead=0.2, t_follow=t_follow, stop_dist=stop_dist)

  # Now MPC asks for -0.80 m/s^2 with mild closing (v_rel=-0.7, d_rel=24.0 -> d_headroom ~ +2.5..+3m vs desired_follow_distance)
  # Pass stale negative v_target_mpc = v_ego - 0.60 (simulating v_desired_filter lag) to ensure shaper neutralizes ghost P brake!
  targets = []
  lc_accels = []
  radar = SimpleNamespace(leadOne=SimpleNamespace(status=True, dRel=24.0))
  for _ in range(12):  # 0.6s within accel->coast dwell window
    res = _step_shaper(
      shaper, a_mpc=-0.80, v_ego=v_ego, a_ego=0.10, d_rel=24.0, v_rel=-0.7, a_lead=-0.45,
      t_follow=t_follow, stop_dist=stop_dist, v_target_mpc=v_ego - 0.60,
    )
    targets.append(res.a_target)
    cs = SimpleNamespace(
      vEgo=v_ego, aEgo=0.0, brakePressed=False, gasPressed=False,
      carrotCruise=False, softHoldActive=0, cruiseState=SimpleNamespace(standstill=False),
    )
    lp = SimpleNamespace(
      aTarget=res.a_target, vTargetNow=res.v_target_now, jTargetNow=res.j_target_now,
      shouldStop=False, hasLead=True, longitudinalPlanSource='lead0', fcw=False,
      cruiseTarget=100.0, cruiseCoastingTarget=0.0, cruiseCoastingPercent=0,
    )
    out_accel, _, _ = lc.update(True, cs, lp, [-3.5, 2.5], 0.0, radar)
    lc_accels.append(out_accel)

  # Should unwind gently into the natural coast band [-0.30, 0.05] rather than immediately slamming -0.8..-1.4 m/s^2 brakes
  assert min(targets) >= -0.30
  assert max(abs(t2 - t1) for t1, t2 in zip(targets, targets[1:])) <= 0.05
  assert min(lc_accels) >= -0.35


def test_cruising_gap_buffer_damps_small_hunting():
  """Near desired gap at 65 km/h, small positive MPC commands stay feathered inside buffer instead of chasing."""
  shaper = HumanLongitudinalShaper(dt=0.05)
  v_ego = 65.0 / 3.6
  # Seed steady cruising first (initialized at +0.05 m/s^2)
  _step_shaper(
    shaper, a_mpc=0.05, v_ego=v_ego, a_ego=0.05, d_rel=22.75, v_rel=0.0, a_lead=0.0,
    t_follow=0.90, stop_dist=5.0,
  )
  # Slightly above desired distance (+1.5m) with tiny v_rel=+0.15 m/s:
  # even if MPC jumps to +0.90 m/s^2, cruising buffer keeps output <= 0.45 m/s^2
  outputs = []
  for _ in range(10):
    res = _step_shaper(
      shaper, a_mpc=0.90, v_ego=v_ego, a_ego=0.05, d_rel=22.75, v_rel=0.15, a_lead=0.0,
      t_follow=0.90, stop_dist=5.0,
    )
    outputs.append(res.a_target)
  assert all(0.0 < val <= 0.45 for val in outputs)

  # Compute pre-MPC ramp target at 65 km/h vs 15 km/h launch
  t_cruise, max_cruise, tau_c, hr_c, _ = compute_accel_ramp_targets(
    v_ego, base_max_accel=1.10, has_lead=True, d_rel=22.75, v_rel=0.15, v_lead=v_ego + 0.15,
    a_lead_filt=0.0, t_follow=0.90, comfort_brake=2.5, stop_dist=5.0,
  )
  assert max_cruise == pytest.approx(1.10)
  assert t_cruise < 0.65
  assert tau_c >= 1.50
  assert hr_c <= 0.40


def test_anti_kickdown_at_cruising_vs_crisp_low_speed_launch():
  """Mid/high-speed cruising never inflates allowed_max_accel above base_max_accel; 0-20 km/h departure stays responsive."""
  # 60 km/h cruising with departing lead: no artificial 1.65-2.0 m/s^2 boost above base_max_accel (1.10)
  _, max_60, tau_60, hr_60, _ = compute_accel_ramp_targets(
    60.0 / 3.6, base_max_accel=1.10, has_lead=True, d_rel=30.0, v_rel=1.5, v_lead=65.4 / 3.6,
    a_lead_filt=1.0, t_follow=0.90, comfort_brake=2.5, stop_dist=5.0,
  )
  assert max_60 == pytest.approx(1.10)
  assert tau_60 >= 1.50

  # 10 km/h launch with departing lead: preserves strong initial response
  t_launch, max_launch, tau_launch, hr_launch, start_floor = compute_accel_ramp_targets(
    10.0 / 3.6, base_max_accel=1.40, has_lead=True, d_rel=8.0, v_rel=1.6, v_lead=15.8 / 3.6,
    a_lead_filt=1.2, t_follow=1.0, comfort_brake=2.5, stop_dist=5.0,
  )
  assert max_launch >= 1.55
  assert t_launch >= 1.30
  assert tau_launch <= 0.80
  assert hr_launch >= 0.65
  assert start_floor >= 1.0


@pytest.mark.parametrize("case", [
  # Hard braking behind decelerating lead (seg 29 style)
  {"a_mpc": -2.5, "v_ego": 15.0, "d_rel": 25.0, "v_rel": -5.0, "a_lead": -4.0},
  # Close cut-in at highway speed (seg 26 t=49.6s)
  {"a_mpc": -1.2, "v_ego": 22.6, "d_rel": 18.0, "v_rel": -4.2, "a_lead": 0.0},
  # Low-speed stop approach
  {"a_mpc": -0.6, "v_ego": 3.0, "d_rel": 6.5, "v_rel": -1.5, "v_lead": 1.5, "should_stop": True},
  # FCW / low predicted danger margin
  {"a_mpc": -0.9, "v_ego": 18.0, "d_rel": 24.0, "v_rel": -1.2, "fcw": True, "predicted_danger_margin": 1.2},
])
def test_urgent_braking_safety_invariant_never_lifted(case):
  shaper = HumanLongitudinalShaper(dt=0.05)
  # Seed with prior positive accel so both emergency drop & non-lifting are verified
  _step_shaper(shaper, a_mpc=0.5, v_ego=case["v_ego"], a_ego=0.3, d_rel=35.0, v_rel=0.5, a_lead=0.2)
  res = _step_shaper(shaper, **case)
  assert res.is_urgent is True
  # Downward urgent braking must never be lifted above a_mpc (and converges when not strengthened by high-speed pre-decel)
  for _ in range(15):
    res = _step_shaper(shaper, **case)
  assert res.a_target <= case["a_mpc"] + 1e-3
  assert res.v_target_now <= case["v_ego"] + case["a_mpc"] * 0.25 + 1e-3
  if case["v_ego"] * 3.6 <= 55.0:
    assert res.a_target == pytest.approx(case["a_mpc"], abs=1e-3)
    assert res.v_target_now == pytest.approx(case["v_ego"] + case["a_mpc"] * 0.25, abs=1e-3)


def test_natural_coast_accel_accounts_for_speed_and_slope():
  flat_low = natural_coast_accel(20.0 / 3.6, pitch=0.0)
  flat_high = natural_coast_accel(90.0 / 3.6, pitch=0.0)
  uphill = natural_coast_accel(60.0 / 3.6, pitch=0.04)    # nose up -> stronger deceleration
  downhill = natural_coast_accel(60.0 / 3.6, pitch=-0.04)  # nose down -> milder deceleration
  assert flat_high < flat_low < -0.10
  assert -0.40 <= uphill < natural_coast_accel(60.0 / 3.6, 0.0) < downhill <= -0.12
  # Non-finite safety
  assert -0.40 <= natural_coast_accel(float("nan"), float("nan")) <= -0.12


def test_reset_and_track_switch_clear_state_cleanly():
  shaper = HumanLongitudinalShaper(dt=0.05)
  _step_shaper(shaper, a_mpc=0.8, v_ego=15.0, a_ego=0.4, a_lead=1.5, track_id=10)
  assert shaper.initialized and shaper.track_id == 10

  # Switch track ID -> filter re-anchors immediately to new lead
  f_new = shaper.update_lead_filter(True, -1.2, track_id=11)
  assert shaper.track_id == 11
  assert f_new == pytest.approx(-1.2)

  # Reset via gas/brake/blended mode passes through raw MPC target
  res_reset = _step_shaper(shaper, a_mpc=-0.7, v_ego=15.0, a_ego=0.25, reset_state=True)
  assert not shaper.initialized
  assert res_reset.a_target == pytest.approx(-0.7)
  assert not res_reset.shaped
  assert math.isfinite(shaper.prev_a)


def _make_curve_model(v_model: float, curvature_fn):
  t_idxs = np.asarray(ModelConstants.T_IDXS, dtype=float)
  curv = np.asarray([float(curvature_fn(float(t))) for t in t_idxs], dtype=float)
  yaw_rate = curv * v_model
  dt = np.diff(t_idxs, prepend=0.0)
  heading = np.cumsum(yaw_rate * dt)
  x = np.cumsum(v_model * np.cos(heading) * dt)
  y = np.cumsum(v_model * np.sin(heading) * dt)
  x[0] = 0.0
  y[0] = 0.0
  z = np.zeros_like(t_idxs)
  vel_x = np.full_like(t_idxs, float(v_model))
  return SimpleNamespace(
    position=SimpleNamespace(x=x.tolist(), y=y.tolist(), z=z.tolist()),
    velocity=SimpleNamespace(x=vel_x.tolist()),
    orientationRate=SimpleNamespace(z=yaw_rate.tolist()),
  )


def test_cruise_source_deceleration_not_lifted_by_distant_lead():
  """Stage 1: mpc_source='cruise' curve/camera/bump decel must not be lifted to coast floor."""
  v_ego = 18.0
  common = dict(
    a_mpc=-0.85, v_ego=v_ego, a_ego=0.10, d_rel=48.0, v_rel=-0.3, a_lead=0.0,
    t_follow=1.0, stop_dist=5.0, mpc_mode="acc",
  )

  # First verify lead-driven source with same distant mild closing DOES dwell in coast band
  shaper_lead = HumanLongitudinalShaper(dt=0.05)
  _step_shaper(shaper_lead, a_mpc=0.5, v_ego=v_ego, a_ego=0.4, d_rel=48.0, v_rel=0.2, a_lead=0.1)
  lead_targets = [
    shaper_lead.shape(
      a_mpc=common["a_mpc"], v_target_mpc=v_ego - 0.5, j_target_mpc=0.0,
      v_ego=v_ego, a_ego=common["a_ego"], pitch=0.0, action_t=0.25,
      lead_status=True, d_rel=common["d_rel"], v_rel=common["v_rel"], v_lead=v_ego + common["v_rel"],
      a_lead=common["a_lead"], t_follow=common["t_follow"], comfort_brake=2.5,
      stop_dist=common["stop_dist"], should_stop=False, fcw=False, reset_state=False,
      mpc_mode="acc", mpc_source="lead0",
    ).a_target
    for _ in range(10)
  ]
  assert min(lead_targets) >= -0.30

  # Now with mpc_source='cruise', coast-floor must not suppress the commanded deceleration
  shaper_cruise = HumanLongitudinalShaper(dt=0.05)
  _step_shaper(shaper_cruise, a_mpc=0.5, v_ego=v_ego, a_ego=0.4, d_rel=48.0, v_rel=0.2, a_lead=0.1)
  cruise_targets = [
    shaper_cruise.shape(
      a_mpc=common["a_mpc"], v_target_mpc=v_ego - 0.5, j_target_mpc=0.0,
      v_ego=v_ego, a_ego=common["a_ego"], pitch=0.0, action_t=0.25,
      lead_status=True, d_rel=common["d_rel"], v_rel=common["v_rel"], v_lead=v_ego + common["v_rel"],
      a_lead=common["a_lead"], t_follow=common["t_follow"], comfort_brake=2.5,
      stop_dist=common["stop_dist"], should_stop=False, fcw=False, reset_state=False,
      mpc_mode="acc", mpc_source="cruise",
    ).a_target
    for _ in range(30)
  ]
  assert cruise_targets[-1] <= -0.80


def test_curve_entry_decel_straight_mild_and_moderate_curve():
  """Stage 2: silent on straight/mild sweep, gentle jerk-limited entry decel on moderate curve."""
  # Straight road at 80 km/h
  straight = _make_curve_model(80.0 / 3.6, lambda _t: 0.0)
  assert compute_curve_entry_decel_target(straight, 80.0 / 3.6)[0] == 0.0

  # Mild highway sweep at 80 km/h (R=500m -> lat accel ~0.99 m/s^2 < ~1.17 threshold)
  mild_sweep = _make_curve_model(80.0 / 3.6, lambda _t: 1.0 / 500.0)
  assert compute_curve_entry_decel_target(mild_sweep, 80.0 / 3.6)[0] == 0.0

  # Moderate curve ahead at 70 km/h (R=250m from t>=1.2s -> lat accel ~1.51 m/s^2)
  v_ego = 70.0 / 3.6
  moderate_curve = _make_curve_model(v_ego, lambda t: (1.0 / 250.0) if t >= 1.2 else 0.0)
  shaper = HumanLongitudinalShaper(dt=0.05)
  outputs = []
  res = None
  for _ in range(35):
    shaper.update_lead_filter(False, 0.0, -1)
    res = shaper.shape(
      a_mpc=0.0, v_target_mpc=v_ego, j_target_mpc=0.0,
      v_ego=v_ego, a_ego=0.0, pitch=0.0, action_t=0.25,
      lead_status=False, d_rel=0.0, v_rel=0.0, v_lead=0.0, a_lead=0.0,
      t_follow=1.3, comfort_brake=2.5, stop_dist=5.0, should_stop=False, fcw=False,
      reset_state=False, mpc_mode="acc", mpc_source="cruise", model_msg=moderate_curve,
    )
    outputs.append(res.a_target)

  assert res is not None
  assert -0.45 <= res.a_target <= -0.18
  assert res.curve_decel > 0.18
  assert res.shaped
  assert res.v_target_now <= v_ego + res.a_target * 0.25 + 1e-6
  # Asymmetric jerk slew check (never steeper than CURVE_ENTRY_ONSET_JERK * dt)
  step_drops = [a - b for a, b in zip(outputs, outputs[1:])]
  assert max(step_drops) <= CURVE_ENTRY_ONSET_JERK * 0.05 + 1e-4

  # Higher CurveInnerBias increases entry deceleration authority; TurnSpeedControlMode=0 disables it
  decel_bias0 = compute_curve_entry_decel_target(moderate_curve, v_ego, curve_inner_bias=0.0)[0]
  decel_bias20 = compute_curve_entry_decel_target(moderate_curve, v_ego, curve_inner_bias=20.0)[0]
  assert decel_bias20 > decel_bias0
  assert compute_curve_entry_decel_target(moderate_curve, v_ego, turn_speed_control_mode=0)[0] == 0.0


def test_curve_entry_roll_compensation_and_scurve_guard():
  """Banked curve reduces decel; off-camber increases decel; crowned straight and S-curve stay safe."""
  v_ego = 72.0 / 3.6  # 20 m/s
  left_curve = _make_curve_model(v_ego, lambda t: 0.0042 if t >= 0.8 else 0.0)  # +1.68 m/s^2 raw

  flat_decel = compute_curve_entry_decel_target(left_curve, v_ego, roll=0.0)[0]
  banked_decel = compute_curve_entry_decel_target(left_curve, v_ego, roll=0.05)[0]
  off_camber_decel = compute_curve_entry_decel_target(left_curve, v_ego, roll=-0.05)[0]
  assert 0.0 < banked_decel < flat_decel < off_camber_decel

  # Crowned straight road must never falsify a curve
  straight = _make_curve_model(v_ego, lambda _t: 0.0)
  assert compute_curve_entry_decel_target(straight, v_ego, roll=0.06)[0] == 0.0

  # Opposite-sign S-curve inflection suppresses current-bank contamination
  scurve_next = _make_curve_model(v_ego, lambda t: -0.0040 if t >= 1.0 else 0.0040)
  scurve_with_roll = compute_curve_entry_decel_target(
    scurve_next, v_ego, roll=0.08, current_curvature=0.0040,
  )[0]
  scurve_flat = compute_curve_entry_decel_target(
    scurve_next, v_ego, roll=0.0, current_curvature=0.0040,
  )[0]
  assert scurve_with_roll == pytest.approx(scurve_flat, abs=1e-6)


def test_curve_entry_no_double_braking_and_apex_exit_release():
  """Curve cap never stacks below stronger MPC braking and releases cleanly after the apex."""
  v_ego = 70.0 / 3.6
  sharp_curve = _make_curve_model(v_ego, lambda _t: 0.0055)
  shaper = HumanLongitudinalShaper(dt=0.05)

  # Ramp into curve first
  for _ in range(20):
    shaper.shape(
      a_mpc=-0.10, v_target_mpc=v_ego, j_target_mpc=0.0, v_ego=v_ego, a_ego=-0.10,
      pitch=0.0, action_t=0.25, lead_status=False, d_rel=0.0, v_rel=0.0, v_lead=0.0, a_lead=0.0,
      t_follow=1.3, comfort_brake=2.5, stop_dist=5.0, should_stop=False, fcw=False,
      reset_state=False, mpc_mode="acc", mpc_source="cruise", model_msg=sharp_curve,
    )

  # If vturn/MPC already commands stronger braking (-1.25 m/s^2), total output converges to -1.25 (never stacks)
  res = None
  for _ in range(25):
    res = shaper.shape(
      a_mpc=-1.25, v_target_mpc=v_ego - 0.30, j_target_mpc=0.0, v_ego=v_ego, a_ego=-1.25,
      pitch=0.0, action_t=0.25, lead_status=False, d_rel=0.0, v_rel=0.0, v_lead=0.0, a_lead=0.0,
      t_follow=1.3, comfort_brake=2.5, stop_dist=5.0, should_stop=False, fcw=False,
      reset_state=False, mpc_mode="acc", mpc_source="cruise", model_msg=sharp_curve,
    )
  assert res is not None
  assert res.a_target == pytest.approx(-1.25, abs=1e-3)

  # Apex-exit taper check: current_curvature=0.006 unwinding to 0.0033 ahead tapers below constant-0.0033 entry
  apex_unwind = _make_curve_model(v_ego, lambda _t: 0.0033)
  entry_decel = compute_curve_entry_decel_target(apex_unwind, v_ego, current_curvature=0.0)[0]
  exiting_decel = compute_curve_entry_decel_target(apex_unwind, v_ego, current_curvature=0.0060)[0]
  assert 0.0 <= exiting_decel < entry_decel

  # Once upcoming curvature drops to mild (0.0015), CurveEntryDecel releases smoothly back to 0.0
  straight_exit = _make_curve_model(v_ego, lambda _t: 0.0015)
  ce = CurveEntryDecel(dt=0.05)
  for _ in range(20):
    ce.update(sharp_curve, v_ego, current_curvature=0.0055)
  assert ce.decel > 0.25
  for _ in range(40):
    ce.update(straight_exit, v_ego, current_curvature=0.0060)
  assert ce.decel == pytest.approx(0.0, abs=1e-6)


def test_curve_entry_suppression_safety_and_blended_mode():
  """Lane change, blinker, lower limit, reset, non-finite inputs, and blended mode behave safely."""
  v_ego = 70.0 / 3.6
  curve = _make_curve_model(v_ego, lambda _t: 0.005)
  assert compute_curve_entry_decel_target(curve, v_ego, lane_change_active=True)[0] == 0.0
  assert compute_curve_entry_decel_target(curve, v_ego, blinker_active=True)[0] == 0.0
  assert compute_curve_entry_decel_target(curve, 38.0 / 3.6, lower_limit_kph=40.0)[0] == 0.0
  assert compute_curve_entry_decel_target(curve, v_ego, roll=float("nan"), current_curvature=float("nan"))[0] > 0.0
  assert compute_curve_entry_decel_target(curve, float("nan")) == (0.0, 0.0)

  # Blended mode applies gentle curve-entry ceiling while reset_state clears curve_entry state
  shaper = HumanLongitudinalShaper(dt=0.05)
  res = None
  for _ in range(25):
    res = shaper.shape(
      a_mpc=0.20, v_target_mpc=v_ego + 0.05, j_target_mpc=0.0, v_ego=v_ego, a_ego=0.0,
      pitch=0.0, action_t=0.25, lead_status=False, d_rel=0.0, v_rel=0.0, v_lead=0.0, a_lead=0.0,
      t_follow=1.3, comfort_brake=2.5, stop_dist=5.0, should_stop=False, fcw=False,
      reset_state=False, mpc_mode="blended", model_msg=curve,
    )
  assert res is not None
  assert res.a_target < 0.0
  assert res.curve_decel > 0.0
  assert res.shaped
  assert res.v_target_now <= v_ego + res.a_target * 0.25 + 1e-6

  res_reset = shaper.shape(
    a_mpc=0.20, v_target_mpc=v_ego, j_target_mpc=0.0, v_ego=v_ego, a_ego=0.0,
    pitch=0.0, action_t=0.25, lead_status=False, d_rel=0.0, v_rel=0.0, v_lead=0.0, a_lead=0.0,
    t_follow=1.3, comfort_brake=2.5, stop_dist=5.0, should_stop=False, fcw=False,
    reset_state=True, mpc_mode="acc", model_msg=curve,
  )
  assert shaper.curve_entry.decel == 0.0
  assert res_reset.curve_decel == 0.0
  assert res_reset.a_target == pytest.approx(0.20)


def test_severe_curve_entry_escalates_past_gentle_cap_for_tight_ramp():
  """Seg 9 style tight ramp at 98 km/h (even with factor=0.95) escalates cleanly >1.0 m/s^2 while 80 km/h R=500m stays 0."""
  mild_sweep = _make_curve_model(80.0 / 3.6, lambda _t: 1.0 / 500.0)
  assert compute_curve_entry_decel_target(mild_sweep, 80.0 / 3.6, curve_speed_factor=1.2)[0] == 0.0

  v_ego = 98.0 / 3.6
  # Tightening ramp 3s..6s ahead (R=200m -> R=110m)
  tight_ramp = _make_curve_model(v_ego, lambda t: 0.0 if t < 1.5 else (1.0 / 200.0 if t < 3.5 else 1.0 / 110.0))
  ce = CurveEntryDecel(dt=0.05)
  vals = []
  for _ in range(30):
    vals.append(ce.update(
      tight_ramp, v_ego, current_curvature=0.0,
      curve_inner_bias=10.0, curve_speed_factor=0.95, lower_limit_kph=35.0,
    ))
  assert vals[-1] >= 1.15
  assert vals[-1] <= 1.56
  step_rises = [b - a for a, b in zip(vals, vals[1:])]
  assert max(step_rises) <= CURVE_ENTRY_SEVERE_ONSET_JERK * 0.05 + 1e-4


def test_distant_highway_lead_never_triggers_false_urgent_or_165_slam():
  """Replays seg 40 (129 km/h, V, d=104m, vr=-4.4) and seg 49 (89.5 km/h, R, d=65m, vr=-7.2, al=+1.42)."""
  # Case A: seg 40 t=2433.8 — 129.2 km/h, vision lead at 103.7m, vr=-4.4, a_mpc=-0.20 (no braking panic needed!)
  shaper_a = HumanLongitudinalShaper(dt=0.05)
  _step_shaper(shaper_a, a_mpc=0.15, v_ego=129.0 / 3.6, a_ego=0.15, lead_status=False)
  res_a = None
  outputs_a = []
  for _ in range(12):
    shaper_a.update_lead_filter(True, -0.02, -1)
    res_a = shaper_a.shape(
      a_mpc=-0.20, v_target_mpc=129.0 / 3.6, j_target_mpc=0.0,
      v_ego=129.2 / 3.6, a_ego=0.15, pitch=0.0, action_t=0.25,
      lead_status=True, d_rel=103.7, v_rel=-4.4, v_lead=129.2 / 3.6 - 4.4, a_lead=-0.02,
      t_follow=1.0, comfort_brake=2.5, stop_dist=5.0,
      should_stop=False, fcw=False, reset_state=False, mpc_mode="acc",
      lead_radar=False,
    )
    outputs_a.append(res_a.a_target)
  assert res_a is not None
  assert not res_a.is_urgent
  assert min(outputs_a) > -0.90  # Never slams -1.65

  # Case B: seg 49 t=2962.9 — 89.5 km/h, radar lead at 65m accelerating strongly away (al=+1.42)
  shaper_b = HumanLongitudinalShaper(dt=0.05)
  _step_shaper(shaper_b, a_mpc=0.05, v_ego=89.5 / 3.6, a_ego=0.05, lead_status=False)
  outputs_b = []
  res_b = None
  for _ in range(12):
    shaper_b.update_lead_filter(True, 1.42, 0)
    res_b = shaper_b.shape(
      a_mpc=-0.45, v_target_mpc=89.5 / 3.6 - 0.1, j_target_mpc=0.0,
      v_ego=89.5 / 3.6, a_ego=0.05, pitch=0.0, action_t=0.25,
      lead_status=True, d_rel=65.0, v_rel=-7.2, v_lead=89.5 / 3.6 - 7.2, a_lead=1.42,
      t_follow=1.0, comfort_brake=2.5, stop_dist=5.0,
      should_stop=False, fcw=False, reset_state=False, mpc_mode="acc",
      lead_radar=True,
    )
    outputs_b.append(res_b.a_target)
  assert res_b is not None
  assert not res_b.is_urgent
  assert min(outputs_b) > -1.00
  drops_b = [(a - b) / 0.05 for a, b in zip(outputs_b, outputs_b[1:])]
  assert max(drops_b) <= 1.25


def test_sudden_lead_acquisition_and_lane_change_shock_damping():
  """Post-reset anchors smoothly to aEgo, sudden dRel drop resets lead_age, while close cut-in brakes immediately."""
  shaper = HumanLongitudinalShaper(dt=0.05)
  v_ego = 123.0 / 3.6

  # 1) Simulate gasPressed=True (reset_state=True) at a_ego=+0.38 m/s^2, followed by sudden -2.85 m/s^2 MPC request
  #    with left blinker active and comfortable physical gap (d=58m, ttc=4.68s) -> must anchor at +0.38 and damp shock!
  res_reset = shaper.shape(
    a_mpc=0.38, v_target_mpc=v_ego, j_target_mpc=0.0, v_ego=v_ego, a_ego=0.38,
    pitch=0.0, action_t=0.25, lead_status=True, d_rel=62.0, v_rel=-12.0, v_lead=v_ego - 12.0, a_lead=0.0,
    t_follow=1.0, comfort_brake=2.5, stop_dist=5.0, should_stop=False, fcw=False,
    reset_state=True, mpc_mode="acc", blinker_active=True, lead_radar=True,
  )
  assert res_reset.a_target == pytest.approx(0.38)

  first_active = shaper.shape(
    a_mpc=-2.85, v_target_mpc=v_ego - 0.7, j_target_mpc=0.0, v_ego=v_ego, a_ego=0.38,
    pitch=0.0, action_t=0.25, lead_status=True, d_rel=58.0, v_rel=-12.4, v_lead=v_ego - 12.4, a_lead=-0.4,
    t_follow=1.0, comfort_brake=2.5, stop_dist=5.0, should_stop=False, fcw=False,
    reset_state=False, mpc_mode="acc", blinker_active=True, lead_radar=True,
  )
  # Anchored to +0.38 (no instant cliff to -2.85 / -20 m/s^3)
  assert first_active.a_target > -0.10

  # 2) Track sudden target switch when radarTrackId stays 0 (EnableRadarTracks==0 style dRel jump 80m -> 49m)
  shaper2 = HumanLongitudinalShaper(dt=0.05)
  for _ in range(30):
    shaper2.update_lead_filter(True, 0.0, 0)
    shaper2.shape(
      a_mpc=0.0, v_target_mpc=30.0, j_target_mpc=0.0, v_ego=30.0, a_ego=0.0,
      pitch=0.0, action_t=0.25, lead_status=True, d_rel=80.0, v_rel=0.0, v_lead=30.0, a_lead=0.0,
      t_follow=1.0, comfort_brake=2.5, stop_dist=5.0, should_stop=False, fcw=False,
      reset_state=False, mpc_mode="acc", lead_radar=True,
    )
  assert shaper2.lead_age > 1.0
  shaper2.update_lead_filter(True, -0.5, 0)
  res_jump = shaper2.shape(
    a_mpc=-1.60, v_target_mpc=29.6, j_target_mpc=0.0, v_ego=30.0, a_ego=0.0,
    pitch=0.0, action_t=0.25, lead_status=True, d_rel=49.0, v_rel=-7.0, v_lead=23.0, a_lead=-0.5,
    t_follow=1.0, comfort_brake=2.5, stop_dist=5.0, should_stop=False, fcw=False,
    reset_state=False, mpc_mode="acc", lead_radar=True,
  )
  assert shaper2.lead_age == 0.0
  assert res_jump.a_target > -0.25

  # 3) Safety contrast: genuine close cut-in (d=18m, v_ego=22.6 m/s, vr=-6.5 -> ttc=2.77s) must brake immediately!
  shaper3 = HumanLongitudinalShaper(dt=0.05)
  _step_shaper(shaper3, a_mpc=0.3, v_ego=22.6, a_ego=0.3, lead_status=False)
  shaper3.update_lead_filter(True, -1.0, 99)
  res_emerg = shaper3.shape(
    a_mpc=-2.50, v_target_mpc=21.9, j_target_mpc=0.0, v_ego=22.6, a_ego=0.3,
    pitch=0.0, action_t=0.25, lead_status=True, d_rel=18.0, v_rel=-6.5, v_lead=16.1, a_lead=-1.0,
    t_follow=1.0, comfort_brake=2.5, stop_dist=5.0, should_stop=False, fcw=False,
    reset_state=False, mpc_mode="acc", lead_radar=True,
  )
  assert res_emerg.is_urgent
  assert res_emerg.a_target <= 0.01  # Drops at full -6.0 m/s^3 (0.3 -> 0.0 on very first step)


def test_crisp_brake_release_when_lead_clears():
  """Replays seg 38 t=2317: when lead vanishes from -1.27 m/s^2, braking unwinds crisply (>=2.0 m/s^3) to coast."""
  shaper = HumanLongitudinalShaper(dt=0.05)
  v_ego = 120.0 / 3.6
  # Seed heavy braking at -1.27 m/s^2
  shaper.reset(-1.27)
  shaper.shape(
    a_mpc=-1.27, v_target_mpc=v_ego - 0.3, j_target_mpc=0.0, v_ego=v_ego, a_ego=-1.27,
    pitch=0.0, action_t=0.25, lead_status=True, d_rel=45.0, v_rel=-5.0, v_lead=v_ego - 5.0, a_lead=0.0,
    t_follow=1.0, comfort_brake=2.5, stop_dist=5.0, should_stop=False, fcw=False,
    reset_state=False, mpc_mode="acc", lead_radar=True,
  )

  # Lead clears: MPC asks for +0.5 m/s^2; should release to >= natural coast within 0.50s (10 steps)
  outputs = []
  for _ in range(10):
    shaper.update_lead_filter(False, 0.0, -1)
    res = shaper.shape(
      a_mpc=0.50, v_target_mpc=v_ego + 0.1, j_target_mpc=0.0, v_ego=v_ego, a_ego=-0.5,
      pitch=0.0, action_t=0.25, lead_status=False, d_rel=0.0, v_rel=0.0, v_lead=0.0, a_lead=0.0,
      t_follow=1.0, comfort_brake=2.5, stop_dist=5.0, should_stop=False, fcw=False,
      reset_state=False, mpc_mode="acc", lead_radar=False,
    )
    outputs.append(res.a_target)

  assert outputs[-1] >= -0.35  # Released to coast band inside 0.5s (vs >1.5s hang at 0.65 m/s^3)
  assert (outputs[1] - outputs[0]) / 0.05 >= 2.0


def test_high_speed_slowing_lead_and_cutout_stopped_car_brakes_promptly():
  """High-speed closing/braking leads, vision-to-radar stopped cars, and cut-out reveals brake promptly without -1.25 clamping."""
  # Case 1: 85 km/h closing on slowing vision lead at d=66m (a_lead=-1.5) triggers timely braking >= -1.40 within 0.50s
  shaper1 = HumanLongitudinalShaper(dt=0.05)
  v1 = 85.0 / 3.6
  for _ in range(5):
    _step_shaper(shaper1, a_mpc=0.15, v_ego=v1, a_ego=0.15, lead_status=False)
  outs1 = []
  for i in range(10):
    shaper1.update_lead_filter(True, -1.5, 1)
    res1 = shaper1.shape(
      a_mpc=-1.55, v_target_mpc=v1 - 0.3, j_target_mpc=0.0, v_ego=v1, a_ego=0.15 if i == 0 else outs1[-1],
      pitch=0.0, action_t=0.25, lead_status=True, d_rel=66.0 - i * 0.4, v_rel=-8.3, v_lead=v1 - 8.3, a_lead=-1.5,
      t_follow=0.92, comfort_brake=2.5, stop_dist=5.0, should_stop=False, fcw=False,
      reset_state=False, mpc_mode="acc", lead_radar=False,
    )
    outs1.append(res1.a_target)
  assert res1.is_urgent
  assert outs1[-1] <= -1.40

  # Case 2: seg 19 t=57.5..58.3 style: accelerating +0.85, vision lead closes -> crisp lift-off, then stopped radar at 38.8m -> <= -2.0 inside 0.35s
  shaper2 = HumanLongitudinalShaper(dt=0.05)
  v2 = 43.0 / 3.6
  for _ in range(5):
    _step_shaper(shaper2, a_mpc=0.85, v_ego=v2, a_ego=0.85, lead_status=False)
  lift_outs = []
  for i in range(6):
    shaper2.update_lead_filter(True, -0.6, -1)
    r = shaper2.shape(
      a_mpc=-0.50, v_target_mpc=v2, j_target_mpc=0.0, v_ego=v2, a_ego=0.85,
      pitch=0.0, action_t=0.25, lead_status=True, d_rel=50.0 - i * 0.25, v_rel=-5.0, v_lead=v2 - 5.0, a_lead=-0.6,
      t_follow=1.0, comfort_brake=2.5, stop_dist=5.0, should_stop=False, fcw=False,
      reset_state=False, mpc_mode="acc", lead_radar=False,
    )
    lift_outs.append(r.a_target)
  assert lift_outs[-1] <= 0.15

  stop_outs = []
  for i in range(7):
    shaper2.update_lead_filter(True, 0.0, 2)
    r = shaper2.shape(
      a_mpc=-2.80, v_target_mpc=v2 - 0.5, j_target_mpc=0.0, v_ego=v2, a_ego=-0.5,
      pitch=0.0, action_t=0.25, lead_status=True, d_rel=38.8 - i * 0.55, v_rel=0.4 - v2, v_lead=0.4, a_lead=0.0,
      t_follow=1.0, comfort_brake=2.5, stop_dist=5.0, should_stop=False, fcw=False,
      reset_state=False, mpc_mode="acc", lead_radar=True,
    )
    stop_outs.append(r.a_target)
  assert stop_outs[-1] <= -2.0

  # Case 3: lead cut-out suddenly reveals stopped/slow car ahead at 58m (v_ego=23 m/s, v_lead=2.0 m/s) -> must NOT clamp at -1.25
  shaper3 = HumanLongitudinalShaper(dt=0.05)
  v3 = 23.0
  for _ in range(6):
    _step_shaper(shaper3, a_mpc=0.20, v_ego=v3, a_ego=0.20, lead_status=True, d_rel=75.0, v_rel=0.0, v_lead=v3, track_id=10)
  cutout_outs = []
  for i in range(8):
    shaper3.update_lead_filter(True, 0.0, 11)
    r = shaper3.shape(
      a_mpc=-2.60, v_target_mpc=v3 - 0.6, j_target_mpc=0.0, v_ego=v3, a_ego=0.20 if i == 0 else cutout_outs[-1],
      pitch=0.0, action_t=0.25, lead_status=True, d_rel=58.0 - i * 1.0, v_rel=2.0 - v3, v_lead=2.0, a_lead=0.0,
      t_follow=0.92, comfort_brake=2.5, stop_dist=5.0, should_stop=False, fcw=False,
      reset_state=False, mpc_mode="acc", lead_radar=True,
    )
    cutout_outs.append(r.a_target)
  assert shaper3.lead_age < 0.5
  assert cutout_outs[3] < -1.25
  assert cutout_outs[-1] <= -2.40


def test_tesla_traffic_active_bypasses_shaper():
  from openpilot.selfdrive.controls.lib.human_longitudinal import HumanLongitudinalShaper

  shaper = HumanLongitudinalShaper()
  # Build up positive acceleration state first
  for _ in range(10):
    shaper.shape(
      a_mpc=0.9, v_target_mpc=16.0, j_target_mpc=0.5,
      v_ego=15.0, a_ego=0.8, pitch=0.0, action_t=0.05,
      lead_status=False, d_rel=100.0, v_rel=0.0, v_lead=20.0, a_lead=0.0,
      t_follow=1.2, comfort_brake=2.4, stop_dist=4.0,
      should_stop=False, fcw=False, reset_state=False,
    )

  # When LongitudinalPlanner activates tesla_traffic_active -> passes reset_state=True,
  # shaper must immediately return MPC deceleration without ramp lag.
  res = shaper.shape(
    a_mpc=-2.5, v_target_mpc=12.5, j_target_mpc=-3.0,
    v_ego=15.0, a_ego=0.0, pitch=0.0, action_t=0.05,
    lead_status=False, d_rel=100.0, v_rel=0.0, v_lead=20.0, a_lead=0.0,
    t_follow=1.2, comfort_brake=2.4, stop_dist=4.0,
    should_stop=True, fcw=False, reset_state=True,
  )
  assert res.a_target == -2.5
  assert res.v_target_now == 12.5
  assert res.j_target_now == -3.0
  assert res.shaped is False
  assert res.is_urgent is True

  # Immediately after reset_state releases, internal state starts from a_ego (0.0), not prior +0.9,
  # and ramps down at urgent rate (-0.4 on first 0.05s step).
  res_after = shaper.shape(
    a_mpc=-2.5, v_target_mpc=12.4, j_target_mpc=-3.0,
    v_ego=14.9, a_ego=-1.5, pitch=0.0, action_t=0.05,
    lead_status=False, d_rel=100.0, v_rel=0.0, v_lead=20.0, a_lead=0.0,
    t_follow=1.2, comfort_brake=2.4, stop_dist=4.0,
    should_stop=True, fcw=False, reset_state=False,
  )
  assert res_after.a_target <= -0.35

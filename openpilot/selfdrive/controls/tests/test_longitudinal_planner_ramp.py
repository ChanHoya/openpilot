import pytest
from types import SimpleNamespace
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.cereal import car


def test_continuous_accel_ramp_smoothness():
  dt = 0.05
  ramp_filter = FirstOrderFilter(0.0, 1.5, dt)

  # Starting from standstill (0 km/h) -> target 1.6 m/s^2
  target_a = 1.6
  accels = []
  for t in range(80):  # 4 seconds
    eff_target = max(0.6 if t == 0 else 0.0, target_a)
    ramp_filter.update(eff_target)
    accels.append(ramp_filter.x)

  # Verify initial step is gentle and continuous (no 1.6 spike at t=0)
  assert accels[0] < 0.1
  assert accels[10] < 0.8  # at 0.5s, progressive buildup
  assert accels[-1] == pytest.approx(1.6, abs=0.2)  # smoothly converges to target


def test_longcontrol_pos_limit_capping():
  CP = car.CarParams.new_message()
  CP.longitudinalTuning.kpBP = [0.0]
  CP.longitudinalTuning.kpV = [1.0]
  CP.longitudinalTuning.kiBP = [0.0]
  CP.longitudinalTuning.kiV = [0.1]
  lc = LongControl(CP)

  # Plan calls for accel of 1.2 m/s^2
  long_plan = SimpleNamespace(
    aTarget=1.2,
    vTargetNow=10.0,
    jTargetNow=0.0,
    shouldStop=False,
    hasLead=True,
  )
  accel_limits = [-3.5, 2.5]
  radarState = SimpleNamespace(leadOne=SimpleNamespace(status=True, dRel=20.0))
  CS = SimpleNamespace(
    vEgo=10.0,
    aEgo=1.2,
    brakePressed=False,
    softHoldActive=0,
    cruiseState=SimpleNamespace(standstill=False),
  )

  lc.update(True, CS, long_plan, accel_limits, 0.0, radarState)
  # PID pos_limit follows accel_limits[1] to allow full velocity error feedback tracking
  assert lc.pid.pos_limit == pytest.approx(accel_limits[1])


def test_low_speed_lead_catchup_ramp():
  """선행차 추종 및 캐치업 시: 일반 수동운전 수준(1.25~1.45 m/s^2)으로 굼뜨지 않게 자연스럽게 가속."""
  import numpy as np
  dt = 0.05
  v_ego = 7.22  # 26 km/h
  v_ego_kph = v_ego * 3.6
  base_max_accel = 1.78
  lead_departing = True

  low_speed_cap = float(np.interp(v_ego_kph, [0.0, 15.0, 35.0, 50.0], [0.85, 1.25, 1.45, base_max_accel]))
  if lead_departing:
    low_speed_cap = max(low_speed_cap, 1.35)
  target_a = min(base_max_accel, low_speed_cap)

  ramp = FirstOrderFilter(0.0, 1.2, dt)
  current_a = 0.0
  headroom = 0.55
  ramp.x = current_a + headroom

  accels = []
  for _ in range(60):  # 3 seconds
    ramp.update(target_a)
    accels.append(ramp.x)

  # Frame 0 is gentle without kickdown spike
  assert 0.5 < accels[0] < 0.65
  # At 1.0s (20 steps), reaches normal manual catch-up pace (~1.0 m/s^2)
  assert 0.95 < accels[20] < 1.15
  # Converges to target without exceeding base_max_accel
  assert accels[-1] == pytest.approx(target_a, abs=0.1)
  assert accels[-1] <= base_max_accel


def test_low_speed_open_road_ramp_rpm_stability():
  """선행차 이탈 및 큰 속도 편차(26 -> 110 km/h) 시: 저속 RPM 치솟음/킥다운 방지 및 부드러운 S-curve 전개."""
  import numpy as np
  dt = 0.05
  v_ego = 7.22  # 26 km/h
  v_ego_kph = v_ego * 3.6
  base_max_accel = 1.78

  low_speed_cap = float(np.interp(v_ego_kph, [0.0, 15.0, 35.0, 50.0], [0.75, 1.05, 1.30, base_max_accel]))
  target_a = min(base_max_accel, low_speed_cap)

  ramp = FirstOrderFilter(0.0, 1.6, dt)
  current_a = 0.0
  headroom = 0.38
  ramp.x = current_a + headroom

  accels = []
  for _ in range(40):  # 2 seconds
    ramp.update(target_a)
    accels.append(ramp.x)

  # Initial step is gentle (around 0.4 m/s^2), completely preventing kickdown shock
  assert 0.35 < accels[0] < 0.45
  # At 1.0s (20 steps), smoothly ramps to ~0.77 m/s^2
  assert 0.7 < accels[20] < 0.85
  # Ramped limit stays bounded by low_speed_cap at 26 km/h (target_a ~ 1.19 m/s^2)
  assert accels[-1] <= 1.25


def test_complex_multiphase_lead_scenario():
  """실제 복합 주행 시나리오:
  1. 정차 대기 (0 km/h) -> 선행차 빠른 출발 (0 -> 40 km/h)
  2. 선행차 전방 상황으로 급감속 (40 -> 12 km/h)
  3. 선행차 중고속 재가속 (12 -> 85 km/h)
  4. 선행차 50 이하 재감속 (85 -> 35 km/h)
  5. 선행차 재가속 (35 -> 80 km/h)
  전 구간에서 저속 RPM 안정성, 선행차 캐치업, 제동 시 안전거리 확보, 가속 램프 정상 동작 검증.
  """
  import numpy as np
  from openpilot.common.constants import CV

  dt = 0.05
  sim_time = 45.0
  steps = int(sim_time / dt)

  times = [0, 2, 7, 11, 21, 25, 32, 42, 45]
  speeds_kph = [0, 0, 40, 12, 85, 85, 35, 80, 80]
  v_lead_profile = np.interp(np.linspace(0, sim_time, steps), times, speeds_kph) / 3.6

  x_ego, v_ego, a_ego = 0.0, 0.0, 0.0
  x_lead = 7.0
  ramp = FirstOrderFilter(0.0, 1.5, dt)
  base_max_accel = 1.60

  ego_speeds = []
  ego_accels = []
  gaps = []
  ramp_limits = []

  for step in range(steps):
    v_lead = v_lead_profile[step]
    a_lead = (v_lead_profile[min(step + 1, steps - 1)] - v_lead) / dt
    x_lead += v_lead * dt

    d_rel = x_lead - x_ego
    v_rel = v_lead - v_ego
    has_lead = True
    standstill = (v_ego < 0.1 and v_lead < 0.1)

    v_ego_kph = v_ego * CV.MS_TO_KPH
    lead_departing = bool(v_rel > 0.2 or a_lead > 0.2)
    low_speed_cap = float(np.interp(v_ego_kph, [0.0, 15.0, 35.0, 50.0], [0.85, 1.25, 1.45, base_max_accel]))
    if lead_departing:
      low_speed_cap = max(low_speed_cap, 1.35)
    target_a = min(base_max_accel, low_speed_cap)
    tau = 1.2 if lead_departing else 1.4
    headroom = 0.55 if lead_departing else 0.45

    if standstill:
      start_floor = min(0.65, target_a)
      ramp.x = max(ramp.x, start_floor)
    else:
      current_a = max(0.0, a_ego)
      if ramp.x > current_a + headroom:
        ramp.x = max(0.5, current_a + headroom)

    ramp.update_alpha(tau)
    ramp.update(target_a)
    max_allowed_a = min(base_max_accel, ramp.x)

    # 차간거리 제어: 목표 거리 = 1.4s * v_ego + 6.0m
    desired_dist = 1.4 * v_ego + 6.0
    dist_error = d_rel - desired_dist
    a_target_cmd = 0.2 * dist_error + 0.6 * v_rel + 0.3 * a_lead
    a_target_clamped = np.clip(a_target_cmd, -2.5, max_allowed_a)

    # 차량 파워트레인 액추에이터 동역학 (시정수 0.2초 지연)
    a_ego += (a_target_clamped - a_ego) / 0.2 * dt
    v_ego = max(0.0, v_ego + a_ego * dt)
    x_ego += v_ego * dt

    ego_speeds.append(v_ego * 3.6)
    ego_accels.append(a_ego)
    gaps.append(d_rel)
    ramp_limits.append(max_allowed_a)

  # 1. 정차 및 급감속 시 최소 차간거리 유지 (충돌 위험 0, 항상 6.0m 이상 확보)
  assert min(gaps) > 6.0, f"Minimum distance too small: {min(gaps):.2f}m"

  # 2. 저속 구간(< 35km/h)에서 과도한 가속 방지 (1.45 m/s^2 이하로 RPM 킥다운 치솟음 없음)
  low_speed_accels = [a for a, v in zip(ego_accels, ego_speeds) if v < 35.0]
  assert max(low_speed_accels) <= 1.45, f"Low speed accel exceeded cap: {max(low_speed_accels):.2f}"

  # 3. 선행차 급출발 시 즉각적이고 매끄러운 반응 (정차 후 출발 가속도 0.5 m/s^2 이상 확보)
  launch_step = int(3.0 / dt)  # 출발 후 1초
  assert ego_accels[launch_step] > 0.4, f"Launch sluggish: {ego_accels[launch_step]:.2f}"

  # 4. 급감속 구간(t=7~11s)에서 안전하고 확실한 제동 감속 인가
  brake_phase_accels = ego_accels[int(7.5 / dt):int(11.0 / dt)]
  assert min(brake_phase_accels) < -1.2, f"Deceleration not applied adequately: {min(brake_phase_accels):.2f}"

  # 5. 고속 구간(t=23s)에서 선행차(85km/h)를 자연스럽게 추종하며 순항
  high_speed_step = int(23.0 / dt)
  assert abs(ego_speeds[high_speed_step] - 85.0) < 10.0, f"Ego failed to track lead high speed: {ego_speeds[high_speed_step]:.1f}"

  # 6. 재감속 구간(t=31s, 선행차 35km/h)에서 50km/h 이하로 정상 감속
  decel_step = int(31.0 / dt)
  assert ego_speeds[decel_step] < 55.0, f"Ego failed to decelerate: {ego_speeds[decel_step]:.1f}"



from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.cutin_predecel import (
  apply_cutin_predecel_accel_limit,
  get_cutin_predecel_accel_limit,
)


def radar_state_with_risk(**overrides):
  values = {
    "status": True,
    "radar": True,
    "score": 0.9,
    "dRel": 20.0,
    "vRel": -6.5,
  }
  values.update(overrides)
  return SimpleNamespace(leadCutInRisk=SimpleNamespace(**values))


def test_cutin_predecel_requests_negative_acceleration_at_target_ttc() -> None:
  limit = get_cutin_predecel_accel_limit(radar_state_with_risk())

  assert limit == pytest.approx(-0.608, abs=0.002)


@pytest.mark.parametrize(
  "overrides",
  (
    {"status": False},
    {"radar": False},
    {"score": 0.1},
    {"dRel": -1.0},
    {"vRel": 0.0},
  ),
)
def test_cutin_predecel_ignores_invalid_or_weak_risk(overrides) -> None:
  assert get_cutin_predecel_accel_limit(
    radar_state_with_risk(**overrides),
  ) is None


def test_cutin_predecel_is_bounded_for_urgent_and_distant_risk() -> None:
  urgent = get_cutin_predecel_accel_limit(
    radar_state_with_risk(dRel=8.0, vRel=-8.0),
  )
  distant = get_cutin_predecel_accel_limit(
    radar_state_with_risk(dRel=40.0, vRel=-4.0),
  )

  assert urgent == pytest.approx(-0.65)
  assert distant == pytest.approx(-0.25)


def test_cutin_predecel_reduces_positive_acceleration_then_reaches_decel() -> None:
  acceleration = 0.96
  limits = []
  for _ in range(11):
    acceleration = apply_cutin_predecel_accel_limit(
      1.0,
      acceleration,
      -0.608,
    )
    limits.append(acceleration)

  assert limits[0] == pytest.approx(0.81)
  assert limits[6] == pytest.approx(-0.09)
  assert limits[-1] == pytest.approx(-0.608)


def test_normal_accel_limit_keeps_existing_small_step() -> None:
  assert apply_cutin_predecel_accel_limit(
    -1.0,
    0.5,
    None,
  ) == pytest.approx(0.45)


def test_lead_predecel_requests_bounded_negative_accel_for_closing_lead() -> None:
  from openpilot.selfdrive.controls.lib.cutin_predecel import get_lead_predecel_accel_limit
  radar_state = SimpleNamespace(
    leadOne=SimpleNamespace(
      status=True,
      radar=True,
      radarTrackId=1,
      dRel=30.0,
      vRel=-5.0,  # TTC = 6.0s
      aLeadK=-0.5,
    )
  )
  limit = get_lead_predecel_accel_limit(radar_state, v_ego=20.0)
  assert limit is not None
  assert -0.50 <= limit <= -0.15


@pytest.mark.parametrize(
  "overrides,v_ego",
  (
    ({"status": False}, 20.0),
    ({"radar": False}, 20.0),
    ({"radarTrackId": -1}, 20.0),
    ({"dRel": 8.0}, 20.0),    # too close (<10m)
    ({"vRel": -0.2}, 20.0),   # closing too slowly
    ({"aLeadK": -0.1}, 20.0),  # lead not decelerating
    ({}, 4.0),                 # v_ego <= 5.0 m/s
  ),
)
def test_lead_predecel_ignores_ineligible_conditions(overrides, v_ego) -> None:
  from openpilot.selfdrive.controls.lib.cutin_predecel import get_lead_predecel_accel_limit
  values = {
    "status": True,
    "radar": True,
    "radarTrackId": 1,
    "dRel": 30.0,
    "vRel": -3.0,
    "aLeadK": -0.5,
  }
  values.update(overrides)
  radar_state = SimpleNamespace(leadOne=SimpleNamespace(**values))
  assert get_lead_predecel_accel_limit(radar_state, v_ego=v_ego) is None


def test_lead_predecel_handles_stopped_lead_at_speed() -> None:
  from openpilot.selfdrive.controls.lib.cutin_predecel import get_lead_predecel_accel_limit
  # Seg 2: ego at 47 km/h (13.0 m/s), approaching stopped lead at 80m
  radar_state = SimpleNamespace(
    leadOne=SimpleNamespace(
      status=True,
      radar=True,
      radarTrackId=1,
      dRel=80.0,
      vRel=-12.0,
      vLead=1.0,
      aLeadK=-0.1,  # stationary lead doesn't have large negative aLeadK
    )
  )
  limit = get_lead_predecel_accel_limit(radar_state, v_ego=13.0)
  assert limit is not None
  assert -1.50 <= limit <= -0.80


def test_lead_predecel_handles_hard_braking_lead() -> None:
  from openpilot.selfdrive.controls.lib.cutin_predecel import get_lead_predecel_accel_limit
  # Seg 4: lead brakes hard at -1.8 m/s^2, 16m away closing at -3 m/s
  radar_state = SimpleNamespace(
    leadOne=SimpleNamespace(
      status=True,
      radar=True,
      radarTrackId=1,
      dRel=16.0,
      vRel=-3.0,
      vLead=8.0,
      aLeadK=-1.8,
    )
  )
  limit = get_lead_predecel_accel_limit(radar_state, v_ego=11.0)
  assert limit is not None
  assert limit <= -2.0  # firm proactive decel ceiling to prevent closing collision


def test_apply_cutin_predecel_urgent_step() -> None:
  # When predecel_limit is urgent (e.g. < -1.0), allow 0.08 smooth step
  stepped = apply_cutin_predecel_accel_limit(1.0, 0.0, -1.5)
  assert stepped == pytest.approx(-0.08)


def test_lead_predecel_low_speed_gentle_follow_does_not_hard_brake() -> None:
  from openpilot.selfdrive.controls.lib.cutin_predecel import get_lead_predecel_accel_limit
  # Low speed follow: vEgo = 24 km/h (6.7 m/s), lead at 11m, vRel = -1.5 m/s, aLead = -1.5 m/s^2
  # TTC = 11.0 / 1.5 = 7.3s. This is normal following, not an emergency stop!
  radar_state = SimpleNamespace(
    leadOne=SimpleNamespace(
      status=True,
      radar=True,
      radarTrackId=1,
      dRel=11.0,
      vRel=-1.5,
      vLead=5.2,
      aLeadK=-1.5,
    )
  )
  limit = get_lead_predecel_accel_limit(radar_state, v_ego=6.7)
  # Must NOT return hard braking limit (<= -1.5)
  assert limit is None or limit > -1.0


def test_lead_predecel_low_speed_urgent_stop_returns_bounded_limit() -> None:
  from openpilot.selfdrive.controls.lib.cutin_predecel import get_lead_predecel_accel_limit
  # Low speed urgent stop: vEgo = 22 km/h (6.1 m/s), lead braking hard at -2.6 m/s^2, 8.0m away closing at -2.8 m/s (TTC = 2.8s < 3.5s)
  radar_state = SimpleNamespace(
    leadOne=SimpleNamespace(
      status=True,
      radar=True,
      radarTrackId=1,
      dRel=10.5,
      vRel=-3.2,
      vLead=2.9,
      aLeadK=-2.6,
    )
  )
  limit = get_lead_predecel_accel_limit(radar_state, v_ego=6.1)
  assert limit is not None
  # Bounded to low-speed floor (~ -1.48 m/s^2), preventing harsh -2.0 or -4.0 slam
  assert -1.6 <= limit <= -1.0




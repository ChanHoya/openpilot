from __future__ import annotations

from types import SimpleNamespace as NS

from openpilot.selfdrive.carrot.drive_report import (
  DriveReportStore,
  DriveReportTracker,
  format_duration_compact_ko,
  format_duration_count_ko,
  format_duration_ko,
  normalize_gear_shifter,
)


def _cs(
  gear="drive",
  v=15.0,
  a=0.0,
  steer=False,
  gas=False,
  brake=False,
):
  return NS(
    gearShifter=gear,
    vEgo=v,
    aEgo=a,
    steeringPressed=steer,
    gasPressed=gas,
    brakePressed=brake,
  )


def _cc(enabled=True, lat=True, long=True):
  return NS(enabled=enabled, latActive=lat, longActive=long)


def _sd(enabled=True):
  return NS(enabled=enabled)


def test_format_helpers_and_gear_normalization():
  assert format_duration_ko(0) == "0초"
  assert format_duration_ko(45) == "45초"
  assert format_duration_ko(779) == "12분 59초"
  assert format_duration_ko(3912) == "1시간 05분 12초"
  assert format_duration_compact_ko(0) == "00분00초"
  assert format_duration_compact_ko(65) == "01분05초"
  assert format_duration_compact_ko(3912) == "65분12초"
  assert format_duration_count_ko(779, 5) == "12분 59초 (5회)"
  assert normalize_gear_shifter(1) == "park"
  assert normalize_gear_shifter(2) == "drive"
  assert normalize_gear_shifter("GearShifter.park") == "park"
  assert normalize_gear_shifter("reverse") == "reverse"


def test_short_trip_below_gate_does_not_emit_report():
  tracker = DriveReportTracker()
  t = 100.0
  # 20 seconds at 5 m/s = 100m (< 300m and < 60s)
  for step in range(200):
    res = tracker.update(t + step * 0.1, _cs(gear="drive", v=5.0), _cc(), _sd(), wall_t=1700000000.0 + step * 0.1)
    assert res is None
  report = tracker.update(t + 20.1, _cs(gear="park", v=0.0), _cc(False, False, False), _sd(False), wall_t=1700000020.1)
  assert report is None


def test_full_trip_aggregation_ratios_hard_brakes_and_park_trigger(tmp_path):
  tracker = DriveReportTracker(vehicle_name="GENESIS G80 FL")
  t = 0.0
  wall0 = 1795430000.0

  # Start in park for 5s -> no report
  for i in range(10):
    assert tracker.update(t + i * 0.5, _cs(gear="park", v=0.0), _cc(False, False, False), _sd(False), wall_t=wall0) is None

  t = 10.0
  # Segment 1: OP auto driving for 40s at 20 m/s (800m >= 300m gate)
  for i in range(401):
    dt_i = i * 0.1
    tracker.update(
      t + dt_i,
      _cs(gear="drive", v=20.0, a=0.1),
      _cc(True, True, True),
      _sd(True),
      controls_state=NS(curvature=0.001),
      wall_t=wall0 + 10.0 + dt_i,
    )

  t += 40.1
  # Segment 2: Manual steering override for 5s (count=1)
  for i in range(50):
    dt_i = i * 0.1
    tracker.update(t + dt_i, _cs(gear="drive", v=18.0, steer=True), _cc(True, False, True), _sd(True), wall_t=wall0 + 50.1 + dt_i)

  t += 5.0
  # Segment 3: Manual gas for 4s (count=1)
  for i in range(40):
    dt_i = i * 0.1
    tracker.update(t + dt_i, _cs(gear="drive", v=22.0, gas=True), _cc(False, False, False), _sd(True), wall_t=wall0 + 55.1 + dt_i)

  t += 4.0
  # Segment 4: Auto hard brake (<= -2.8 for 0.4s without brakePressed -> 1 auto hard brake)
  for i in range(5):
    dt_i = i * 0.1
    tracker.update(t + dt_i, _cs(gear="drive", v=15.0 - i, a=-3.1, brake=False), _cc(True, True, True), _sd(True), wall_t=wall0 + 59.1 + dt_i)

  # Recovery / normal cruise > 3s cooldown before next hard brake event
  t += 1.0
  for i in range(40):
    dt_i = i * 0.1
    tracker.update(t + dt_i, _cs(gear="drive", v=12.0, a=0.0, brake=False), _cc(True, True, True), _sd(True), wall_t=wall0 + 60.0 + dt_i)

  t += 4.0
  # Manual hard brake (brakePressed=True, -3.5 m/s^2 for 0.5s -> 1 manual hard brake + brake count=1)
  for i in range(6):
    dt_i = i * 0.1
    tracker.update(t + dt_i, _cs(gear="drive", v=12.0 - i, a=-3.5, brake=True), _cc(False, False, False), _sd(False), wall_t=wall0 + 64.0 + dt_i)

  t += 1.0
  # Come to stop and shift to park
  report = tracker.update(t, _cs(gear="park", v=0.0, a=0.0), _cc(False, False, False), _sd(False), wall_t=wall0 + 66.0)
  assert report is not None
  assert report["vehicle_name"] == "GENESIS G80 FL"
  assert report["distance_m"] >= 800.0
  assert report["ratios"]["auto_pct"] + report["ratios"]["steer_pct"] + report["ratios"]["pedal_pct"] == 100
  assert report["controls"]["auto"]["count"] >= 1
  assert report["controls"]["steer"]["count"] == 1
  assert report["controls"]["gas"]["count"] == 1
  assert report["controls"]["brake"]["count"] == 1
  assert "(" in report["controls"]["auto"]["text"] and "회)" in report["controls"]["auto"]["text"]
  assert report["hard_brakes"]["auto_count"] == 1
  assert report["hard_brakes"]["manual_count"] == 1
  assert report["hard_brakes"]["total_count"] == 2
  assert len(report["hard_brakes"]["events"]) == 2
  assert 0 <= report["score"] <= 100
  assert "kakao_status" not in report

  # Staying in park must not emit a duplicate report
  assert tracker.update(t + 1.0, _cs(gear="park", v=0.0), _cc(False, False, False), _sd(False), wall_t=wall0 + 66.0) is None

  # Save and verify store rotation/listing
  store = DriveReportStore(base_dir=tmp_path, max_reports=2)
  saved = store.save_report(report)
  assert (tmp_path / f"{saved['report_id']}.json").exists()
  loaded = store.get_report(saved["report_id"])
  assert loaded["score"] == report["score"]
  assert len(store.list_reports()) == 1


def test_idle_stop_time_is_tracked_separately_and_excluded_from_control_ratios():
  tracker = DriveReportTracker()
  t = 10.0
  wall0 = 1795430000.0

  # Move on OP auto for 20s at 20 m/s = 400m
  for i in range(201):
    dt_i = i * 0.1
    tracker.update(t + dt_i, _cs(gear="drive", v=20.0), _cc(True, True, True), _sd(True), wall_t=wall0 + dt_i)

  # Hold stationary at traffic light with foot on brake for 30s (v=0.0, brake=True, op=False)
  t += 20.1
  for i in range(300):
    dt_i = i * 0.1
    tracker.update(t + dt_i, _cs(gear="drive", v=0.0, brake=True), _cc(False, False, False), _sd(False), wall_t=wall0 + 20.1 + dt_i)

  t += 30.0
  report = tracker.update(t, _cs(gear="park", v=0.0), _cc(False, False, False), _sd(False), wall_t=wall0 + 51.0)
  assert report is not None
  assert report["duration_s"] >= 49.0
  assert 19.5 <= report["moving_s"] <= 20.5
  assert 29.0 <= report["idle_s"] <= 31.5
  # Ratios exclude stationary time -> 100% auto during moving, 0% pedal
  assert report["ratios"] == {"auto_pct": 100, "steer_pct": 0, "pedal_pct": 0}
  assert report["controls"]["brake"]["seconds"] == 0.0


def test_live_stats_tracks_four_channels_and_active_flags():
  tracker = DriveReportTracker()
  wall0 = 1795430000.0

  # Start in drive on OP auto for 5s, steer override 3s, gas 2s, brake 1s while moving
  tracker.update(10.0, _cs(gear="drive", v=10.0), _cc(True, True, True), _sd(True), wall_t=wall0)
  for i in range(1, 51):
    tracker.update(10.0 + i * 0.1, _cs(gear="drive", v=10.0), _cc(True, True, True), _sd(True), wall_t=wall0 + i * 0.1)

  live_auto = tracker.get_live_stats()
  assert live_auto["trip_started"] is True
  by_k = {it["key"]: it for it in live_auto["items"]}
  assert by_k["auto"]["active"] is True
  assert by_k["steer"]["active"] is False
  assert by_k["gas"]["active"] is False
  assert by_k["brake"]["active"] is False
  assert by_k["auto"]["time_text"] == "00분05초"
  assert by_k["auto"]["pct"] == 100.0

  for i in range(1, 21):
    tracker.update(15.0 + i * 0.1, _cs(gear="drive", v=12.0, gas=True), _cc(False, False, False), _sd(False), wall_t=wall0 + 5.0 + i * 0.1)
  live_gas = tracker.get_live_stats()
  by_k2 = {it["key"]: it for it in live_gas["items"]}
  assert by_k2["auto"]["active"] is False
  assert by_k2["gas"]["active"] is True
  assert by_k2["gas"]["time_text"] == "00분02초"
  assert abs(sum(it["pct"] for it in live_gas["items"]) - 100.0) <= 0.2

  # Stationary brake in D keeps brake active flag without adding moving brake seconds
  tracker.update(17.1, _cs(gear="drive", v=0.0, brake=True), _cc(False, False, False), _sd(False), wall_t=wall0 + 7.1)
  by_k3 = {it["key"]: it for it in tracker.get_live_stats()["items"]}
  assert by_k3["brake"]["active"] is True
  assert by_k3["brake"]["seconds"] == 0.0

  # Entering Park clears all active flags
  tracker.update(18.0, _cs(gear="park", v=0.0), _cc(False, False, False), _sd(False), wall_t=wall0 + 8.0)
  by_k4 = {it["key"]: it for it in tracker.get_live_stats()["items"]}
  assert not any(by_k4[k]["active"] for k in ("auto", "steer", "gas", "brake"))


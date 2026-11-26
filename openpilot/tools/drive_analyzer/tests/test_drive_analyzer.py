"""Unit tests for openpilot.tools.drive_analyzer (sync, scanner, regression gate)."""

from __future__ import annotations

from dataclasses import replace
import functools
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import types

import pytest

from openpilot.tools.drive_analyzer import cli as cli_mod
from openpilot.tools.drive_analyzer.cli import main as cli_main
from openpilot.tools.drive_analyzer.html_report import build_recommendations, render_html_report
from openpilot.tools.drive_analyzer.regression import (
  compare_lateral_frames,
  compare_segment_frames,
  evaluate_regression_gate,
  load_drive_helpers_from_git_ref,
  load_shaper_from_git_ref,
  render_regression_markdown,
  run_regression,
)
from openpilot.tools.drive_analyzer.scanner import (
  FrameSample,
  ScanReport,
  _match_reports_for_segments,
  analyze_frames,
  render_scan_markdown,
)
from openpilot.tools.drive_analyzer.server import create_studio_server, serve_studio
from openpilot.tools.drive_analyzer.sync import (
  _resolve_route_times,
  add_to_golden,
  discover_device_on_wifi,
  extract_dongle_id,
  list_route_groups,
  load_config,
  probe_and_save_host,
  render_route_groups_markdown,
  resolve_device_target,
  save_config,
  sync_from_device,
)


def _frame(t: float, **overrides) -> FrameSample:
  base = FrameSample(
    t=t,
    v_kph=50.0,
    a_ego=0.0,
    long_active=True,
    gas_pressed=False,
    brake_pressed=False,
    lead_status=False,
    lead_radar=False,
    lead_d=0.0,
    lead_v=0.0,
    lead_v_rel=0.0,
    lead_a=0.0,
    lead_prob=0.0,
    a_base=0.0,
    a_target=0.0,
    mpc_source="cruise",
    should_stop=False,
    fcw=False,
    v_target_now=50.0 / 3.6,
    j_target_now=0.0,
    t_follow=1.3,
  )
  return replace(base, **overrides)


def test_config_and_incremental_wifi_sync(tmp_path: Path):
  cfg_path = tmp_path / "config.json"
  save_config("192.168.0.44", port=7000, config_path=cfg_path)
  assert load_config(cfg_path)["host"] == "192.168.0.44"
  host, port, _ = resolve_device_target(auto_discover=False, config_path=cfg_path)
  assert (host, port) == ("192.168.0.44", 7000)

  download_calls: list[str] = []

  class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
      return

    def do_GET(self):
      if self.path == "/api/tools/device_info":
        body = json.dumps({"ok": True, "info": {"dongle_id": "abc123", "branch": "carrot-wip"}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)
      elif self.path.startswith("/api/drive-reports"):
        body = json.dumps({"ok": True, "reports": [{"report_id": "20261126_080000", "score": 92, "grade": "A", "ended_at": 100.0}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)
      elif self.path.startswith("/api/dashcam/routes"):
        body = json.dumps({
          "ok": True,
          "routes": [{
            "route": "0000012b--deadbeef00",
            "segmentFolders": ["0000012b--deadbeef00--0"],
            "segmentsHasMore": True,
          }],
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)
      elif self.path.startswith("/api/dashcam/segments/"):
        body = json.dumps({
          "ok": True,
          "segments": ["0000012b--deadbeef00--0", "0000012b--deadbeef00--1"],
          "hasMore": False,
          "nextOffset": None,
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)
      elif self.path == "/api/dashcam/download/0000012b--deadbeef00--0/rlog":
        download_calls.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/zstd")
        self.send_header("Content-Disposition", 'attachment; filename="0000012b--deadbeef00--0--rlog.zst"')
        self.end_headers()
        self.wfile.write(b"ZSTD_FAKE_SEG0")
      elif self.path == "/api/dashcam/download/0000012b--deadbeef00--1/rlog":
        # Force fallback to /api/dashcam/replay-source/
        download_calls.append(self.path)
        self.send_response(404)
        self.end_headers()
      elif self.path == "/api/dashcam/replay-source/0000012b--deadbeef00--1/rlog":
        download_calls.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/zstd")
        self.end_headers()
        self.wfile.write(b"ZSTD_FAKE_SEG1")
      else:
        self.send_response(404)
        self.end_headers()

  server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
  srv_port = server.server_address[1]
  t = threading.Thread(target=server.serve_forever, daemon=True)
  t.start()
  try:
    base_dir = tmp_path / "analysis"
    res1 = sync_from_device(
      host="127.0.0.1",
      port=srv_port,
      latest_routes=2,
      workers=2,
      base_dir=base_dir,
      config_path=cfg_path,
    )
    assert res1.errors == []
    assert res1.reports_saved == 1
    assert res1.segments_downloaded == 2
    assert res1.segments_skipped == 0
    assert (base_dir / "reports" / "20261126_080000.json").is_file()
    assert (base_dir / "routes" / "0000012b--deadbeef00" / "0000012b--deadbeef00--0_rlog.zst").read_bytes() == b"ZSTD_FAKE_SEG0"
    assert (base_dir / "routes" / "0000012b--deadbeef00" / "0000012b--deadbeef00--1_rlog.zst").read_bytes() == b"ZSTD_FAKE_SEG1"

    # Second sync must skip already downloaded files and identical report
    calls_before = len(download_calls)
    res2 = sync_from_device(
      host="127.0.0.1",
      port=srv_port,
      latest_routes=2,
      workers=2,
      base_dir=base_dir,
      config_path=cfg_path,
    )
    assert res2.segments_downloaded == 0
    assert res2.segments_skipped == 2
    assert res2.reports_saved == 0
    assert res2.reports_skipped == 1
    assert len(download_calls) == calls_before

    # Golden copy helper
    added, skipped = add_to_golden([base_dir / "routes" / "0000012b--deadbeef00"], golden_dir=base_dir / "golden_v1.0")
    assert len(added) == 2 and skipped == []
    added2, skipped2 = add_to_golden([base_dir / "routes" / "0000012b--deadbeef00"], golden_dir=base_dir / "golden_v1.0")
    assert added2 == [] and len(skipped2) == 2
  finally:
    server.shutdown()
    server.server_close()


def test_scanner_detects_all_five_anomaly_categories():
  frames: list[FrameSample] = []
  t = 0.0

  # 1) auto_hard_brake around t=1.0..1.5s
  for i in range(30):
    a = -2.9 if 18 <= i <= 24 else -0.2
    frames.append(_frame(t, v_kph=45.0, a_ego=a, a_target=a, a_base=a, lead_status=True, lead_radar=True, lead_d=14.0, lead_v_rel=-3.0))
    t += 0.05

  # Gap with brake pedal override -> 2) manual_brake_override
  for i in range(40):
    bp = i >= 22
    ae = -2.7 if bp else 0.1
    la = not bp
    frames.append(_frame(
      t, v_kph=52.0, long_active=la, brake_pressed=bp, a_ego=ae,
      a_target=0.25 if not bp else -0.5, a_base=0.25 if not bp else -0.5,
      lead_status=True, lead_radar=False, lead_prob=0.82, lead_d=38.0, lead_v_rel=-4.0,
    ))
    t += 0.05

  # 3) low_speed_surge (1..25 km/h alternating aTarget + / -)
  for i in range(100):
    cyc = (i // 15) % 2
    at = 0.70 if cyc == 0 else -0.70
    frames.append(_frame(t, v_kph=14.0, a_target=at, a_base=at, a_ego=at * 0.8, lead_status=True, lead_radar=True, lead_d=7.0))
    t += 0.05

  # 4) mid_speed_surge (35 km/h abrupt snap down with flat aBase)
  for i in range(40):
    at = -0.50 if 10 <= i <= 20 else 0.20
    frames.append(_frame(t, v_kph=40.0, a_target=at, a_base=0.10, a_ego=0.0, lead_status=True, lead_radar=True, lead_d=15.0))
    t += 0.05

  # 5) far_lead_late_decel (>= 50 km/h, vision lead at 70m vr=-4.0 while accelerating)
  for i in range(30):
    frames.append(_frame(
      t, v_kph=75.0, a_target=0.55, a_base=0.55, a_ego=0.35,
      lead_status=True, lead_radar=False, lead_prob=0.80, lead_d=70.0, lead_v_rel=-4.0,
    ))
    t += 0.05

  summary = analyze_frames("0000012a--abc--11", frames, segment_path="/tmp/seg11.zst")
  cats = {e.category for e in summary.anomalies}
  assert {"auto_hard_brake", "manual_brake_override", "low_speed_surge", "mid_speed_surge", "far_lead_late_decel"} <= cats

  md = render_scan_markdown(ScanReport(segments=[summary], anomalies=summary.anomalies))
  assert "0000012a--abc--11" in md
  assert "far_lead_late_decel" in md


def test_regression_gate_passes_ver1_and_blocks_forbidden_band():
  base_mod = load_shaper_from_git_ref("ver-1.0")
  import openpilot.selfdrive.controls.lib.human_longitudinal as curr_mod

  frames = [
    _frame(
      i * 0.05,
      v_kph=15.0 if i < 40 else 75.0,
      a_base=0.30,
      a_target=0.30,
      lead_status=True,
      lead_radar=(i < 40),
      lead_d=12.0 if i < 40 else 75.0,
      lead_v=(15.0 / 3.6) if i < 40 else (60.0 / 3.6),
      lead_v_rel=0.0 if i < 40 else -4.0,
      lead_prob=0.85,
    )
    for i in range(80)
  ]

  # HEAD must match ver-1.0 identically (0 changed frames)
  comp_same = compare_segment_frames(base_mod, curr_mod, [("syn--0", {"stop_dist": 5.0}, frames)], dataset_label="golden")
  assert comp_same.changed_frames == 0
  assert comp_same.ramp_changed_frames == 0
  passed, reasons = evaluate_regression_gate(comp_same, forbid_bands=("0-25",))
  assert passed and reasons == []

  # Synthetic candidate that perturbs 0-25 km/h output must be blocked by gate
  fake_mod = types.SimpleNamespace(
    compute_accel_ramp_targets=curr_mod.compute_accel_ramp_targets,
    HumanLongitudinalShaper=None,
  )

  class PerturbedShaper(curr_mod.HumanLongitudinalShaper):
    @functools.wraps(curr_mod.HumanLongitudinalShaper.shape)
    def shape(self, **kwargs):
      res = super().shape(**kwargs)
      if kwargs.get("v_ego", 0.0) * 3.6 < 25.0 and not kwargs.get("reset_state", False):
        return replace(res, a_target=res.a_target + 0.20)
      return res

  fake_mod.HumanLongitudinalShaper = PerturbedShaper
  comp_bad = compare_segment_frames(base_mod, fake_mod, [("syn--0", {"stop_dist": 5.0}, frames)], dataset_label="golden")
  assert comp_bad.bands["0-25"].changed_frames > 0
  assert comp_bad.bands["55-85"].changed_frames == 0
  passed_bad, reasons_bad = evaluate_regression_gate(comp_bad, forbid_bands=("0-25",))
  assert not passed_bad
  assert any("0-25" in r for r in reasons_bad)

  rep = run_regression(baseline_ref="ver-1.0")
  assert rep.passed
  assert "ver-1.0" in render_regression_markdown(rep)


def test_dongle_id_discovery_and_dhcp_failover(tmp_path: Path, monkeypatch):
  assert extract_dongle_id({"identity": {"dongle_id": " 0123456789ABCDEF "}}) == "0123456789abcdef"
  assert extract_dongle_id({"dongle_id": "abc123"}) == "abc123"
  assert extract_dongle_id({}) == ""

  cfg_path = tmp_path / "config.json"
  save_config(dongle_id="0123456789abcdef", config_path=cfg_path)
  save_config(host="192.0.2.10", port=7000, config_path=cfg_path)
  cfg = load_config(cfg_path)
  assert cfg["host"] == "192.0.2.10"
  assert cfg["dongle_id"] == "0123456789abcdef"

  # Clearing dongle_id removes pinned ID cleanly
  save_config(clear_dongle_id=True, config_path=cfg_path)
  assert "dongle_id" not in load_config(cfg_path)

  class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
      return

    def do_GET(self):
      if self.path == "/api/tools/device_info":
        body = json.dumps({
          "ok": True,
          "info": {
            "identity": {"dongle_id": "0123456789abcdef"},
            "software": {"branch": "carrot-wip", "commit": "ver-1.0"},
          },
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)
      else:
        self.send_response(404)
        self.end_headers()

  server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
  srv_port = server.server_address[1]
  t = threading.Thread(target=server.serve_forever, daemon=True)
  t.start()
  try:
    assert discover_device_on_wifi(
      dongle_id="wrong_id",
      port=srv_port,
      candidate_ips=["127.0.0.1"],
      probe_timeout_s=0.5,
    ) is None

    found = discover_device_on_wifi(
      dongle_id="0123456789ABCDEF",
      port=srv_port,
      candidate_ips=["127.0.0.1"],
      probe_timeout_s=0.5,
    )
    assert found is not None and found[0] == "127.0.0.1"

    # probe_and_save_host automatically detects & stores dongle_id from reachable IP
    res_probe = probe_and_save_host("127.0.0.1", port=srv_port, config_path=cfg_path)
    assert res_probe["probed"] is True
    assert res_probe["config"]["dongle_id"] == "0123456789abcdef"

    # Simulate stale saved host (DHCP changed) -> falls back to discover_device_on_wifi & saves new host
    save_config(host="192.0.2.1", port=srv_port, dongle_id="0123456789abcdef", config_path=cfg_path)
    monkeypatch.setattr("openpilot.tools.drive_analyzer.sync.DEFAULT_CANDIDATE_HOSTS", ())
    import openpilot.tools.drive_analyzer.sync as sync_mod
    monkeypatch.setattr(
      sync_mod,
      "discover_device_on_wifi",
      lambda *, dongle_id=None, port=srv_port, **kw: ("127.0.0.1", {"identity": {"dongle_id": dongle_id or "0123456789abcdef"}}),
    )
    host, port, _ = resolve_device_target(port=srv_port, timeout_s=0.5, config_path=cfg_path)
    assert (host, port) == ("127.0.0.1", srv_port)
    assert load_config(cfg_path)["host"] == "127.0.0.1"
    assert load_config(cfg_path)["dongle_id"] == "0123456789abcdef"
  finally:
    server.shutdown()
    server.server_close()


def test_scanner_detects_all_four_lateral_anomalies_and_ignores_lane_change():
  frames: list[FrameSample] = []
  t = 0.0

  # 1) manual_steer_override in corner after lat_active
  for i in range(30):
    sp = i >= 20
    frames.append(_frame(
      t, v_kph=55.0, lat_active=not sp, steering_pressed=sp,
      steer_torque=3.5 if sp else 0.5,
      steer_angle_deg=25.0, steer_angle_desired_deg=30.0,
      lat_accel=2.2 if not sp else 1.5, desired_lat_accel=2.4,
      steer_saturated=(not sp), lane_offset_m=-0.48,
    ))
    t += 0.05

  # Clean blinker / lane-change segment must NOT emit any lateral anomalies
  for i in range(50):
    frames.append(_frame(
      t, v_kph=60.0, lat_active=True, blinker_active=True, lane_changing=True,
      steer_rate_deg=20.0 if (i % 6 < 3) else -20.0,
      steer_angle_deg=8.0 if (i % 6 < 3) else -8.0,
      lat_accel=2.9, desired_lat_accel=2.9, lane_offset_m=0.60,
    ))
    t += 0.05

  # 2) steer_pingpong in cornering ("와리가리")
  for i in range(80):
    cyc = (i // 10) % 2
    rate = 18.0 if cyc == 0 else -18.0
    ang = 32.0 if cyc == 0 else 22.0
    frames.append(_frame(
      t, v_kph=50.0, lat_active=True, steer_rate_deg=rate,
      steer_angle_deg=ang, steer_angle_desired_deg=27.0,
      lat_accel=1.4, desired_lat_accel=1.5,
    ))
    t += 0.05

  # 3) curve_lat_overshoot (high lateral accel + sustained torque saturation)
  for i in range(30):
    frames.append(_frame(
      t, v_kph=65.0, lat_active=True, steer_saturated=True,
      lat_accel=2.85, desired_lat_accel=2.95, lane_offset_m=-0.50,
    ))
    t += 0.05

  # 4) lane_center_bias on straight road (>= 2.5s)
  for i in range(60):
    frames.append(_frame(
      t, v_kph=80.0, lat_active=True, curvature=0.0005, lat_accel=0.25,
      lane_offset_m=0.42,
    ))
    t += 0.05

  summary = analyze_frames("0000012b--lat--0", frames, segment_path="/tmp/lat0.zst")
  cats = {e.category for e in summary.anomalies}
  assert {"manual_steer_override", "steer_pingpong", "curve_lat_overshoot", "lane_center_bias"} <= cats
  assert summary.curve_steer_rate_rms > 0.0
  # Verify lane-change window (t=1.5..4.0s) produced zero lateral events
  lc_events = [e for e in summary.anomalies if 1.6 <= e.t_start <= 3.9]
  assert lc_events == []


def test_lateral_regression_gate_passes_ver1_and_blocks_straight_band():
  base_lat = load_drive_helpers_from_git_ref("ver-1.0")
  import openpilot.selfdrive.controls.lib.drive_helpers as curr_lat

  model_t = tuple(i * 0.2 for i in range(33))
  straight_yaws = tuple(0.0 for _ in range(33))
  straight_rates = tuple(0.0 for _ in range(33))
  curve_yaws = tuple(0.005 * (i * 0.2) for i in range(33))
  curve_rates = tuple(0.005 * 20.0 for _ in range(33))

  frames = [
    _frame(
      i * 0.05,
      v_kph=72.0,
      lat_active=True,
      desired_curvature=0.0005 if i < 20 else 0.005,
      curvature=0.0005 if i < 20 else 0.005,
      lat_accel=0.20 if i < 20 else 2.0,
      desired_lat_accel=0.20 if i < 20 else 2.0,
      model_t=model_t,
      model_yaws=straight_yaws if i < 20 else curve_yaws,
      model_yaw_rates=straight_rates if i < 20 else curve_rates,
      lane_y_left=1.7,
      lane_y_right=-1.7,
      lane_prob_left=0.9,
      lane_prob_right=0.9,
    )
    for i in range(40)
  ]

  comp_same = compare_lateral_frames(base_lat, curr_lat, [("syn--0", {"curve_inner_bias": 40.0}, frames)], dataset_label="golden")
  assert comp_same.changed_frames == 0
  assert comp_same.ramp_changed_frames == 0
  ok, reasons = evaluate_regression_gate(comp_same, forbid_bands=("straight",))
  assert ok and reasons == []

  # Candidate that alters straight curvature must be rejected by forbid_lat_bands=("straight",)
  fake_lat = types.SimpleNamespace(
    compute_curve_inner_curvature_assist=lambda base_curvature, *a, **kw: float(base_curvature) + 0.001,
    compute_curve_entry_smooth_factor=curr_lat.compute_curve_entry_smooth_factor,
  )
  comp_bad = compare_lateral_frames(base_lat, fake_lat, [("syn--0", {"curve_inner_bias": 40.0}, frames)], dataset_label="golden")
  assert comp_bad.bands["straight"].changed_frames > 0
  ok_bad, reasons_bad = evaluate_regression_gate(comp_bad, forbid_bands=("straight",))
  assert not ok_bad
  assert any("straight" in r for r in reasons_bad)

  rep = run_regression(baseline_ref="ver-1.0", mode="both")
  assert rep.passed


def test_html_report_generation_and_recommendations(tmp_path: Path):
  # Clean route: zero anomalies -> keep ver-1.0 code recommendation + telemetry extracted at ~2Hz
  clean_frames = [
    _frame(
      i * 0.05,
      v_kph=65.0 + (i % 5) * 0.2,
      a_ego=0.1,
      a_base=0.1,
      a_target=0.1,
      lat_active=True,
      lat_accel=0.35,
      desired_lat_accel=0.34,
      steer_angle_deg=4.2,
      steer_angle_desired_deg=4.1,
      lane_offset_m=0.02,
    )
    for i in range(60)
  ]
  clean_seg = analyze_frames(
    "0000012c--clean--0",
    clean_frames,
    segment_path="/tmp/clean0.zst",
    metadata={
      "commit": "738f018ef",
      "branch": "carrot-wip",
      "started_at": "2026-11-26 06:55:50",
      "ended_at": "06:56:46",
      "started_at_ts": 1795643750.0,
      "params": {"AutoCurveSpeedFactor": "100", "MapTurnSpeedFactor": "90"},
    },
  )
  assert clean_seg.telemetry
  assert len(clean_seg.telemetry["t"]) >= 6
  assert len(clean_seg.telemetry["t"]) < len(clean_frames)
  assert clean_seg.telemetry["fl"][0] == (1 | 2)

  clean_report = ScanReport(segments=[clean_seg], anomalies=[], drive_reports=[{
    "report_id": "20261126_000000",
    "score": 96,
    "grade": "A",
    "distance_km": 8.4,
    "duration_s": 420.0,
    "hard_brake_count": 0,
  }])
  clean_guides, clean_plans = build_recommendations(clean_report.to_dict())
  assert any("AutoCurveSpeedFactor" in g["param"] for g in clean_guides)
  assert any("수정 불필요" in p["status"] for p in clean_plans)

  # Anomaly route: produces [승인 대기] code plans & preserves </script> escaping + zero external CDN
  bad_frames = list(clean_frames)
  t = 3.0
  for i in range(30):
    bad_frames.append(_frame(
      t, v_kph=68.0, a_ego=-3.2, a_base=-3.0, a_target=-3.1,
      lat_active=True, steer_saturated=True, lat_accel=2.9, desired_lat_accel=3.0,
      lane_offset_m=-0.48, lead_status=True, lead_d=24.0,
    ))
    t += 0.05
  bad_seg = analyze_frames("0000012c--anom--1", bad_frames, segment_path="/tmp/anom1.zst")
  bad_report = ScanReport(segments=[clean_seg, bad_seg], anomalies=list(bad_seg.anomalies), drive_reports=clean_report.drive_reports)
  _, bad_plans = build_recommendations(bad_report.to_dict())
  assert any(p["status"] == "승인 대기" and "ver-1.0" in p["verify_cmd"] for p in bad_plans)

  html_out = render_html_report(
    bad_report,
    custom_notes={"headline": "테스트 진단 </script><script>alert(1)</script>"},
  )
  assert "<!DOCTYPE html>" in html_out
  assert "2026-11-26 06:55:50" in html_out
  assert "2026-11-26 06:55:50~06:56:46" in render_scan_markdown(bad_report)
  assert "AutoCurveSpeedFactor" in html_out
  assert "forbid-band 0-25" in html_out
  assert "https://" not in html_out and "http://" not in html_out
  assert "<\\/script>" in html_out
  assert "alert(1)</script>" not in html_out
  assert 'data-height="' in html_out
  assert 'getAttribute("height")' not in html_out

  summary_json = tmp_path / "scan_summary.json"
  out_html = tmp_path / "custom_report.html"
  summary_json.write_text(json.dumps(bad_report.to_dict(), ensure_ascii=False), encoding="utf-8")
  rc = cli_main(["report", "--summary", str(summary_json), "--out", str(out_html)])
  assert rc == 0
  assert out_html.is_file() and "CarrotPilot" in out_html.read_text(encoding="utf-8")


def test_route_group_listing_and_selective_analyze(tmp_path: Path, monkeypatch, capsys):
  # 1) _resolve_route_times guards against pre-NTP boot clock jump spanning > 7 hours
  s_ep, e_ep, dur_m = _resolve_route_times({
    "route": "0000012e--c046ba227a",
    "routeStartEpoch": 1795659371,
    "routeEndEpoch": 1795685020,
    "segmentCount": 4,
    "segmentFolders": [
      "0000012e--c046ba227a--0",
      "0000012e--c046ba227a--1",
      "0000012e--c046ba227a--2",
      "0000012e--c046ba227a--3",
    ],
    "segmentTimes": {
      "0000012e--c046ba227a--0": {"startEpoch": 1795659371, "endEpoch": 1795659431},
      "0000012e--c046ba227a--1": {"startEpoch": 1795659431, "endEpoch": 1795684900},
      "0000012e--c046ba227a--2": {"startEpoch": 1795684900, "endEpoch": 1795684960},
      "0000012e--c046ba227a--3": {"startEpoch": 1795684960, "endEpoch": 1795685020},
    },
  })
  assert e_ep == 1795685020
  assert s_ep == 1795684900
  assert dur_m == 2.0

  # 2) Mock Carrot Web server serving 2 routes & matching drive report for route 1 only
  reports_payload = [
    {"report_id": "20261126_new", "score": 88, "grade": "B", "distance_km": 11.2, "started_at": 1795684800, "ended_at": 1795685030, "hard_brake_count": 0},
    {"report_id": "20261125_old", "score": 71, "grade": "C", "distance_km": 4.5, "started_at": 1795600000, "ended_at": 1795600300, "hard_brake_count": 1},
  ]
  routes_payload = [
    {
      "route": "0000012e--c046ba227a",
      "segmentCount": 1,
      "segmentFolders": ["0000012e--c046ba227a--0"],
      "routeStartEpoch": 1795684900,
      "routeEndEpoch": 1795685020,
      "latestModifiedLabel": "5분 전",
    },
    {
      "route": "0000012d--5710a28145",
      "segmentCount": 1,
      "segmentFolders": ["0000012d--5710a28145--0"],
      "routeStartEpoch": 1795600000,
      "routeEndEpoch": 1795600300,
      "latestModifiedLabel": "1일 전",
    },
  ]
  downloaded: list[str] = []

  class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
      return

    def do_GET(self):
      if self.path == "/api/tools/device_info":
        body = json.dumps({"ok": True, "info": {"identity": {"dongle_id": "0123456789abcdef"}}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return
      if self.path.startswith("/api/drive-reports"):
        body = json.dumps({"ok": True, "reports": reports_payload}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return
      if self.path.startswith("/api/dashcam/routes"):
        body = json.dumps({"ok": True, "routes": routes_payload}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return
      if self.path.endswith("/rlog"):
        downloaded.append(self.path)
        body = b"fake-zst"
        self.send_response(200)
        self.send_header("Content-Type", "application/zstd")
        self.send_header("Content-Disposition", 'attachment; filename="rlog.zst"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return
      self.send_error(404)

  srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
  th = threading.Thread(target=srv.serve_forever, daemon=True)
  th.start()
  try:
    port = srv.server_address[1]
    base_dir = tmp_path / "da"
    cfg_path = base_dir / "config.json"
    save_config(host="127.0.0.1", port=port, dongle_id="0123456789abcdef", config_path=cfg_path)

    groups, dev_label, dev_err = list_route_groups(base_dir=base_dir, config_path=cfg_path, limit=10)
    assert dev_err == "" and "127.0.0.1" in dev_label
    assert [g.route for g in groups] == ["0000012e--c046ba227a", "0000012d--5710a28145"]
    assert groups[0].matched_report["report_id"] == "20261126_new"
    assert groups[1].matched_report["report_id"] == "20261125_old"
    md = render_route_groups_markdown(groups, device_label=dev_label)
    assert "0000012e--c046ba227a" in md and "88점(B)" in md

    # _match_reports_for_segments filters to segments' timestamp window
    old_seg = analyze_frames("0000012d--5710a28145--0", [_frame(0.0)], metadata={"started_at_ts": 1795600000.0})
    old_seg = replace(old_seg, started_at_ts=1795600000.0, duration_s=120.0)
    matched_only_old = _match_reports_for_segments([old_seg], reports_payload)
    assert [r["report_id"] for r in matched_only_old] == ["20261125_old"]

    # CLI routes + analyze --route 1 (only syncs group #1 and generates route_0000012e--c046ba227a/analysis_report.html)
    monkeypatch.setattr(cli_mod, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(cli_mod, "ROUTES_DIR", base_dir / "routes")
    monkeypatch.setattr(cli_mod, "REPORTS_DIR", base_dir / "reports")
    monkeypatch.setattr(cli_mod, "SCANS_DIR", base_dir / "scans")

    assert cli_main(["routes", "--limit", "5"]) == 0
    out_routes = capsys.readouterr().out
    assert "0000012e--c046ba227a" in out_routes

    assert cli_main(["analyze", "--route", "1", "--workers", "1"]) == 0
    out_ana = capsys.readouterr().out
    assert "0000012e--c046ba227a" in out_ana
    assert len(downloaded) == 1 and "0000012e--c046ba227a--0" in downloaded[0]
    assert (base_dir / "scans" / "route_0000012e--c046ba227a" / "analysis_report.html").is_file()
  finally:
    srv.shutdown()
    srv.server_close()


def test_html_studio_server_routes_and_on_demand_analyze(tmp_path: Path, monkeypatch):
  from urllib.request import Request, urlopen

  routes_payload = [
    {
      "route": "0000012e--c046ba227a",
      "segmentCount": 1,
      "segmentFolders": ["0000012e--c046ba227a--0"],
      "routeStartEpoch": 1795684900,
      "routeEndEpoch": 1795685020,
      "latestModifiedLabel": "3분 전",
    },
    {
      "route": "0000012d--5710a28145",
      "segmentCount": 1,
      "segmentFolders": ["0000012d--5710a28145--0"],
      "routeStartEpoch": 1795682000,
      "routeEndEpoch": 1795682600,
      "latestModifiedLabel": "20분 전",
    },
  ]
  reports_payload = [
    {"report_id": "20261126_182413", "score": 78, "grade": "B", "distance_km": 5.2, "started_at": 1795684800, "ended_at": 1795685030, "hard_brakes": {"auto_count": 2, "manual_count": 0}},
  ]

  class MockDeviceHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
      return

    def do_GET(self):
      if self.path == "/api/tools/device_info":
        body = json.dumps({"ok": True, "info": {"identity": {"dongle_id": "0123456789abcdef"}}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return
      if self.path.startswith("/api/drive-reports"):
        body = json.dumps({"ok": True, "reports": reports_payload}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return
      if self.path.startswith("/api/dashcam/routes"):
        body = json.dumps({"ok": True, "routes": routes_payload}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return
      if self.path.endswith("/rlog"):
        body = b"fake-zst"
        self.send_response(200)
        self.send_header("Content-Type", "application/zstd")
        self.send_header("Content-Disposition", 'attachment; filename="rlog.zst"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return
      self.send_error(404)

  dev_srv = ThreadingHTTPServer(("127.0.0.1", 0), MockDeviceHandler)
  threading.Thread(target=dev_srv.serve_forever, daemon=True).start()

  studio_srv = None
  try:
    base_dir = tmp_path / "studio_da"
    cfg_path = base_dir / "config.json"
    save_config(host="127.0.0.1", port=dev_srv.server_address[1], dongle_id="0123456789abcdef", config_path=cfg_path)

    studio_srv = create_studio_server(host="127.0.0.1", port=0, base_dir=base_dir, config_path=cfg_path)
    threading.Thread(target=studio_srv.serve_forever, daemon=True).start()
    studio_url = f"http://127.0.0.1:{studio_srv.server_address[1]}"

    with urlopen(f"{studio_url}/") as resp:
      landing_html = resp.read().decode("utf-8")
    assert 'id="sec-routes"' in landing_html
    assert 'data-height="165"' in landing_html
    assert 'getAttribute("height")' not in landing_html
    assert "0000012e--c046ba227a" in landing_html
    assert "https://" not in landing_html and "http://" not in landing_html

    with urlopen(f"{studio_url}/api/routes") as resp:
      routes_json = json.loads(resp.read().decode("utf-8"))
    assert len(routes_json["groups"]) == 2

    with urlopen(f"{studio_url}/api/config") as resp:
      init_cfg = json.loads(resp.read().decode("utf-8"))
    assert init_cfg["dongle_id"] == "0123456789abcdef"

    import openpilot.tools.drive_analyzer.server as srv_mod
    monkeypatch.setattr(
      srv_mod,
      "discover_device_on_wifi",
      lambda *, dongle_id=None, port=dev_srv.server_address[1], **kw: (
        "127.0.0.1",
        {"identity": {"dongle_id": dongle_id or "0123456789abcdef"}},
      ),
    )

    # Manual dongle_id config -> clear -> auto discover from Wi-Fi mock
    req_cfg = Request(
      f"{studio_url}/api/config",
      data=json.dumps({"dongle_id": "manualdeadbeef12"}).encode("utf-8"),
      headers={"Content-Type": "application/json"},
      method="POST",
    )
    with urlopen(req_cfg) as resp:
      cfg_resp = json.loads(resp.read().decode("utf-8"))
    assert cfg_resp["config"]["dongle_id"] == "manualdeadbeef12"

    req_clear = Request(
      f"{studio_url}/api/config",
      data=json.dumps({"clear_dongle_id": True}).encode("utf-8"),
      headers={"Content-Type": "application/json"},
      method="POST",
    )
    with urlopen(req_clear) as resp:
      clear_resp = json.loads(resp.read().decode("utf-8"))
    assert "dongle_id" not in clear_resp["config"]

    req_disc = Request(
      f"{studio_url}/api/discover",
      data=json.dumps({"any_device": True, "port": dev_srv.server_address[1]}).encode("utf-8"),
      headers={"Content-Type": "application/json"},
      method="POST",
    )
    with urlopen(req_disc) as resp:
      disc_resp = json.loads(resp.read().decode("utf-8"))
    assert disc_resp["ok"] and disc_resp["dongle_id"] == "0123456789abcdef"

    req_host = Request(
      f"{studio_url}/api/config",
      data=json.dumps({"host": "127.0.0.1", "port": dev_srv.server_address[1]}).encode("utf-8"),
      headers={"Content-Type": "application/json"},
      method="POST",
    )
    with urlopen(req_host) as resp:
      host_resp = json.loads(resp.read().decode("utf-8"))
    assert host_resp["config"]["dongle_id"] == "0123456789abcdef"

    # Calling serve_studio on a port already running our studio must reuse it cleanly without OSError
    serve_studio(
      host="127.0.0.1",
      port=studio_srv.server_address[1],
      open_browser=False,
      base_dir=base_dir,
      config_path=cfg_path,
    )

    req = Request(
      f"{studio_url}/api/analyze",
      data=json.dumps({"route": "0000012e--c046ba227a"}).encode("utf-8"),
      headers={"Content-Type": "application/json"},
      method="POST",
    )
    with urlopen(req) as resp:
      job_id = json.loads(resp.read().decode("utf-8"))["job_id"]

    job_status = ""
    for _ in range(40):
      with urlopen(f"{studio_url}/api/jobs/{job_id}") as resp:
        job_status = json.loads(resp.read().decode("utf-8"))["job"]["status"]
      if job_status in ("done", "error"):
        break
      time.sleep(0.05)
    assert job_status == "done"

    with urlopen(f"{studio_url}/api/report/0000012e--c046ba227a") as resp:
      rep_payload = json.loads(resp.read().decode("utf-8"))
    assert rep_payload["active_route"] == "0000012e--c046ba227a"
    assert len(rep_payload["segments"]) == 1
    assert "param_guides" in rep_payload and "code_plans" in rep_payload
  finally:
    if studio_srv is not None:
      studio_srv.shutdown()
      studio_srv.server_close()
    dev_srv.shutdown()
    dev_srv.server_close()

from __future__ import annotations

import asyncio

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from openpilot.selfdrive.carrot.drive_report import DriveReportStore
from openpilot.selfdrive.carrot.server.features import drive_reports


def _make_test_app(tmp_path) -> web.Application:
  app = web.Application()
  app[drive_reports.DRIVE_REPORT_STORE_KEY] = DriveReportStore(base_dir=tmp_path)
  drive_reports.register(app)
  return app


def test_drive_reports_endpoints_without_kakao(tmp_path):
  async def _run():
    app = _make_test_app(tmp_path)
    store: DriveReportStore = app[drive_reports.DRIVE_REPORT_STORE_KEY]
    saved = store.save_report({
      "report_id": "20261123_185012",
      "time_range_label": "2026-11-23 18:12 ~ 18:50",
      "distance_km": 12.4,
      "duration_s": 900,
      "duration_text": "15분 00초",
      "idle_text": "2분 10초",
      "score": 88,
      "grade": "A",
      "ratios": {"auto_pct": 80, "steer_pct": 8, "pedal_pct": 12},
      "controls": {
        "auto": {"seconds": 720, "count": 4, "text": "12분 00초 (4회)"},
        "steer": {"seconds": 72, "count": 3, "text": "1분 12초 (3회)"},
        "gas": {"seconds": 60, "count": 5, "text": "1분 00초 (5회)"},
        "brake": {"seconds": 48, "count": 4, "text": "48초 (4회)"},
      },
      "speed": {"avg_kph": 41.2, "max_kph": 79.0},
      "hard_brakes": {"auto_count": 0, "manual_count": 1, "total_count": 1, "events": []},
    })

    async with TestClient(TestServer(app)) as client:
      resp = await client.get("/drive-reports")
      assert resp.status == 200
      body = await resp.text()
      assert "주행 품질 리포트" in body
      assert "카카오" not in body
      for icon in ("drive_report_auto.png", "drive_report_steer.png", "drive_report_gas.png", "drive_report_brake.png"):
        assert f"/shared-assets/icons/{icon}" in body

      resp = await client.get("/api/drive-reports")
      assert resp.status == 200
      payload = await resp.json()
      assert payload["ok"] is True
      assert "kakao" not in payload
      assert len(payload["reports"]) == 1
      assert payload["reports"][0]["controls"]["auto"]["text"] == "12분 00초 (4회)"

      resp = await client.get(f"/api/drive-reports/{saved['report_id']}")
      assert resp.status == 200

      resp = await client.get("/api/kakao/config")
      assert resp.status == 404

      resp = await client.delete(f"/api/drive-reports/{saved['report_id']}")
      assert resp.status == 200
      resp = await client.get(f"/api/drive-reports/{saved['report_id']}")
      assert resp.status == 404

  asyncio.run(_run())

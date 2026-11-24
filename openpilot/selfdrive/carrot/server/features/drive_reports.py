"""Carrot Web API endpoints for saved driving quality reports."""

from __future__ import annotations

import asyncio
import os

from aiohttp import web

from ...drive_report import DriveReportStore
from ..config import WEB_DIR

DRIVE_REPORT_STORE_KEY = web.AppKey("drive_report_store", DriveReportStore)


def _store_for(request_or_app) -> DriveReportStore:
  app = request_or_app.app if isinstance(request_or_app, web.Request) else request_or_app
  store = app.get(DRIVE_REPORT_STORE_KEY)
  if isinstance(store, DriveReportStore):
    return store
  store = DriveReportStore()
  app[DRIVE_REPORT_STORE_KEY] = store
  return store


async def page_drive_reports(request: web.Request) -> web.Response:
  page_path = os.path.join(WEB_DIR, "drive_reports.html")
  if not os.path.isfile(page_path):
    return web.Response(status=404, text="drive_reports.html not found")
  return web.FileResponse(page_path)


async def api_list_drive_reports(request: web.Request) -> web.Response:
  store = _store_for(request)
  try:
    limit = max(1, min(100, int(request.query.get("limit", "50"))))
  except Exception:
    limit = 50
  reports = await asyncio.to_thread(store.list_reports, limit=limit, include_full=True)
  return web.json_response({"ok": True, "reports": reports})


async def api_get_drive_report(request: web.Request) -> web.Response:
  store = _store_for(request)
  report_id = request.match_info.get("report_id", "")
  report = await asyncio.to_thread(store.get_report, report_id)
  if report is None:
    return web.json_response({"ok": False, "error": "report not found"}, status=404)
  return web.json_response({"ok": True, "report": report})


async def api_delete_drive_report(request: web.Request) -> web.Response:
  store = _store_for(request)
  report_id = request.match_info.get("report_id", "")
  deleted = await asyncio.to_thread(store.delete_report, report_id)
  if not deleted:
    return web.json_response({"ok": False, "error": "report not found"}, status=404)
  return web.json_response({"ok": True})


def register(app: web.Application) -> None:
  app.router.add_get("/drive-reports", page_drive_reports)
  app.router.add_get("/api/drive-reports", api_list_drive_reports)
  app.router.add_get("/api/drive-reports/{report_id}", api_get_drive_report)
  app.router.add_delete("/api/drive-reports/{report_id}", api_delete_drive_report)

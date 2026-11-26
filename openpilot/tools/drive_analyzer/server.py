"""Local HTTP studio server for interactive route group selection and on-demand drive analysis."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import errno
import json
from pathlib import Path
import threading
import traceback
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import urlopen
import uuid


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
  allow_reuse_address = True


def _is_studio_alive(host: str, port: int) -> bool:
  try:
    with urlopen(f"http://{host}:{int(port)}/api/config", timeout=1.0) as resp:
      data = json.loads(resp.read().decode("utf-8"))
      return bool(isinstance(data, dict) and data.get("ok"))
  except Exception:
    return False

from . import CONFIG_PATH, DEFAULT_PORT, ensure_analysis_dirs
from .html_report import build_report_payload, render_html_report
from .scanner import scan_paths
from .sync import (
  _SAFE_NAME_RE,
  discover_device_on_wifi,
  extract_dongle_id,
  list_route_groups,
  load_config,
  probe_and_save_host,
  save_config,
  sync_from_device,
)


def _route_scan_dir(scans_dir: Path, route_name: str) -> Path:
  safe_route = _SAFE_NAME_RE.sub("_", route_name)
  return scans_dir / f"route_{safe_route}"


def _pick_initial_route(groups: list[dict[str, Any]], scans_dir: Path) -> tuple[str, dict[str, Any] | None]:
  """Pick newest already-analyzed route summary if present."""
  for g in groups:
    rname = str(g.get("route") or "")
    if not rname:
      continue
    summary_path = _route_scan_dir(scans_dir, rname) / "scan_summary.json"
    if summary_path.is_file():
      try:
        return rname, json.loads(summary_path.read_text(encoding="utf-8"))
      except Exception:
        continue
  return "", None


def create_studio_server(
  *,
  host: str = "127.0.0.1",
  port: int = 8765,
  default_limit: int = 20,
  local_only: bool = False,
  base_dir: Path | None = None,
  config_path: Path = CONFIG_PATH,
) -> ThreadingHTTPServer:
  """Create a zero-dependency local HTTP studio for browsing & analyzing drive route groups."""
  dirs = ensure_analysis_dirs(base_dir)
  routes_dir = dirs["routes"]
  reports_dir = dirs["reports"]
  scans_dir = dirs["scans"]

  jobs_lock = threading.Lock()
  jobs: dict[str, dict[str, Any]] = {}

  def _run_job(job_id: str, route_name: str, force_rescan: bool, req_local_only: bool) -> None:
    def _update(**kw: Any) -> None:
      with jobs_lock:
        if job_id in jobs:
          jobs[job_id].update(kw)

    try:
      safe_route = _SAFE_NAME_RE.sub("_", route_name)
      route_dir = routes_dir / safe_route
      out_dir = _route_scan_dir(scans_dir, route_name)
      summary_path = out_dir / "scan_summary.json"

      if summary_path.is_file() and not force_rescan:
        _update(
          status="done",
          progress=100,
          message=f"{route_name} 기존 분석 리포트를 불러왔습니다.",
          report_url=f"/api/report/{quote(route_name, safe='')}",
          html_path=str(out_dir / "analysis_report.html"),
        )
        return

      groups, _dev, _err = list_route_groups(
        limit=max(30, int(default_limit)),
        local_only=req_local_only,
        base_dir=base_dir,
        config_path=config_path,
      )
      matched = next((g for g in groups if g.route == route_name), None)
      need_sync = (not req_local_only) and (
        matched is None
        or matched.local_segments == 0
        or (matched.device_segments > 0 and matched.local_segments < matched.device_segments)
      )

      if need_sync:
        _update(status="syncing", progress=25, message=f"{route_name} 단말 로그 다운로드 중...")
        sync_res = sync_from_device(
          latest_routes=max(50, int(default_limit)),
          route_filter=route_name,
          workers=4,
          base_dir=base_dir,
          config_path=config_path,
        )
        if sync_res.errors and not route_dir.exists():
          raise RuntimeError("; ".join(sync_res.errors[:3]))

      if not route_dir.exists():
        raise RuntimeError(f"로컬 주행로그 경로를 찾을 수 없습니다: {route_dir}")

      _update(status="scanning", progress=65, message=f"{route_name} 종·횡방향 이상치 스캔 및 차트 생성 중...")
      refreshed_groups, dev_label, dev_err = list_route_groups(
        limit=max(20, int(default_limit)),
        local_only=req_local_only,
        base_dir=base_dir,
        config_path=config_path,
      )
      scan_paths(
        [route_dir],
        reports_dir=reports_dir,
        out_dir=out_dir,
        workers=4,
        route_groups=[g.to_dict() for g in refreshed_groups],
        active_route=route_name,
        device_label=dev_label,
        device_error=dev_err,
      )
      _update(
        status="done",
        progress=100,
        message=f"{route_name} 분석 완료",
        report_url=f"/api/report/{quote(route_name, safe='')}",
        html_path=str(out_dir / "analysis_report.html"),
      )
    except Exception as e:
      _update(status="error", error=str(e), message=f"실패: {e}", trace=traceback.format_exc())

  class StudioHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
      return

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
      self.send_response(status)
      self.send_header("Content-Type", content_type)
      self.send_header("Content-Length", str(len(body)))
      self.send_header("Cache-Control", "no-store")
      self.end_headers()
      self.wfile.write(body)

    def _send_json(self, status: int, payload: Any) -> None:
      raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
      self._send_bytes(status, raw, "application/json; charset=utf-8")

    def do_GET(self) -> None:
      parsed = urlparse(self.path)
      path = parsed.path
      qs = parse_qs(parsed.query)

      if path in ("/", "/index.html"):
        req_local = qs.get("local_only", ["1" if local_only else "0"])[0] == "1"
        limit = int(qs.get("limit", [str(default_limit)])[0] or default_limit)
        groups, dev_label, dev_err = list_route_groups(
          limit=limit,
          local_only=req_local,
          base_dir=base_dir,
          config_path=config_path,
        )
        group_dicts = [g.to_dict() for g in groups]
        req_route = unquote(qs.get("route", [""])[0] or "").strip()
        active_route = ""
        active_report: dict[str, Any] | None = None
        if req_route:
          sp = _route_scan_dir(scans_dir, req_route) / "scan_summary.json"
          if sp.is_file():
            active_route = req_route
            active_report = json.loads(sp.read_text(encoding="utf-8"))
        if active_report is None:
          active_route, active_report = _pick_initial_route(group_dicts, scans_dir)
        cfg = load_config(config_path)
        html_doc = render_html_report(
          active_report,
          route_groups=group_dicts,
          active_route=active_route,
          device_label=dev_label,
          device_error=dev_err,
          device_dongle_id=str(cfg.get("dongle_id") or ""),
          live_api=True,
        )
        self._send_bytes(HTTPStatus.OK, html_doc.encode("utf-8"), "text/html; charset=utf-8")
        return

      if path == "/api/config":
        cfg = load_config(config_path)
        self._send_json(HTTPStatus.OK, {
          "ok": True,
          "host": str(cfg.get("host") or ""),
          "port": int(cfg.get("port") or DEFAULT_PORT),
          "dongle_id": str(cfg.get("dongle_id") or ""),
        })
        return

      if path == "/api/routes":
        req_local = qs.get("local_only", ["1" if local_only else "0"])[0] == "1"
        limit = int(qs.get("limit", [str(default_limit)])[0] or default_limit)
        groups, dev_label, dev_err = list_route_groups(
          limit=limit,
          local_only=req_local,
          base_dir=base_dir,
          config_path=config_path,
        )
        cfg = load_config(config_path)
        self._send_json(HTTPStatus.OK, {
          "ok": True,
          "device": dev_label,
          "device_error": dev_err,
          "dongle_id": str(cfg.get("dongle_id") or ""),
          "groups": [g.to_dict() for g in groups],
        })
        return

      if path.startswith("/api/report/"):
        route_name = unquote(path[len("/api/report/") :]).strip()
        summary_path = _route_scan_dir(scans_dir, route_name) / "scan_summary.json"
        if not summary_path.is_file():
          self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"분석 결과가 없습니다: {route_name}"})
          return
        report_dict = json.loads(summary_path.read_text(encoding="utf-8"))
        groups, dev_label, dev_err = list_route_groups(
          limit=default_limit,
          local_only=local_only,
          base_dir=base_dir,
          config_path=config_path,
        )
        cfg = load_config(config_path)
        payload = build_report_payload(
          report_dict,
          active_route=route_name,
          route_groups=[g.to_dict() for g in groups],
          device_label=dev_label,
          device_error=dev_err,
          device_dongle_id=str(cfg.get("dongle_id") or ""),
          live_api=True,
        )
        self._send_json(HTTPStatus.OK, payload)
        return

      if path.startswith("/api/jobs/"):
        job_id = path[len("/api/jobs/") :].strip()
        with jobs_lock:
          job = dict(jobs.get(job_id) or {})
        if not job:
          self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "job not found"})
          return
        self._send_json(HTTPStatus.OK, {"ok": True, "job": job})
        return

      self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown path: {path}"})

    def do_POST(self) -> None:
      parsed = urlparse(self.path)
      length = int(self.headers.get("Content-Length") or "0")
      raw = self.rfile.read(length) if length > 0 else b"{}"
      try:
        req_body = json.loads(raw.decode("utf-8"))
      except Exception as e:
        self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"Invalid JSON: {e}"})
        return

      if parsed.path == "/api/config":
        cfg = load_config(config_path)
        req_port = req_body.get("port")
        resolved_port = int(req_port if req_port is not None else (cfg.get("port") or DEFAULT_PORT))
        clear_dongle = bool(req_body.get("clear_dongle_id", False))
        manual_id = str(req_body.get("dongle_id") or "").strip().lower()
        manual_host = str(req_body.get("host") or "").strip()

        if clear_dongle:
          saved = save_config(
            host=manual_host or None,
            port=resolved_port,
            clear_dongle_id=True,
            config_path=config_path,
          )
          self._send_json(HTTPStatus.OK, {"ok": True, "config": saved, "message": "동글ID 고정 설정을 초기화했습니다."})
          return

        if manual_host and not manual_id:
          probed = probe_and_save_host(manual_host, port=resolved_port, config_path=config_path)
          saved = probed["config"]
          did = saved.get("dongle_id") or "-"
          msg = f"{manual_host}:{resolved_port} 저장 (동글ID={did})"
          self._send_json(HTTPStatus.OK, {"ok": True, "config": saved, "message": msg})
          return

        saved = save_config(
          host=manual_host or None,
          port=resolved_port,
          dongle_id=manual_id or None,
          config_path=config_path,
        )
        if manual_id and not manual_host:
          try:
            found = discover_device_on_wifi(dongle_id=manual_id, port=resolved_port)
            if found is not None:
              saved = save_config(host=found[0], port=resolved_port, dongle_id=manual_id, config_path=config_path)
          except Exception:
            pass
        self._send_json(HTTPStatus.OK, {
          "ok": True,
          "config": saved,
          "message": f"설정 저장 완료 (동글ID={saved.get('dongle_id') or '-'}, IP={saved.get('host') or '-'})",
        })
        return

      if parsed.path == "/api/discover":
        cfg = load_config(config_path)
        req_port = req_body.get("port")
        resolved_port = int(req_port if req_port is not None else (cfg.get("port") or DEFAULT_PORT))
        any_device = bool(req_body.get("any_device", True))
        target_id = None if any_device else (str(req_body.get("dongle_id") or cfg.get("dongle_id") or "").strip().lower() or None)
        try:
          found = discover_device_on_wifi(dongle_id=target_id, port=resolved_port)
        except Exception as e:
          self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})
          return
        if found is None:
          self._send_json(HTTPStatus.NOT_FOUND, {
            "ok": False,
            "error": f"동일 Wi-Fi에서 오파 단말을 찾지 못했습니다 (dongle_id={target_id or 'any'}).",
          })
          return
        found_host, info = found
        found_id = extract_dongle_id(info) or (target_id or "")
        saved = save_config(host=found_host, port=resolved_port, dongle_id=found_id or None, config_path=config_path)
        self._send_json(HTTPStatus.OK, {
          "ok": True,
          "host": found_host,
          "port": resolved_port,
          "dongle_id": str(saved.get("dongle_id") or found_id),
          "device_label": f"{found_host}:{resolved_port}",
          "message": f"Wi-Fi 자동 인식 완료: {found_host}:{resolved_port} (dongle_id={found_id or 'unknown'})",
        })
        return

      if parsed.path != "/api/analyze":
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown path: {parsed.path}"})
        return

      route_name = str(req_body.get("route") or "").strip()
      if not route_name:
        self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "route is required"})
        return
      force_rescan = bool(req_body.get("force_rescan", False))
      req_local_only = bool(req_body.get("local_only", local_only))

      with jobs_lock:
        for existing_id, j in jobs.items():
          if j.get("route") == route_name and j.get("status") in ("queued", "syncing", "scanning"):
            self._send_json(HTTPStatus.ACCEPTED, {"ok": True, "job_id": existing_id, "job": dict(j)})
            return
        job_id = uuid.uuid4().hex[:12]
        job = {
          "job_id": job_id,
          "route": route_name,
          "status": "queued",
          "progress": 5,
          "message": f"{route_name} 작업 대기 중...",
        }
        jobs[job_id] = job

      th = threading.Thread(
        target=_run_job,
        args=(job_id, route_name, force_rescan, req_local_only),
        daemon=True,
      )
      th.start()
      self._send_json(HTTPStatus.ACCEPTED, {"ok": True, "job_id": job_id, "job": dict(job)})

  return _ReusableThreadingHTTPServer((host, int(port)), StudioHandler)


def serve_studio(
  *,
  host: str = "127.0.0.1",
  port: int = 8765,
  default_limit: int = 20,
  local_only: bool = False,
  open_browser: bool = False,
  base_dir: Path | None = None,
  config_path: Path = CONFIG_PATH,
) -> None:
  """Start blocking local HTTP studio server (reuses running instance or picks next free port)."""
  srv: ThreadingHTTPServer | None = None
  try:
    srv = create_studio_server(
      host=host,
      port=port,
      default_limit=default_limit,
      local_only=local_only,
      base_dir=base_dir,
      config_path=config_path,
    )
  except OSError as e:
    if e.errno not in (errno.EADDRINUSE, 48, 98, 10048):
      raise
    if _is_studio_alive(host, int(port)):
      url = f"http://{host}:{int(port)}"
      print(f"이미 실행 중인 CarrotPilot 주행로그 스튜디오가 있습니다: {url}", flush=True)
      if open_browser:
        import webbrowser
        webbrowser.open(url)
      return
    for cand_port in list(range(int(port) + 1, int(port) + 11)) + [0]:
      try:
        srv = create_studio_server(
          host=host,
          port=cand_port,
          default_limit=default_limit,
          local_only=local_only,
          base_dir=base_dir,
          config_path=config_path,
        )
        print(f"포트 {port}가 사용 중이어서 {srv.server_address[1]} 포트로 실행합니다.", flush=True)
        break
      except OSError as e2:
        if e2.errno in (errno.EADDRINUSE, 48, 98, 10048):
          continue
        raise
    if srv is None:
      raise

  actual_port = srv.server_address[1]
  url = f"http://{host}:{actual_port}"
  print(f"CarrotPilot 주행로그 스튜디오 실행 중: {url} (Ctrl+C 로 종료)", flush=True)
  if open_browser:
    import webbrowser
    webbrowser.open(url)
  try:
    srv.serve_forever()
  except KeyboardInterrupt:
    pass
  finally:
    srv.shutdown()
    srv.server_close()

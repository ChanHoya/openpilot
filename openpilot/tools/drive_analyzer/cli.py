"""Unified CLI for Wi-Fi log sync, 1st-pass anomaly scan, and ver-1.0 regression gating."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from . import (
  CONFIG_PATH,
  DEFAULT_BASELINE_REF,
  DEFAULT_PORT,
  GOLDEN_DIR,
  REPORTS_DIR,
  ROUTES_DIR,
  SCANS_DIR,
)
from .html_report import render_html_report
from .regression import render_regression_markdown, run_regression
from .scanner import render_scan_markdown, scan_paths
from .server import serve_studio
from .sync import (
  _SAFE_NAME_RE,
  RouteGroup,
  add_to_golden,
  discover_device_on_wifi,
  extract_dongle_id,
  list_route_groups,
  load_config,
  probe_and_save_host,
  render_route_groups_markdown,
  save_config,
  sync_from_device,
)


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="python3 -m openpilot.tools.drive_analyzer",
    description="오파 단말 Wi-Fi 주행로그 일괄 수집, 1차 이상치 스캔, ver-1.0 회귀 검증 도구",
  )
  sub = parser.add_subparsers(dest="cmd", required=True)

  p_cfg = sub.add_parser("config", help="단말 접속 호스트/포트/dongle_id 저장")
  p_cfg.add_argument("--host", help="단말 IP 또는 호스트명 (예: 192.168.0.25)")
  p_cfg.add_argument("--dongle-id", help="유동 IP 자동 탐색용 dongle_id 수동 지정 (예: 0123456789abcdef)")
  p_cfg.add_argument("--clear-dongle-id", action="store_true", help="저장된 dongle_id를 지우고 자동 인식 모드로 초기화")
  p_cfg.add_argument("--port", type=int, help="Carrot Web 포트 (기본 7000)")
  p_cfg.add_argument("--timeout", type=float)

  p_disc = sub.add_parser("discover", help="동일 Wi-Fi에서 오파 단말 IP 및 dongle_id 자동 탐색")
  p_disc.add_argument("--dongle-id", help="특정 dongle_id와 일치하는 단말만 찾기 (미지정시 저장된 설정 사용)")
  p_disc.add_argument("--any", action="store_true", help="저장된 dongle_id를 무시하고 현재 Wi-Fi의 단말을 새로 자동 인식")
  p_disc.add_argument("--port", type=int, default=DEFAULT_PORT, help="Carrot Web 포트 (기본 7000)")
  p_disc.add_argument("--no-save", action="store_true", help="찾은 IP/dongle_id를 config.json에 저장하지 않음")

  p_sync = sub.add_parser("sync", help="동일 Wi-Fi 단말에서 주행리포트 및 신규 rlog.zst 일괄 수집")
  p_sync.add_argument("--host", help="단말 IP 또는 호스트명")
  p_sync.add_argument("--dongle-id", help="자동 탐색에 사용할 dongle_id")
  p_sync.add_argument("--port", type=int, help="Carrot Web 포트")
  p_sync.add_argument("--latest-routes", type=int, default=3, help="최신 루트 N개 동기화 (기본 3)")
  p_sync.add_argument("--route", help="특정 루트명 필터")
  p_sync.add_argument("--max-segments", type=int, help="루트당 최신 세그먼트 제한 수")
  p_sync.add_argument("--workers", type=int, default=3)

  p_scan = sub.add_parser("scan", help="로컬 rlog.zst 전수 1차 이상치 스캔 + HTML 시각화 리포트 생성")
  p_scan.add_argument("--dir", nargs="+", help="스캔할 폴더 또는 파일 경로")
  p_scan.add_argument("--latest", action="store_true", help="동기화된 가장 최신 루트만 스캔")
  p_scan.add_argument("--out-dir", help="결과 저장 디렉토리")
  p_scan.add_argument("--notes-json", help="추가 진단 요약/계획 JSON 파일 경로")
  p_scan.add_argument("--open", action="store_true", help="생성된 HTML 리포트를 기본 브라우저로 열기")
  p_scan.add_argument("--workers", type=int, default=4)

  p_run = sub.add_parser("run", help="sync + 1차 이상치 스캔 + HTML 리포트 한 번에 실행")
  p_run.add_argument("--host", help="단말 IP 또는 호스트명")
  p_run.add_argument("--dongle-id", help="자동 탐색에 사용할 dongle_id")
  p_run.add_argument("--port", type=int, help="Carrot Web 포트")
  p_run.add_argument("--latest-routes", type=int, default=2, help="최신 루트 N개 수집 (기본 2)")
  p_run.add_argument("--route", help="특정 루트명 필터")
  p_run.add_argument("--notes-json", help="추가 진단 요약/계획 JSON 파일 경로")
  p_run.add_argument("--open", action="store_true", help="생성된 HTML 리포트를 기본 브라우저로 열기")
  p_run.add_argument("--workers", type=int, default=3)

  p_rep = sub.add_parser("report", help="기존 scan_summary.json으로부터 HTML 시각화 리포트 재생성")
  p_rep.add_argument("--summary", required=True, help="scan_summary.json 경로")
  p_rep.add_argument("--notes-json", help="추가 진단 요약/계획 JSON 파일 경로")
  p_rep.add_argument("--out", help="출력 HTML 경로 (미지정시 summary와 같은 폴더의 analysis_report.html)")
  p_rep.add_argument("--open", action="store_true", help="생성된 HTML 리포트를 기본 브라우저로 열기")

  p_routes = sub.add_parser("routes", help="단말/로컬의 주행로그 그룹(루트 단위) 목록 조회")
  p_routes.add_argument("--host", help="단말 IP 또는 호스트명")
  p_routes.add_argument("--dongle-id", help="자동 탐색에 사용할 dongle_id")
  p_routes.add_argument("--port", type=int, help="Carrot Web 포트")
  p_routes.add_argument("--limit", type=int, default=15, help="표시할 최신 주행로그 그룹 수 (기본 15)")
  p_routes.add_argument("--local-only", action="store_true", help="단말 조회 없이 로컬 동기화 캐시만 표시")
  p_routes.add_argument("--json", action="store_true", help="JSON 형식으로 출력")

  p_ana = sub.add_parser("analyze", help="선택한 주행로그 그룹만 동기화 및 분석 + HTML 리포트 생성")
  p_ana.add_argument("--route", help="분석할 주행로그 그룹 번호(routes 순번) 또는 루트명(일부 포함 가능)")
  p_ana.add_argument("--host", help="단말 IP 또는 호스트명")
  p_ana.add_argument("--dongle-id", help="자동 탐색에 사용할 dongle_id")
  p_ana.add_argument("--port", type=int, help="Carrot Web 포트")
  p_ana.add_argument("--limit", type=int, default=20, help="조회할 최신 루트 범위 (기본 20)")
  p_ana.add_argument("--local-only", action="store_true", help="단말 다운로드 없이 로컬 캐시만 분석")
  p_ana.add_argument("--out-dir", help="결과 저장 디렉토리 (기본 scans/route_<루트>)")
  p_ana.add_argument("--notes-json", help="추가 진단 요약/계획 JSON 파일 경로")
  p_ana.add_argument("--open", action="store_true", help="생성된 HTML 리포트를 기본 브라우저로 열기")
  p_ana.add_argument("--workers", type=int, default=4)

  p_srv = sub.add_parser("serve", help="HTML 내 주행로그 그룹 선택 · 실시간 분석 로컬 웹 스튜디오 실행")
  p_srv.add_argument("--host", default="127.0.0.1", help="바인딩 호스트 (기본 127.0.0.1)")
  p_srv.add_argument("--port", type=int, default=8765, help="로컬 HTTP 포트 (기본 8765)")
  p_srv.add_argument("--limit", type=int, default=20, help="표시할 최신 주행로그 그룹 수 (기본 20)")
  p_srv.add_argument("--local-only", action="store_true", help="단말 조회 없이 로컬 캐시만 표시")
  p_srv.add_argument("--open", action="store_true", help="실행 후 기본 브라우저 자동 열기")

  p_reg = sub.add_parser("regress", help="ver-1.0 기준 A/B 리플레이 회귀 게이트 실행 (종방향+조향)")
  p_reg.add_argument("--baseline-ref", default=DEFAULT_BASELINE_REF, help="비교 기준 Git 태그/커밋 (기본 ver-1.0)")
  p_reg.add_argument("--mode", choices=("both", "long", "lat"), default="both", help="검증 대상 (기본 both)")
  p_reg.add_argument("--target-dir", nargs="*", default=(), help="개선 타겟 로그 경로")
  p_reg.add_argument("--golden-dir", nargs="*", default=(), help="ver-1.0 골든 회귀 로그 경로 (미지정시 golden_v1.0 사용)")
  p_reg.add_argument("--forbid-band", action="append", default=["0-25"], help="종방향 변경 금지 속도대 (반복 지정 가능)")
  p_reg.add_argument("--forbid-lat-band", action="append", default=["straight"], help="조향 변경 금지 구간대 (기본 straight)")
  p_reg.add_argument("--max-golden-change-pct", type=float, default=0.5)
  p_reg.add_argument("--max-golden-abs-delta", type=float, default=0.15)
  p_reg.add_argument("--json-out", help="JSON 결과 저장 경로")

  p_gold = sub.add_parser("golden-add", help="검증된 주행 로그를 golden_v1.0 코퍼스에 추가")
  p_gold.add_argument("paths", nargs="+", help="추가할 rlog 파일 또는 폴더")

  return parser


def _resolve_scan_dirs(args: argparse.Namespace) -> list[Path]:
  if args.dir:
    return [Path(d) for d in args.dir]
  if not ROUTES_DIR.is_dir():
    return []
  route_dirs = sorted(
    [p for p in ROUTES_DIR.iterdir() if p.is_dir()],
    key=lambda p: p.stat().st_mtime,
    reverse=True,
  )
  if args.latest and route_dirs:
    return [route_dirs[0]]
  return route_dirs[:3]


def _load_notes_json(path_str: str | None) -> dict[str, Any] | None:
  if not path_str:
    return None
  p = Path(path_str).expanduser()
  return json.loads(p.read_text(encoding="utf-8"))


def _maybe_open_html(html_path: Path, should_open: bool) -> None:
  if not should_open:
    return
  try:
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, str(html_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  except Exception:
    pass


def main(argv: Sequence[str] | None = None) -> int:
  parser = _build_parser()
  args = parser.parse_args(list(argv) if argv is not None else None)

  if args.cmd == "config":
    if not any([args.host, args.dongle_id, args.clear_dongle_id, args.port is not None, args.timeout is not None]):
      print("`--host`, `--dongle-id`, `--clear-dongle-id` 중 하나 이상을 지정하세요.", file=sys.stderr)
      return 1
    if args.host and not args.dongle_id and not args.clear_dongle_id:
      probed = probe_and_save_host(
        args.host,
        port=args.port,
        timeout_s=args.timeout,
        config_path=CONFIG_PATH,
      )
      cfg = probed["config"]
    else:
      cfg = save_config(
        host=args.host,
        port=args.port,
        timeout_s=args.timeout,
        dongle_id=args.dongle_id,
        clear_dongle_id=args.clear_dongle_id,
        config_path=CONFIG_PATH,
      )
    print(f"설정 저장 완료 ({CONFIG_PATH}): {json.dumps(cfg, ensure_ascii=False)}")
    return 0

  if args.cmd == "discover":
    cfg = load_config(CONFIG_PATH)
    target_id = None if args.any else (args.dongle_id or cfg.get("dongle_id"))
    found = discover_device_on_wifi(dongle_id=target_id, port=args.port)
    if found is None:
      print(f"Wi-Fi에서 단말을 찾지 못했습니다 (dongle_id={target_id or 'any'}).", file=sys.stderr)
      return 1
    found_host, info = found
    found_id = extract_dongle_id(info) or (target_id or "")
    if not args.no_save:
      save_config(host=found_host, port=args.port, dongle_id=found_id or None, config_path=CONFIG_PATH)
    sw = info.get("software") if isinstance(info.get("software"), dict) else {}
    print(
      f"탐색 성공: http://{found_host}:{args.port} | dongle_id={found_id or 'unknown'} | "
      f"branch={sw.get('branch', '-')} commit={str(sw.get('commit', '-'))[:8]}"
    )
    return 0

  if args.cmd == "sync":
    res = sync_from_device(
      host=args.host,
      port=args.port,
      dongle_id=args.dongle_id,
      latest_routes=args.latest_routes,
      route_filter=args.route,
      max_segments_per_route=args.max_segments,
      workers=args.workers,
    )
    print(json.dumps(res.to_dict(), indent=2, ensure_ascii=False))
    return 0 if not res.errors else 1

  if args.cmd == "scan":
    dirs = _resolve_scan_dirs(args)
    if not dirs:
      print("스캔할 로그가 없습니다. `--dir`을 지정하거나 `sync`를 먼저 실행하세요.", file=sys.stderr)
      return 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else (SCANS_DIR / stamp)
    notes = _load_notes_json(args.notes_json)
    report = scan_paths(dirs, reports_dir=REPORTS_DIR, out_dir=out_dir, workers=args.workers, custom_notes=notes)
    print(render_scan_markdown(report))
    html_path = out_dir / "analysis_report.html"
    print(f"저장 경로: {out_dir / 'scan_report.md'}")
    print(f"HTML 리포트: {html_path}")
    _maybe_open_html(html_path, args.open)
    return 0

  if args.cmd == "run":
    res = sync_from_device(
      host=args.host,
      port=args.port,
      dongle_id=args.dongle_id,
      latest_routes=args.latest_routes,
      route_filter=args.route,
      workers=args.workers,
    )
    scan_targets = [
      ROUTES_DIR / _SAFE_NAME_RE.sub("_", r)
      for r in res.routes_synced
      if (ROUTES_DIR / _SAFE_NAME_RE.sub("_", r)).exists()
    ]
    if not scan_targets and res.downloaded_files:
      scan_targets = [Path(p) for p in res.downloaded_files]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SCANS_DIR / stamp
    notes = _load_notes_json(args.notes_json)
    report = scan_paths(scan_targets, reports_dir=REPORTS_DIR, out_dir=out_dir, workers=max(2, args.workers), custom_notes=notes)
    print(f"[동기화] 루트 {len(res.routes_synced)}개 | 신규 {res.segments_downloaded}개 / 스킵 {res.segments_skipped}개 | 리포트 +{res.reports_saved}")
    if res.errors:
      print(f"[동기화 경고] {len(res.errors)}건: {res.errors[:3]}")
    print(render_scan_markdown(report))
    html_path = out_dir / "analysis_report.html"
    print(f"저장 경로: {out_dir / 'scan_report.md'}")
    print(f"HTML 리포트: {html_path}")
    _maybe_open_html(html_path, args.open)
    return 0

  if args.cmd == "report":
    summary_path = Path(args.summary).expanduser()
    report_dict = json.loads(summary_path.read_text(encoding="utf-8"))
    notes = _load_notes_json(args.notes_json)
    out_html = Path(args.out).expanduser() if args.out else (summary_path.parent / "analysis_report.html")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(render_html_report(report_dict, custom_notes=notes), encoding="utf-8")
    print(f"HTML 리포트 생성 완료: {out_html}")
    _maybe_open_html(out_html, args.open)
    return 0

  if args.cmd == "regress":
    golden_dirs = [Path(p) for p in args.golden_dir] if args.golden_dir else ([GOLDEN_DIR] if GOLDEN_DIR.is_dir() else [])
    target_dirs = [Path(p) for p in args.target_dir]
    rep = run_regression(
      baseline_ref=args.baseline_ref,
      target_paths=target_dirs,
      golden_paths=golden_dirs,
      mode=args.mode,
      forbid_bands=tuple(dict.fromkeys(args.forbid_band)),
      forbid_lat_bands=tuple(dict.fromkeys(args.forbid_lat_band)),
      max_golden_change_pct=args.max_golden_change_pct,
      max_golden_abs_delta=args.max_golden_abs_delta,
    )
    print(render_regression_markdown(rep))
    if args.json_out:
      Path(args.json_out).write_text(json.dumps(rep.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if rep.passed else 2

  if args.cmd == "golden-add":
    added, skipped = add_to_golden(args.paths, golden_dir=GOLDEN_DIR)
    print(f"golden_v1.0 추가: {len(added)}개 / 기존 스킵: {len(skipped)}개 ({GOLDEN_DIR})")
    return 0

  if args.cmd == "routes":
    groups, dev_label, dev_err = list_route_groups(
      host=args.host,
      port=args.port,
      dongle_id=args.dongle_id,
      limit=args.limit,
      local_only=args.local_only,
      base_dir=ROUTES_DIR.parent,
      config_path=CONFIG_PATH,
    )
    if args.json:
      print(json.dumps({
        "device": dev_label,
        "device_error": dev_err,
        "groups": [g.to_dict() for g in groups],
      }, indent=2, ensure_ascii=False))
    else:
      print(render_route_groups_markdown(groups, device_label=dev_label, device_error=dev_err))
    return 0

  if args.cmd == "analyze":
    groups, dev_label, dev_err = list_route_groups(
      host=args.host,
      port=args.port,
      dongle_id=args.dongle_id,
      limit=args.limit,
      local_only=args.local_only,
      base_dir=ROUTES_DIR.parent,
      config_path=CONFIG_PATH,
    )
    if not groups:
      print(f"분석할 주행로그 그룹이 없습니다. {dev_err}", file=sys.stderr)
      return 1
    query = (args.route or "").strip()
    if not query:
      print(render_route_groups_markdown(groups, device_label=dev_label, device_error=dev_err))
      if not sys.stdin.isatty():
        print("비대화형 모드입니다. `--route <번호 또는 루트명>`을 지정하세요.", file=sys.stderr)
        return 1
      try:
        query = input("분석할 주행로그 그룹 번호 또는 루트명을 입력하세요: ").strip()
      except EOFError:
        return 1
      if not query:
        return 1
    try:
      selected = _resolve_selected_route_group(groups, query)
    except ValueError as e:
      print(str(e), file=sys.stderr)
      return 1

    if not args.local_only and (selected.local_segments == 0 or (selected.device_segments > 0 and selected.local_segments < selected.device_segments)):
      res = sync_from_device(
        host=args.host,
        port=args.port,
        dongle_id=args.dongle_id,
        latest_routes=max(50, int(args.limit)),
        route_filter=selected.route,
        workers=args.workers,
        base_dir=ROUTES_DIR.parent,
        config_path=CONFIG_PATH,
      )
      print(f"[선택 동기화] {selected.route}: 신규 {res.segments_downloaded}개 / 스킵 {res.segments_skipped}개 / 리포트 +{res.reports_saved}")
      if res.errors:
        print(f"[동기화 경고] {len(res.errors)}건: {res.errors[:3]}")

    safe_route = _SAFE_NAME_RE.sub("_", selected.route)
    route_dir = Path(selected.local_dir) if (selected.local_dir and Path(selected.local_dir).exists()) else (ROUTES_DIR / safe_route)
    if not route_dir.exists():
      print(f"로컬 경로가 없습니다: {route_dir}", file=sys.stderr)
      return 1
    out_dir = Path(args.out_dir) if args.out_dir else (SCANS_DIR / f"route_{safe_route}")
    notes = _load_notes_json(args.notes_json)
    report = scan_paths([route_dir], reports_dir=REPORTS_DIR, out_dir=out_dir, workers=args.workers, custom_notes=notes)
    print(render_scan_markdown(report))
    html_path = out_dir / "analysis_report.html"
    print(f"저장 경로: {out_dir / 'scan_report.md'}")
    print(f"HTML 리포트: {html_path}")
    _maybe_open_html(html_path, args.open)
    return 0

  if args.cmd == "serve":
    serve_studio(
      host=args.host,
      port=args.port,
      default_limit=args.limit,
      local_only=args.local_only,
      open_browser=args.open,
      base_dir=ROUTES_DIR.parent,
      config_path=CONFIG_PATH,
    )
    return 0

  return 1


def _resolve_selected_route_group(groups: Sequence[RouteGroup], query: str) -> RouteGroup:
  if query.isdigit():
    idx = int(query)
    if 1 <= idx <= len(groups):
      return groups[idx - 1]
    raise ValueError(f"유효하지 않은 그룹 번호입니다: {idx} (가능 범위: 1..{len(groups)})")
  exact = [g for g in groups if g.route == query]
  if exact:
    return exact[0]
  matches = [g for g in groups if query in g.route]
  if len(matches) == 1:
    return matches[0]
  if len(matches) > 1:
    cands = ", ".join(g.route for g in matches[:5])
    raise ValueError(f"'{query}'와 일치하는 그룹이 여러 개입니다: {cands}")
  raise ValueError(f"'{query}'에 해당하는 주행로그 그룹을 찾지 못했습니다.")


if __name__ == "__main__":
  raise SystemExit(main())

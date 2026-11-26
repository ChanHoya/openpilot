"""Self-contained interactive HTML visual report generator for drive_analyzer."""

from __future__ import annotations

from datetime import datetime
import html
import json
import re
from typing import Any, Sequence

_SAFE_ROUTE_RE = re.compile(r"[^A-Za-z0-9._-]+")

LAT_CATEGORIES = frozenset({
  "manual_steer_override",
  "steer_pingpong",
  "curve_lat_overshoot",
  "lane_center_bias",
})

LONG_CATEGORIES = frozenset({
  "auto_hard_brake",
  "manual_brake_override",
  "low_speed_surge",
  "mid_speed_surge",
  "far_lead_late_decel",
})

CATEGORY_META: dict[str, dict[str, str]] = {
  "auto_hard_brake": {"label": "자동 급감속", "axis": "long"},
  "manual_brake_override": {"label": "수동 브레이크 개입", "axis": "long"},
  "low_speed_surge": {"label": "저속 서지/꿀렁임", "axis": "long"},
  "mid_speed_surge": {"label": "중속 서지", "axis": "long"},
  "far_lead_late_decel": {"label": "원거리 선행차 감속지연", "axis": "long"},
  "manual_steer_override": {"label": "수동 조향 개입", "axis": "lat"},
  "steer_pingpong": {"label": "코너 조향 와리가리", "axis": "lat"},
  "curve_lat_overshoot": {"label": "커브 횡가속 과다/포화", "axis": "lat"},
  "lane_center_bias": {"label": "직선 차선 편향", "axis": "lat"},
}


def _as_report_dict(report: Any) -> dict[str, Any]:
  if isinstance(report, dict):
    return report
  if hasattr(report, "to_dict"):
    return report.to_dict()
  raise TypeError(f"Unsupported report type: {type(report)!r}")


def _collect_observed_params(segments: list[dict[str, Any]]) -> dict[str, str]:
  merged: dict[str, str] = {}
  for seg in segments:
    params = seg.get("params") or {}
    if isinstance(params, dict):
      for k, v in params.items():
        if v not in (None, ""):
          merged[str(k)] = str(v)
  return merged


def _compute_axis_scores(report_dict: dict[str, Any]) -> tuple[int, int, int]:
  """Return (overall_score, long_score, lat_score) scaled 0..100."""
  if not (report_dict.get("segments") or []):
    return 100, 100, 100
  drive_reports = report_dict.get("drive_reports_summary") or []
  if drive_reports:
    top = drive_reports[0]
    overall = top.get("score")
    speed_info = top.get("speed") if isinstance(top.get("speed"), dict) else {}
    if isinstance(overall, (int, float)):
      anomalies = report_dict.get("anomalies") or []
      long_pen = 0.0
      lat_pen = 0.0
      for a in anomalies:
        cat = a.get("category", "")
        sev = a.get("severity", "medium")
        if cat in LONG_CATEGORIES:
          long_pen += 12.0 if sev == "high" else 3.5
        else:
          lat_pen += 10.0 if sev == "high" else (2.5 if cat != "manual_steer_override" else 1.0)
      long_s = speed_info.get("long_score")
      lat_s = speed_info.get("lat_score")
      if not isinstance(long_s, (int, float)):
        long_s = int(max(35, min(100, round(100 - min(long_pen, 60)))))
      if not isinstance(lat_s, (int, float)):
        lat_s = int(max(35, min(100, round(100 - min(lat_pen, 60)))))
      return int(round(float(overall))), int(round(float(long_s))), int(round(float(lat_s)))

  anomalies = report_dict.get("anomalies") or []
  long_pen = 0.0
  lat_pen = 0.0
  for a in anomalies:
    cat = a.get("category", "")
    sev = a.get("severity", "medium")
    if cat in LONG_CATEGORIES:
      long_pen += 12.0 if sev == "high" else 3.5
    else:
      lat_pen += 10.0 if sev == "high" else (2.5 if cat != "manual_steer_override" else 1.0)
  long_s = int(max(30, min(100, round(100 - min(long_pen, 65)))))
  lat_s = int(max(30, min(100, round(100 - min(lat_pen, 65)))))
  overall = int(round((long_s + lat_s) / 2))
  return overall, long_s, lat_s


def build_recommendations(report_dict: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  """Build concrete Parameter Tuning Guides and ver-1.0-safe Code Improvement Plans."""
  segments = report_dict.get("segments") or []
  anomalies = report_dict.get("anomalies") or []
  params = _collect_observed_params(segments)

  counts: dict[str, int] = {k: 0 for k in CATEGORY_META}
  for a in anomalies:
    cat = a.get("category", "")
    if cat in counts:
      counts[cat] += 1

  valid_segs = [s for s in segments if not s.get("error")]
  max_lat_accel = max((float(s.get("lat_accel_max", 0.0) or 0.0) for s in valid_segs), default=0.0)
  max_curve_rms = max((float(s.get("curve_steer_rate_rms", 0.0) or 0.0) for s in valid_segs), default=0.0)
  total_sat_frames = sum(int(s.get("steer_sat_frames", 0) or 0) for s in valid_segs)

  param_guides: list[dict[str, Any]] = []
  code_plans: list[dict[str, Any]] = []

  ac_factor = params.get("AutoCurveSpeedFactor", "100")
  map_factor = params.get("MapTurnSpeedFactor", "90")
  ac_mode = params.get("TurnSpeedControlMode", "2")

  if counts["curve_lat_overshoot"] > 0 or max_lat_accel >= 2.6 or total_sat_frames >= 10:
    param_guides.append({
      "param": "AutoCurveSpeedFactor / MapTurnSpeedFactor",
      "axis": "both",
      "priority": "주의",
      "current": f"AutoCurve={ac_factor}%, MapTurn={map_factor}% (Mode={ac_mode})",
      "direction": "코너 진입속도를 높이려면 AutoCurveSpeedFactor ↓ (100→90) · 감속을 늘리려면 ↑ (100→110), MapTurn은 높일수록(90→100) 빠름",
      "condition": (
        "AutoCurveSpeedFactor는 카메라 곡률 기반 감속 감도를 나누는 값(수치↓ = 감속↓ = 진입속도↑, Mode 1·2 상시 적용)이고, "
        "MapTurnSpeedFactor는 지도 TBT 턴 속도에 곱하는 비율(수치↑ = 턴속도↑, Mode 2에서 -500m<xDistToTurn<500m 구간 적용)입니다. "
        "둘은 min(vturn, route) 중 더 낮은 목표속도 하나만 채택되므로 곱셈으로 중복 감속되지 않습니다."
      ),
      "caution": (
        f"이번 주행에서 curve_lat_overshoot {counts['curve_lat_overshoot']}건(최대 횡가속 {max_lat_accel:.2f} m/s², 토크포화 {total_sat_frames}프레임)이 관찰되었습니다. "
        "진입속도를 더 높일 경우 포화 구간에서 횡밀림이 커질 수 있으므로, 파라미터 변경 직후 코너조향RMS와 횡가속 포화를 함께 재스캔하세요."
      ),
    })
  else:
    param_guides.append({
      "param": "AutoCurveSpeedFactor / MapTurnSpeedFactor",
      "axis": "both",
      "priority": "참고",
      "current": f"AutoCurve={ac_factor}%, MapTurn={map_factor}% (Mode={ac_mode})",
      "direction": "진입속도 상향시: AutoCurveSpeedFactor 100→90 (하향), TBT 턴 상향시: MapTurnSpeedFactor 90→100 (상향)",
      "condition": (
        "AutoCurveSpeedFactor(카메라 곡률, Mode 1·2)는 목표 횡가속을 나누는 계수라 낮출수록 진입속도가 빨라지고, "
        "MapTurnSpeedFactor(지도 턴, Mode 2에서는 TBT 전후 500m, Mode 3·4 상시)는 높일수록 빨라집니다. "
        "최종 목표속도는 min(vturn, route)로 결합되어 중복 감속이 발생하지 않습니다."
      ),
      "caution": (
        f"이번 주행의 코너조향RMS 최대 {max_curve_rms:.1f}°/s, 와리가리(steer_pingpong) 0건으로 안정적입니다. "
        "진입속도 상향 후에도 steer_pingpong·curve_lat_overshoot·curve_steer_rate_rms가 증가하지 않는지 바로 비교 확인하세요."
      ),
    })

  t_follow_summary = ", ".join(
    f"G{i}={params[f'TFollowGap{i}']}"
    for i in range(1, 5)
    if f"TFollowGap{i}" in params
  ) or "기본 GAP 설정"
  stop_dist = params.get("StopDistanceCarrot", "기본값")
  if counts["low_speed_surge"] > 0 or counts["manual_brake_override"] >= 4:
    param_guides.append({
      "param": "TFollowGap1~4 / StopDistanceCarrot / LongTuningKpV",
      "axis": "long",
      "priority": "권장" if counts["low_speed_surge"] >= 2 else "참고",
      "current": f"{t_follow_summary} | StopDistance={stop_dist}",
      "direction": "서행·정차 직전 밀림이 크면 TFollowGap 또는 StopDistanceCarrot ↑ (+0.3~0.5m), 저속 꿀렁임시 KpV 소폭 ↓",
      "condition": (
        f"저속 서지 {counts['low_speed_surge']}건, 수동 브레이크 개입 {counts['manual_brake_override']}건이 감지되었습니다. "
        "코드 수정 전 파라미터 차원에서 차간거리 여유(TFollowGap)와 정지 목표거리(StopDistanceCarrot)를 1단계 먼저 조절해 교차로·정체 체감을 맞추는 것이 안전합니다."
      ),
      "caution": "파라미터 조정으로 해결되면 ver-1.0 종방향 제어 코드는 건드리지 않고 그대로 유지합니다.",
    })

  curve_bias = params.get("CurveInnerBias", "0")
  lat_factor = params.get("LatAccelFactor", "-")
  steer_delay = params.get("SteerActuatorDelay", "-")
  if counts["steer_pingpong"] > 0 or counts["lane_center_bias"] > 0 or counts["curve_lat_overshoot"] > 0:
    param_guides.append({
      "param": "CurveInnerBias / LanelessCenterAdjust / SteerActuatorDelay",
      "axis": "lat",
      "priority": "권장",
      "current": f"CurveInnerBias={curve_bias}, LatAccelFactor={lat_factor}, SteerDelay={steer_delay}",
      "direction": "코너 바깥 밀림시 CurveInnerBias 소폭 +, 직선 쏠림시 LanelessCenterAdjust 0.02 단위 보정",
      "condition": (
        "CurveInnerBias는 |curvature| >= 0.002 곡선 구간에서만 내측 바이어스로 동작하므로 직선 주행(0% 영향)에 간섭하지 않습니다. "
        "반면 SteerActuatorDelay / LatAccelFactor 변경은 전 속도대 조향 위상에 영향을 주므로 최소 단위로만 조정하세요."
      ),
      "caution": f"이번 주행 기준 최대 횡가속 {max_lat_accel:.2f} m/s², 코너조향RMS {max_curve_rms:.1f}°/s 입니다.",
    })

  target_paths = [s.get("segment_path", "") for s in valid_segs if s.get("anomalies")]
  sample_target = target_paths[0] if target_paths else "<이상 세그먼트 경로>"

  if counts["auto_hard_brake"] > 0 or counts["far_lead_late_decel"] > 0 or counts["mid_speed_surge"] > 0:
    found_parts = []
    if counts["auto_hard_brake"]:
      found_parts.append(f"자동 급감속 {counts['auto_hard_brake']}건")
    if counts["far_lead_late_decel"]:
      found_parts.append(f"원거리 선행차 감속지연 {counts['far_lead_late_decel']}건")
    if counts["mid_speed_surge"]:
      found_parts.append(f"중속 서지 {counts['mid_speed_surge']}건")
    code_plans.append({
      "id": "PLAN-LONG-01",
      "title": "종방향 이상 감속/서지 고립 보정 계획",
      "axis": "long",
      "status": "승인 대기",
      "reason": ", ".join(found_parts) + "이 검출되어 속도대 고립 보정 검토가 필요합니다.",
      "target_condition": "문제가 발생한 속도·선행거리 구간(예: v >= 50 km/h & d >= 45m 또는 해당 서지 대역)에만 조건부 적용",
      "change_scope": "openpilot/selfdrive/controls/lib/longitudinal_planner.py 또는 carrot longitudinal shaper (승인 전 미수정)",
      "forbidden_regression": "비대상 속도대(0-25 km/h 서행·정체) 변경률 0% 및 golden_v1.0 최대 변화율 0.5% 이내",
      "verify_cmd": (
        f"PYTHONPATH=. uv run --no-sync python3 -m openpilot.tools.drive_analyzer regress "
        f"--baseline-ref ver-1.0 --mode both --target-dir {sample_target} --forbid-band 0-25 --forbid-lat-band straight"
      ),
    })

  if counts["steer_pingpong"] > 0 or counts["curve_lat_overshoot"] > 0 or counts["lane_center_bias"] > 0:
    lat_parts = []
    if counts["steer_pingpong"]:
      lat_parts.append(f"코너 와리가리 {counts['steer_pingpong']}건")
    if counts["curve_lat_overshoot"]:
      lat_parts.append(f"커브 횡가속 과다/포화 {counts['curve_lat_overshoot']}건")
    if counts["lane_center_bias"]:
      lat_parts.append(f"직선 차선 편향 {counts['lane_center_bias']}건")
    code_plans.append({
      "id": "PLAN-LAT-01",
      "title": "커브 조향·횡가속 포화 고립 보정 계획",
      "axis": "lat",
      "status": "승인 대기",
      "reason": ", ".join(lat_parts) + "이 검출되었습니다. 파라미터 조정으로 부족할 경우에만 고립 수정을 검토합니다.",
      "target_condition": "중고속 커브 진입/유지 구간(|curvature| >= 0.002 또는 |lat_accel| >= 1.8 m/s²) 한정",
      "change_scope": "openpilot/selfdrive/controls/lib/drive_helpers.py (커브 진입 스무딩/내측 어시스트, 승인 전 미수정)",
      "forbidden_regression": "직선 주행(straight) 구간 변경률 0% 및 0-25 km/h 종방향 변경률 0% 유지",
      "verify_cmd": (
        f"PYTHONPATH=. uv run --no-sync python3 -m openpilot.tools.drive_analyzer regress "
        f"--baseline-ref ver-1.0 --mode both --target-dir {sample_target} --forbid-band 0-25 --forbid-lat-band straight"
      ),
    })

  if not code_plans:
    code_plans.append({
      "id": "PLAN-KEEP-V1",
      "title": "ver-1.0 품질 양호 — 추가 코드 수정 불필요",
      "axis": "both",
      "status": "수정 불필요 (ver-1.0 유지)",
      "reason": (
        "자동 급정거·중속 서지·원거리 감속지연·조향 와리가리·직선 차선 편향이 모두 0건입니다. "
        "핵심 제어 로직은 ver-1.0 기준선을 그대로 유지하고, 필요 시 위 파라미터 가이드만 미세 조정하는 것을 권장합니다."
      ),
      "target_condition": "해당 없음 (런타임 코드 변경 0줄)",
      "change_scope": "openpilot/selfdrive/ 변경 없음",
      "forbidden_regression": "ver-1.0 대비 종방향·횡방향 전 구간 100% 동일 동작",
      "verify_cmd": "git diff ver-1.0 HEAD -- openpilot/selfdrive/",
    })

  return param_guides, code_plans


def build_report_payload(
  report: Any | None,
  *,
  title: str | None = None,
  custom_notes: dict[str, Any] | None = None,
  active_route: str = "",
  route_groups: Sequence[dict[str, Any]] | None = None,
  device_label: str = "",
  device_error: str = "",
  device_dongle_id: str = "",
  live_api: bool = False,
) -> dict[str, Any]:
  """Build normalized JSON payload consumed by both static HTML and the live studio API."""
  report_dict = _as_report_dict(report) if report is not None else {}
  segments = list(report_dict.get("segments") or [])
  anomalies = list(report_dict.get("anomalies") or [])
  drive_reports = list(report_dict.get("drive_reports_summary") or [])

  overall_score, long_score, lat_score = _compute_axis_scores(report_dict)
  if segments:
    param_guides, code_plans = build_recommendations(report_dict)
  else:
    param_guides, code_plans = [], []

  notes = custom_notes or {}
  if isinstance(notes.get("extra_param_guides"), list):
    param_guides.extend(notes["extra_param_guides"])
  if isinstance(notes.get("extra_code_plans"), list):
    code_plans.extend(notes["extra_code_plans"])

  total_frames = sum(int(s.get("frames", 0) or 0) for s in segments)
  total_long_active = sum(int(s.get("active_frames", 0) or 0) for s in segments)
  total_lat_active = sum(int(s.get("lat_active_frames", 0) or 0) for s in segments)
  total_dur_s = sum(float(s.get("duration_s", 0.0) or 0.0) for s in segments)
  long_count = sum(1 for a in anomalies if a.get("category") in LONG_CATEGORIES)
  lat_count = sum(1 for a in anomalies if a.get("category") in LAT_CATEGORIES)
  high_count = sum(1 for a in anomalies if a.get("severity") == "high")

  commits = sorted({str(s.get("commit", ""))[:8] for s in segments if s.get("commit")})
  branches = sorted({str(s.get("branch", "")) for s in segments if s.get("branch")})
  commit_badge = ", ".join(commits) if commits else "-"
  branch_badge = ", ".join(branches) if branches else "-"

  inferred_active = active_route
  if not inferred_active and segments:
    routes_in_segs = [str(s.get("route") or "") for s in segments if s.get("route")]
    if len(set(routes_in_segs)) == 1:
      inferred_active = routes_in_segs[0]

  report_title = title or notes.get("title") or "CarrotPilot 주행 로그 분석 스튜디오"
  generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

  return {
    "title": report_title,
    "generated_at": generated_at,
    "live_api": bool(live_api),
    "device_label": device_label,
    "device_error": device_error,
    "device_dongle_id": device_dongle_id,
    "active_route": inferred_active,
    "route_groups": list(route_groups or []),
    "scores": {"overall": overall_score, "long": long_score, "lat": lat_score},
    "stats": {
      "segment_count": len(segments),
      "total_duration_s": round(total_dur_s, 1),
      "total_frames": total_frames,
      "long_active_frames": total_long_active,
      "lat_active_frames": total_lat_active,
      "long_active_pct": round(100.0 * total_long_active / max(total_frames, 1), 1),
      "lat_active_pct": round(100.0 * total_lat_active / max(total_frames, 1), 1),
      "v_max_kph": max((float(s.get("v_max_kph", 0.0) or 0.0) for s in segments), default=0.0),
      "lat_accel_max": max((float(s.get("lat_accel_max", 0.0) or 0.0) for s in segments), default=0.0),
      "anomaly_count": len(anomalies),
      "long_count": long_count,
      "lat_count": lat_count,
      "high_count": high_count,
      "commit": commit_badge,
      "branch": branch_badge,
    },
    "category_meta": CATEGORY_META,
    "drive_reports": drive_reports,
    "segments": segments,
    "anomalies": anomalies,
    "param_guides": param_guides,
    "code_plans": code_plans,
    "custom_notes": notes,
  }


def render_html_report(
  report: Any | None,
  *,
  title: str | None = None,
  custom_notes: dict[str, Any] | None = None,
  active_route: str = "",
  route_groups: Sequence[dict[str, Any]] | None = None,
  device_label: str = "",
  device_error: str = "",
  device_dongle_id: str = "",
  live_api: bool = False,
) -> str:
  """Render a completely offline, self-contained interactive HTML visual drive studio & report."""
  payload = build_report_payload(
    report,
    title=title,
    custom_notes=custom_notes,
    active_route=active_route,
    route_groups=route_groups,
    device_label=device_label,
    device_error=device_error,
    device_dongle_id=device_dongle_id,
    live_api=live_api,
  )
  report_title = str(payload["title"])
  payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

  return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{html.escape(report_title)}</title>
<style>
  :root {{
    --bg: #0B0F14;
    --surface: #121821;
    --elevated: #18212D;
    --border: #1E293B;
    --border-strong: #334155;
    --text: #E2E8F0;
    --text-dim: #94A3B8;
    --text-muted: #64748B;
    --cyan: #38BDF8;
    --green: #22C55E;
    --amber: #F59E0B;
    --red: #EF4444;
    --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    --sans: "Pretendard", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  * {{ box-sizing: border-box; }}
  html {{ scroll-behavior: smooth; }}
  body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }}
  .mono {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}
  .muted {{ color: var(--text-dim); }}
  a {{ color: var(--cyan); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  /* Top Sticky Header */
  .topbar {{
    position: sticky;
    top: 0;
    z-index: 40;
    background: #0B0F14F2;
    border-bottom: 1px solid var(--border);
    padding: 10px 20px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }}
  .brand-wrap {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
  .brand-title {{ font-size: 15px; font-weight: 700; letter-spacing: -0.01em; color: var(--text); }}
  .badge {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid var(--border-strong);
    background: var(--surface);
    color: var(--text-dim);
    font-size: 11px;
    font-family: var(--mono);
  }}
  .badge-cyan {{ border-color: #0EA5E966; color: var(--cyan); background: #0C4A6E33; }}
  .badge-green {{ border-color: #22C55E55; color: var(--green); background: #14532D33; }}
  .badge-amber {{ border-color: #F59E0B55; color: var(--amber); background: #78350F33; }}
  .badge-red {{ border-color: #EF444455; color: var(--red); background: #7F1D1D33; }}

  .nav-pills {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .nav-pill {{
    padding: 4px 10px;
    border-radius: 4px;
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text-dim);
    font-size: 12px;
    font-weight: 500;
    transition: border-color .15s, color .15s;
  }}
  .nav-pill:hover {{ border-color: var(--cyan); color: var(--text); text-decoration: none; }}

  /* Layout Container */
  .container {{ max-width: 1440px; margin: 0 auto; padding: 18px 20px 56px; display: flex; flex-direction: column; gap: 24px; }}
  section {{ scroll-margin-top: 64px; }}
  .sec-head {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  .sec-title {{ font-size: 17px; font-weight: 700; letter-spacing: -0.01em; margin: 0; }}
  .sec-sub {{ font-size: 12px; color: var(--text-dim); }}

  .panel {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px 16px;
  }}
  .card-kicker {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-dim);
    margin-bottom: 8px;
    font-weight: 600;
  }}

  /* Executive Summary Grid (Asymmetric Z-pattern) */
  .summary-grid {{
    display: grid;
    grid-template-columns: minmax(280px, 1.15fr) minmax(280px, 1fr) minmax(280px, 1fr);
    gap: 14px;
  }}
  @media (max-width: 1080px) {{ .summary-grid {{ grid-template-columns: 1fr; }} }}

  .gauges-row {{ display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }}
  .gauge-card {{ display: flex; align-items: center; gap: 10px; }}
  .gauge-title {{ font-size: 13px; font-weight: 600; color: var(--text); }}
  .gauge-sub {{ font-size: 11px; color: var(--text-dim); }}

  .kpi-strip {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 8px;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
  }}
  .kpi-cell .kpi-v {{ font-size: 16px; font-weight: 700; color: var(--text); font-family: var(--mono); }}
  .kpi-cell .kpi-l {{ font-size: 11px; color: var(--text-dim); }}

  .report-match-main {{ display: flex; gap: 14px; align-items: center; flex-wrap: wrap; }}
  .grade-box {{
    padding: 8px 12px;
    border: 1px solid var(--border-strong);
    background: var(--elevated);
    border-radius: 5px;
    min-width: 96px;
    text-align: center;
  }}
  .grade-num {{ font-size: 24px; font-weight: 700; font-family: var(--mono); color: var(--cyan); }}
  .grade-unit {{ display: block; font-size: 11px; color: var(--text-dim); }}
  .report-match-stats {{ font-size: 12px; display: flex; flex-direction: column; gap: 3px; }}
  .chip-row {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }}
  .chip {{ font-size: 11px; padding: 2px 7px; border-radius: 4px; border: 1px solid var(--border-strong); }}
  .chip-warn {{ color: var(--amber); border-color: #F59E0B55; background: #78350F22; }}
  .chip-ok {{ color: var(--green); border-color: #22C55E55; background: #14532D22; }}

  /* Anomaly Breakdown Bars */
  .bar-row {{ display: grid; grid-template-columns: 148px 1fr 34px; align-items: center; gap: 8px; margin-bottom: 6px; font-size: 12px; }}
  .bar-track {{ height: 8px; background: var(--bg); border: 1px solid var(--border); border-radius: 2px; overflow: hidden; }}
  .bar-fill {{ height: 100%; }}
  .bar-long {{ background: var(--cyan); }}
  .bar-lat {{ background: var(--amber); }}
  .bar-high {{ background: var(--red); }}

  /* Segment Heatmap Strip */
  .heatmap-wrap {{ margin-top: 14px; }}
  .heatmap-strip {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(92px, 1fr));
    gap: 6px;
    margin-top: 8px;
  }}
  .seg-tile {{
    background: var(--elevated);
    border: 1px solid var(--border);
    border-top: 3px solid var(--green);
    border-radius: 4px;
    padding: 6px 8px;
    cursor: pointer;
    text-align: left;
    color: var(--text);
    transition: border-color .12s, background .12s;
  }}
  .seg-tile:hover {{ border-color: var(--cyan); }}
  .seg-tile.active {{ outline: 2px solid var(--cyan); background: #0C4A6E33; }}
  .seg-tile.sev-med {{ border-top-color: var(--amber); }}
  .seg-tile.sev-high {{ border-top-color: var(--red); }}
  .seg-tile.sev-err {{ border-top-color: var(--text-muted); opacity: 0.7; }}
  .seg-tile-id {{ font-family: var(--mono); font-size: 11px; font-weight: 700; }}
  .seg-tile-meta {{ font-size: 10px; color: var(--text-dim); margin-top: 2px; }}

  .notes-banner {{ margin-top: 14px; border-left: 3px solid var(--cyan); }}
  .notes-headline {{ font-size: 14px; font-weight: 600; margin-bottom: 6px; }}
  .notes-list {{ margin: 0; padding-left: 18px; color: var(--text-dim); font-size: 13px; }}

  /* Charts & Controls */
  .chart-toolbar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 10px;
  }}
  .toolbar-group {{ display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }}
  select, input[type="search"], button.btn {{
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    padding: 5px 10px;
    font-size: 12px;
    font-family: var(--sans);
  }}
  button.btn {{ cursor: pointer; }}
  button.btn:hover {{ border-color: var(--cyan); color: var(--cyan); }}
  button.btn.active-btn {{ background: #0C4A6E55; border-color: var(--cyan); color: var(--cyan); }}

  .legend-row {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 11px; color: var(--text-dim); margin-bottom: 6px; }}
  .leg-item {{ display: inline-flex; align-items: center; gap: 5px; }}
  .leg-line {{ width: 16px; height: 2px; display: inline-block; }}
  .leg-dash {{ width: 16px; height: 0; border-top: 2px dashed; display: inline-block; }}
  .leg-box {{ width: 11px; height: 11px; display: inline-block; border-radius: 2px; }}

  .canvas-box {{ position: relative; width: 100%; border: 1px solid var(--border); border-radius: 4px; background: #080C10; overflow: hidden; }}
  canvas.chart-canvas {{ display: block; width: 100%; height: 100%; cursor: crosshair; }}
  .tooltip-box {{
    position: absolute;
    pointer-events: none;
    display: none;
    background: #0B0F14EB;
    border: 1px solid var(--border-strong);
    border-radius: 4px;
    padding: 7px 9px;
    font-size: 11px;
    font-family: var(--mono);
    color: var(--text);
    z-index: 20;
    min-width: 190px;
    line-height: 1.45;
  }}

  .anom-pills {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
  .anom-pill {{
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    border: 1px solid var(--border-strong);
    background: var(--elevated);
    color: var(--text);
    cursor: pointer;
    font-family: var(--mono);
  }}
  .anom-pill.high {{ border-color: #EF444477; color: #FCA5A5; }}
  .anom-pill.medium {{ border-color: #F59E0B77; color: #FDE68A; }}
  .anom-pill:hover {{ border-color: var(--cyan); }}

  .seg-metrics-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    padding: 8px 10px;
    background: var(--elevated);
    border: 1px solid var(--border);
    border-radius: 4px;
    margin-bottom: 10px;
    font-size: 12px;
  }}

  /* Anomaly Table */
  .table-wrap {{ overflow-x: auto; border: 1px solid var(--border); border-radius: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }}
  th {{ background: var(--elevated); color: var(--text-dim); font-weight: 600; white-space: nowrap; }}
  tbody tr {{ cursor: pointer; }}
  tbody tr:hover {{ background: #18212D99; }}

  /* Action Plan Grid */
  .plan-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
    gap: 14px;
  }}
  @media (max-width: 520px) {{ .plan-grid {{ grid-template-columns: 1fr; }} }}
  .guide-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--cyan);
    border-radius: 6px;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .plan-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--amber);
    border-radius: 6px;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .plan-card.clean {{ border-left-color: var(--green); }}
  .guide-head {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap; }}
  .guide-title {{ font-size: 14px; font-weight: 700; color: var(--text); }}
  .kv-line {{ font-size: 12px; color: var(--text-dim); }}
  .kv-line strong {{ color: var(--text); font-weight: 600; }}
  .cmd-box {{
    margin-top: 4px;
    padding: 8px 10px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--cyan);
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-all;
  }}
  .approval-notice {{
    margin-bottom: 12px;
    padding: 10px 14px;
    border-radius: 5px;
    border: 1px solid #F59E0B55;
    background: #78350F22;
    font-size: 12px;
    color: #FDE68A;
  }}
  .empty-inline {{ color: var(--text-dim); font-size: 13px; padding: 12px 0; }}
  .route-row-active {{ background: #0C4A6E33; outline: 1px solid #0EA5E966; }}
  .route-status-bar {{
    display: none;
    margin-bottom: 10px;
    padding: 9px 12px;
    border-radius: 5px;
    border: 1px solid #0EA5E966;
    background: #0C4A6E33;
    color: var(--text);
    font-size: 12px;
  }}
  .route-status-bar.warn {{ border-color: #F59E0B66; background: #78350F33; color: #FDE68A; }}
  .route-status-bar.err {{ border-color: #EF444466; background: #7F1D1D33; color: #FCA5A5; }}
</style>
</head>
<body>
<header class="topbar">
  <div class="brand-wrap">
    <span class="brand-title">{html.escape(report_title)}</span>
    <span class="badge badge-cyan">기준선 ver-1.0</span>
    <span id="topbar-badges" style="display:inline-flex;gap:6px;flex-wrap:wrap;"></span>
  </div>
  <nav class="nav-pills" aria-label="리포트 섹션 이동">
    <a class="nav-pill" href="#sec-routes">주행로그 그룹</a>
    <a class="nav-pill" href="#sec-summary">종합 요약</a>
    <a class="nav-pill" href="#sec-long">종방향 분석 (<span id="nav-long-cnt">0</span>)</a>
    <a class="nav-pill" href="#sec-lat">횡방향 분석 (<span id="nav-lat-cnt">0</span>)</a>
    <a class="nav-pill" href="#sec-anomalies">이상치 탐색</a>
    <a class="nav-pill" href="#sec-plan">파라미터 · 개선계획</a>
  </nav>
</header>

<main class="container">
  <!-- Section 0: Route Group Selector -->
  <section id="sec-routes">
    <div class="sec-head">
      <h2 class="sec-title">0. 주행로그 그룹 선택 목록 (루트 단위)</h2>
      <span class="sec-sub">목록에서 원하는 주행로그 그룹을 클릭하면 단말 동기화·이상치 분석·시각 리포트를 바로 전환합니다.</span>
    </div>
    <div class="panel">
      <div class="chart-toolbar">
        <div class="toolbar-group">
          <span id="route-dev-badge" class="badge badge-cyan">단말 확인 중</span>
          <span id="route-dongle-badge" class="badge">동글ID: -</span>
          <span id="route-count-badge" class="badge">0개 그룹</span>
        </div>
        <div class="toolbar-group">
          <input type="search" id="route-search" placeholder="루트명 또는 날짜 검색..." />
          <button type="button" class="btn" id="btn-wifi-discover">Wi-Fi 자동인식</button>
          <button type="button" class="btn" id="btn-toggle-dongle-cfg">동글ID 수동설정</button>
          <button type="button" class="btn" id="btn-routes-refresh">목록 새로고침</button>
        </div>
      </div>
      <div id="dongle-config-panel" style="display:none; margin-bottom:10px; padding:10px 12px; border:1px solid var(--border-strong); border-radius:5px; background:var(--elevated); gap:8px; flex-wrap:wrap; align-items:center;">
        <span class="muted" style="font-size:12px;">동글ID:</span>
        <input type="search" id="input-dongle-id" class="mono" style="width:170px;" placeholder="예: 0123456789abcdef" />
        <span class="muted" style="font-size:12px;">단말 IP(선택):</span>
        <input type="search" id="input-host-ip" class="mono" style="width:140px;" placeholder="예: 192.168.0.25" />
        <button type="button" class="btn active-btn" id="btn-save-dongle">저장 · 연결</button>
        <button type="button" class="btn" id="btn-reset-dongle">자동인식으로 초기화</button>
      </div>
      <div id="route-job-status" class="route-status-bar"></div>
      <div class="table-wrap" style="max-height: 310px; overflow-y: auto;">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>주행로그 그룹</th>
              <th>주행일시</th>
              <th>시간</th>
              <th>세그먼트(로컬/단말)</th>
              <th>주행리포트</th>
              <th>상태</th>
              <th>선택</th>
            </tr>
          </thead>
          <tbody id="route-groups-tbody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- Section 1: Executive Visual Summary -->
  <section id="sec-summary">
    <div class="sec-head">
      <h2 class="sec-title">1. 전체 주행 분석 종합 요약</h2>
      <span class="sec-sub">세그먼트 히트맵 타일을 클릭하면 아래 종방향/횡방향 상세 차트가 즉시 해당 구간으로 동기화됩니다.</span>
    </div>

    <div class="summary-grid">
      <div class="panel">
        <div class="card-kicker">종방향 · 횡방향 상태 게이지</div>
        <div class="gauges-row" id="gauges-row"></div>
        <div class="kpi-strip" id="kpi-strip"></div>
      </div>

      <div id="drive-report-card" class="panel report-match-card"></div>

      <div class="panel">
        <div class="card-kicker">카테고리별 이상치 분포 (총 <span id="anom-total-span">0</span>건)</div>
        <div id="anomaly-bars"></div>
      </div>
    </div>

    <div class="panel heatmap-wrap">
      <div class="card-kicker">세그먼트 타임라인 히트맵 (클릭하여 상세 차트 선택)</div>
      <div class="heatmap-strip" id="segment-heatmap"></div>
    </div>
    <div id="headline-notes-wrap"></div>
  </section>

  <!-- Section 2: Longitudinal Analysis -->
  <section id="sec-long">
    <div class="sec-head">
      <h2 class="sec-title">2. 종방향 (가감속 · 차간거리) 분석</h2>
      <span class="sec-sub">전체 세그먼트 속도/가감속 범위 요약 및 세그먼트별 2Hz 정밀 시계열 (a_ego / a_base / a_target)</span>
    </div>

    <div class="panel" style="margin-bottom: 14px;">
      <div class="chart-toolbar">
        <span class="card-kicker" style="margin:0;">전체 세그먼트 종방향 개요 (최대/최소 가속도 m/s² 및 속도 범위 · 막대 클릭 시 아래 상세 이동)</span>
        <div class="legend-row" style="margin:0;">
          <span class="leg-item"><span class="leg-box" style="background:#38BDF8"></span>최대 속도</span>
          <span class="leg-item"><span class="leg-box" style="background:#EF4444"></span>최소 감속 (a_ego_min)</span>
          <span class="leg-item"><span class="leg-box" style="background:#22C55E"></span>최대 가속 (a_ego_max)</span>
        </div>
      </div>
      <div class="canvas-box" style="height: 165px;">
        <canvas id="canvas-long-overview" class="chart-canvas" data-height="165"></canvas>
        <div id="tip-long-overview" class="tooltip-box"></div>
      </div>
    </div>

    <div class="panel">
      <div class="chart-toolbar">
        <div class="toolbar-group">
          <label for="sel-long-seg" class="muted" style="font-size:12px;">세그먼트:</label>
          <select id="sel-long-seg"></select>
          <button type="button" class="btn" id="btn-long-prev">이전</button>
          <button type="button" class="btn" id="btn-long-next">다음</button>
        </div>
        <div class="legend-row" style="margin:0;">
          <span class="leg-item"><span class="leg-line" style="background:#38BDF8"></span>차량속도 / a_ego</span>
          <span class="leg-item"><span class="leg-dash" style="border-color:#94A3B8"></span>a_base / 선행거리(m)</span>
          <span class="leg-item"><span class="leg-line" style="background:#F59E0B"></span>a_target</span>
          <span class="leg-item"><span class="leg-box" style="background:rgba(34,197,94,0.35)"></span>가속페달</span>
          <span class="leg-item"><span class="leg-box" style="background:rgba(239,68,68,0.45)"></span>브레이크 / 이상치</span>
        </div>
      </div>
      <div id="long-seg-meta" class="seg-metrics-bar"></div>
      <div class="canvas-box" style="height: 270px;">
        <canvas id="canvas-long-detail" class="chart-canvas" data-height="270"></canvas>
        <div id="tip-long-detail" class="tooltip-box"></div>
      </div>
      <div id="long-anom-pills" class="anom-pills"></div>
    </div>
  </section>

  <!-- Section 3: Lateral Analysis -->
  <section id="sec-lat">
    <div class="sec-head">
      <h2 class="sec-title">3. 횡방향 (조향 · 코너링 · 차선유지) 분석</h2>
      <span class="sec-sub">세그먼트별 최대 횡가속 · 코너 조향각속도 RMS · 토크 포화 프레임 및 3단 조향 텔레메트리</span>
    </div>

    <div class="panel" style="margin-bottom: 14px;">
      <div class="chart-toolbar">
        <span class="card-kicker" style="margin:0;">전체 세그먼트 횡방향 개요 (최대 횡가속 m/s², 코너 조향 RMS °/s, 포화 프레임)</span>
        <div class="legend-row" style="margin:0;">
          <span class="leg-item"><span class="leg-box" style="background:#38BDF8"></span>최대 횡가속 (m/s²)</span>
          <span class="leg-item"><span class="leg-box" style="background:#F59E0B"></span>코너 조향 RMS (°/s)</span>
          <span class="leg-item"><span class="leg-box" style="background:#EF4444"></span>토크 포화 프레임</span>
        </div>
      </div>
      <div class="canvas-box" style="height: 165px;">
        <canvas id="canvas-lat-overview" class="chart-canvas" data-height="165"></canvas>
        <div id="tip-lat-overview" class="tooltip-box"></div>
      </div>
    </div>

    <div class="panel">
      <div class="chart-toolbar">
        <div class="toolbar-group">
          <label for="sel-lat-seg" class="muted" style="font-size:12px;">세그먼트:</label>
          <select id="sel-lat-seg"></select>
          <button type="button" class="btn" id="btn-lat-prev">이전</button>
          <button type="button" class="btn" id="btn-lat-next">다음</button>
        </div>
        <div class="legend-row" style="margin:0;">
          <span class="leg-item"><span class="leg-line" style="background:#38BDF8"></span>실제 횡가속 / 차선편차</span>
          <span class="leg-item"><span class="leg-dash" style="border-color:#F59E0B"></span>목표 횡가속</span>
          <span class="leg-item"><span class="leg-line" style="background:#22C55E"></span>실제 조향각 (°)</span>
          <span class="leg-item"><span class="leg-dash" style="border-color:#CBD5E1"></span>목표 조향각 (°)</span>
          <span class="leg-item"><span class="leg-box" style="background:rgba(239,68,68,0.45)"></span>토크 포화 / 수동조향</span>
        </div>
      </div>
      <div id="lat-seg-meta" class="seg-metrics-bar"></div>
      <div class="canvas-box" style="height: 320px;">
        <canvas id="canvas-lat-detail" class="chart-canvas" data-height="320"></canvas>
        <div id="tip-lat-detail" class="tooltip-box"></div>
      </div>
      <div id="lat-anom-pills" class="anom-pills"></div>
    </div>
  </section>

  <!-- Section 4: Filterable Anomaly Explorer -->
  <section id="sec-anomalies">
    <div class="sec-head">
      <h2 class="sec-title">4. 이상치 상세 탐색기</h2>
      <span class="sec-sub">행을 클릭하면 해당 세그먼트 차트로 이동하고 문제 타임스탬프 구간을 빨간 커서로 강조합니다.</span>
    </div>
    <div class="panel">
      <div class="chart-toolbar">
        <div class="toolbar-group" id="axis-filter-btns">
          <button type="button" class="btn active-btn" data-axis="all">전체</button>
          <button type="button" class="btn" data-axis="long">종방향 ({payload["stats"]["long_count"]})</button>
          <button type="button" class="btn" data-axis="lat">횡방향 ({payload["stats"]["lat_count"]})</button>
        </div>
        <div class="toolbar-group">
          <select id="filter-sev">
            <option value="all">심각도: 전체</option>
            <option value="high">심각도: 높음</option>
            <option value="medium">심각도: 보통</option>
          </select>
          <select id="filter-cat">
            <option value="all">카테고리: 전체</option>
          </select>
          <input type="search" id="filter-q" placeholder="세그먼트 또는 내용 검색..." />
          <span id="filter-count" class="badge"></span>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>축</th>
              <th>심각도</th>
              <th>카테고리</th>
              <th>세그먼트</th>
              <th>시간대</th>
              <th>속도</th>
              <th>요약 및 핵심 수치</th>
            </tr>
          </thead>
          <tbody id="anomaly-tbody"></tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- Section 5: Action Plan (Parameters + ver-1.0 Regression-Safe Code Plan) -->
  <section id="sec-plan">
    <div class="sec-head">
      <h2 class="sec-title">5. 해결 방안: 파라미터 튜닝 가이드 및 ver-1.0 무회귀 개발 계획</h2>
      <span class="sec-sub">파라미터 조정으로 해결 가능한 항목과 승인 전까지 코드 수정 없이 제안되는 고립 개선 계획</span>
    </div>

    <div class="approval-notice">
      <strong>ver-1.0 보존 원칙:</strong> 현재 리포트 생성 단계에서는 <code>openpilot/selfdrive/</code> 런타임 코드를 일절 수정하지 않습니다.
      파라미터 조정을 우선 검토하시고, 코드 수정이 필요한 계획은 사용자가 명시적으로 승인한 뒤 <code>ver-1.0</code> A/B 회귀 게이트를 통과한 경우에만 반영합니다.
    </div>

    <div class="panel" style="margin-bottom: 14px;">
      <div class="card-kicker">1차 권장: 주행 파라미터 튜닝 가이드</div>
      <div id="param-guides-grid" class="plan-grid"></div>
    </div>

    <div class="panel">
      <div class="card-kicker">2차 검토: 추가 기능 개선 / ver-1.0 무회귀 코드 개발 계획</div>
      <div id="code-plans-grid" class="plan-grid"></div>
    </div>
  </section>
</main>

<script id="report-data" type="application/json">{payload_json}</script>
<script>
(() => {{
  "use strict";
  let DATA = JSON.parse(document.getElementById("report-data").textContent);
  let SEGMENTS = DATA.segments || [];
  let ANOMALIES = DATA.anomalies || [];
  let CAT_META = DATA.category_meta || {{}};
  const IS_LIVE_API = Boolean(DATA.live_api || (window.location && window.location.protocol && window.location.protocol.startsWith("http")));

  let selectedSegIdx = Math.max(0, SEGMENTS.findIndex(s => (s.anomalies || []).length > 0));
  if (selectedSegIdx < 0 && SEGMENTS.length > 0) selectedSegIdx = 0;
  let focusWindow = null; // {{ segName, tStart, tEnd }}
  let axisFilter = "all";
  let busyRoute = "";

  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, m => ({{ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }}[m]));
  const shortSeg = (name) => {{
    const m = String(name || "").match(/--(\\d+)$/);
    return m ? `seg ${{m[1]}}` : String(name || "-").slice(-10);
  }};
  const fmtWallClock = (seg, tSec) => {{
    if (!seg || !seg.started_at_ts) return "";
    const d = new Date((Number(seg.started_at_ts) + (Number(tSec) - Number(seg.t_offset_s || 0))) * 1000);
    if (isNaN(d.getTime())) return "";
    const p2 = (n) => String(n).padStart(2, "0");
    return `${{p2(d.getHours())}}:${{p2(d.getMinutes())}}:${{p2(d.getSeconds())}}`;
  }};
  const segTimeRange = (seg) => {{
    if (!seg || !seg.started_at) return "-";
    return seg.ended_at ? `${{seg.started_at}} ~ ${{seg.ended_at}}` : seg.started_at;
  }};

  // 1) Category Breakdown Bars
  function renderAnomalyBars() {{
    const host = document.getElementById("anomaly-bars");
    const counts = {{}};
    Object.keys(CAT_META).forEach(k => counts[k] = {{ total: 0, high: 0 }});
    ANOMALIES.forEach(a => {{
      if (!counts[a.category]) counts[a.category] = {{ total: 0, high: 0 }};
      counts[a.category].total += 1;
      if (a.severity === "high") counts[a.category].high += 1;
    }});
    const maxVal = Math.max(1, ...Object.values(counts).map(c => c.total));
    const ordered = Object.keys(CAT_META).sort((a, b) => counts[b].total - counts[a].total);
    host.innerHTML = ordered.map(cat => {{
      const meta = CAT_META[cat] || {{ label: cat, axis: "long" }};
      const c = counts[cat];
      const w = (100 * c.total / maxVal).toFixed(1);
      const cls = c.high > 0 ? "bar-high" : (meta.axis === "lat" ? "bar-lat" : "bar-long");
      return `<div class="bar-row" style="opacity:${{c.total ? 1 : 0.45}}">
        <span>${{esc(meta.label)}}</span>
        <div class="bar-track"><div class="bar-fill ${{cls}}" style="width:${{w}}%"></div></div>
        <span class="mono" style="text-align:right">${{c.total}}</span>
      </div>`;
    }}).join("");
  }}

  // 2) Segment Heatmap Strip
  function renderHeatmap() {{
    const host = document.getElementById("segment-heatmap");
    if (!SEGMENTS.length) {{
      host.innerHTML = '<div class="empty-inline">스캔된 세그먼트가 없습니다.</div>';
      return;
    }}
    host.innerHTML = SEGMENTS.map((s, idx) => {{
      const anoms = s.anomalies || [];
      const hasHigh = anoms.some(a => a.severity === "high");
      const cls = s.error ? "sev-err" : (hasHigh ? "sev-high" : (anoms.length ? "sev-med" : ""));
      const activeCls = idx === selectedSegIdx ? "active" : "";
      const vTxt = s.error ? "ERR" : `${{Math.round(s.v_max_kph || 0)}}km/h`;
      const hmTime = s.started_at ? s.started_at.slice(11, 19) : "";
      return `<button type="button" class="seg-tile ${{cls}} ${{activeCls}}" data-seg-idx="${{idx}}" title="${{esc(s.segment)}} | ${{esc(segTimeRange(s))}}">
        <div class="seg-tile-id">${{esc(shortSeg(s.segment))}}</div>
        ${{hmTime ? `<div class="seg-tile-meta mono">${{esc(hmTime)}}</div>` : ""}}
        <div class="seg-tile-meta">${{vTxt}} · ${{anoms.length}}건</div>
      </button>`;
    }}).join("");
    host.querySelectorAll(".seg-tile").forEach(btn => {{
      btn.addEventListener("click", () => {{
        focusWindow = null;
        selectSegment(Number(btn.dataset.segIdx));
      }});
    }});
  }}

  // Populate segment selectors (listeners bound once in bindStaticListenersOnce)
  function populateSelectors() {{
    const opts = SEGMENTS.map((s, idx) => {{
      const n = (s.anomalies || []).length;
      const err = s.error ? " [오류]" : "";
      const dt = s.started_at ? ` · ${{segTimeRange(s)}}` : "";
      return `<option value="${{idx}}">${{esc(shortSeg(s.segment))}}${{esc(dt)}} (${{esc(s.segment)}}) · ${{Math.round(s.duration_s || 0)}}s · 이상치 ${{n}}건${{err}}</option>`;
    }}).join("");
    ["sel-long-seg", "sel-lat-seg"].forEach(id => {{
      const el = document.getElementById(id);
      el.innerHTML = opts || "<option value='-1'>세그먼트 없음</option>";
      el.value = String(SEGMENTS.length ? selectedSegIdx : -1);
    }});
  }}

  function setRouteStatus(msg, level) {{
    const bar = document.getElementById("route-job-status");
    if (!msg) {{ bar.style.display = "none"; bar.textContent = ""; return; }}
    bar.style.display = "block";
    bar.className = "route-status-bar" + (level ? ` ${{level}}` : "");
    bar.textContent = msg;
  }}

  function renderHeaderAndSummary() {{
    const st = DATA.stats || {{}};
    const sc = DATA.scores || {{ overall: 100, long: 100, lat: 100 }};
    const highCount = Number(st.high_count || 0);
    const totalAnoms = Number(st.anomaly_count || 0);
    const sevCls = highCount ? "badge-red" : (totalAnoms ? "badge-amber" : "badge-green");
    document.getElementById("topbar-badges").innerHTML = `
      ${{DATA.active_route ? `<span class="badge badge-cyan">그룹 ${{esc(DATA.active_route)}}</span>` : '<span class="badge">그룹 미선택</span>'}}
      <span class="badge">커밋 ${{esc(st.commit || "-")}}</span>
      <span class="badge">세그먼트 ${{st.segment_count || 0}}개 · ${{(Number(st.total_duration_s || 0) / 60).toFixed(1)}}분</span>
      <span class="badge ${{sevCls}}">이상치 ${{totalAnoms}}건 (높음 ${{highCount}})</span>
      <span class="badge muted">${{esc(DATA.generated_at || "")}}</span>
    `;
    document.getElementById("nav-long-cnt").textContent = String(st.long_count || 0);
    document.getElementById("nav-lat-cnt").textContent = String(st.lat_count || 0);
    document.getElementById("anom-total-span").textContent = String(totalAnoms);

    const axisLongBtn = document.querySelector('#axis-filter-btns button[data-axis="long"]');
    const axisLatBtn = document.querySelector('#axis-filter-btns button[data-axis="lat"]');
    if (axisLongBtn) axisLongBtn.textContent = `종방향 (${{st.long_count || 0}})`;
    if (axisLatBtn) axisLatBtn.textContent = `횡방향 (${{st.lat_count || 0}})`;

    const gaugeSvg = (score, label, sub) => {{
      const r = 38, circ = 2 * Math.PI * r;
      const dash = circ * Math.max(0, Math.min(1, Number(score) / 100));
      const color = score >= 85 ? "#22C55E" : (score >= 70 ? "#F59E0B" : "#EF4444");
      return `<div class="gauge-card">
        <svg width="96" height="96" viewBox="0 0 96 96" aria-hidden="true">
          <circle cx="48" cy="48" r="38" fill="none" stroke="#1E293B" stroke-width="8"/>
          <circle cx="48" cy="48" r="38" fill="none" stroke="${{color}}" stroke-width="8"
            stroke-dasharray="${{dash.toFixed(1)}} ${{circ.toFixed(1)}}" stroke-linecap="round" transform="rotate(-90 48 48)"/>
          <text x="48" y="47" text-anchor="middle" fill="#E2E8F0" font-size="20" font-weight="700"
            font-family="JetBrains Mono, ui-monospace, monospace">${{Math.round(score)}}</text>
          <text x="48" y="63" text-anchor="middle" fill="#94A3B8" font-size="10">/ 100</text>
        </svg>
        <div class="gauge-meta"><div class="gauge-title">${{esc(label)}}</div><div class="gauge-sub">${{esc(sub)}}</div></div>
      </div>`;
    }};

    if (!SEGMENTS.length) {{
      document.getElementById("gauges-row").innerHTML = '<div class="empty-inline">위 목록에서 주행로그 그룹을 선택하면 종·횡방향 분석 게이지가 표시됩니다.</div>';
      document.getElementById("kpi-strip").innerHTML = "";
    }} else {{
      document.getElementById("gauges-row").innerHTML =
        gaugeSvg(sc.long, "종방향 (가감속)", `감지 ${{st.long_count || 0}}건 · 활성 ${{st.long_active_pct || 0}}%`) +
        gaugeSvg(sc.lat, "횡방향 (조향)", `감지 ${{st.lat_count || 0}}건 · 활성 ${{st.lat_active_pct || 0}}%`);
      document.getElementById("kpi-strip").innerHTML = `
        <div class="kpi-cell"><div class="kpi-v">${{st.segment_count || 0}}</div><div class="kpi-l">분석 세그먼트</div></div>
        <div class="kpi-cell"><div class="kpi-v">${{(Number(st.total_duration_s || 0) / 60).toFixed(1)}}m</div><div class="kpi-l">총 주행시간</div></div>
        <div class="kpi-cell"><div class="kpi-v">${{Math.round(st.v_max_kph || 0)}}</div><div class="kpi-l">최고속도 (km/h)</div></div>
        <div class="kpi-cell"><div class="kpi-v">${{Number(st.lat_accel_max || 0).toFixed(2)}}</div><div class="kpi-l">최대횡가속 (m/s²)</div></div>
      `;
    }}

    const drCard = document.getElementById("drive-report-card");
    const drs = DATA.drive_reports || [];
    if (drs.length) {{
      const dr = drs[0];
      const hb = dr.hard_brakes || {{}};
      const autoHb = dr.auto_hard_brake_count ?? hb.auto_count ?? dr.hard_brake_count ?? 0;
      const manHb = dr.manual_hard_brake_count ?? hb.manual_count ?? 0;
      const deds = Array.isArray(dr.deductions) ? dr.deductions : [];
      const chips = deds.slice(0, 4).map(d => {{
        const lbl = d.reason || d.label || d.key || "감점";
        const pts = d.points ?? d.penalty ?? "";
        const ptsTxt = String(pts).startsWith("-") ? String(pts) : `-${{pts}}`;
        return `<span class="chip chip-warn">${{esc(lbl)}} ${{esc(ptsTxt)}}</span>`;
      }}).join("") || '<span class="chip chip-ok">추가 감점 항목 없음</span>';
      drCard.innerHTML = `
        <div class="card-kicker">매칭된 차량 주행리포트 (drive_reports)</div>
        <div class="report-match-main">
          <div class="grade-box">
            <span class="grade-num">${{esc(dr.score ?? "-")}}</span>
            <span class="grade-unit">점 (${{esc(dr.grade ?? "-")}})</span>
          </div>
          <div class="report-match-stats">
            <div><span class="muted">Report ID</span> <code class="mono">${{esc(dr.report_id || "-")}}</code></div>
            <div><span class="muted">주행 거리</span> <strong class="mono">${{Number(dr.distance_km || 0).toFixed(2)}} km</strong> ${{dr.time_range_label ? `· <span class="muted">${{esc(dr.time_range_label)}}</span>` : ""}}</div>
            <div><span class="muted">급정거(자동 / 수동)</span> <strong class="mono">${{autoHb}} / ${{manHb}}</strong></div>
          </div>
        </div>
        <div class="chip-row">${{chips}}</div>
      `;
    }} else {{
      drCard.innerHTML = `
        <div class="card-kicker">매칭된 차량 주행리포트 (drive_reports)</div>
        <div class="empty-inline">해당 시간대와 일치하는 주행리포트 JSON이 없거나 아직 분석 전입니다.</div>
      `;
    }}

    const notesWrap = document.getElementById("headline-notes-wrap");
    const notes = DATA.custom_notes || {{}};
    const sn = Array.isArray(notes.summary_notes) ? notes.summary_notes : [];
    if (notes.headline || sn.length) {{
      notesWrap.innerHTML = `<div class="panel notes-banner">
        <div class="card-kicker">핵심 진단 요약</div>
        ${{notes.headline ? `<div class="notes-headline">${{esc(notes.headline)}}</div>` : ""}}
        ${{sn.length ? `<ul class="notes-list">${{sn.map(x => `<li>${{esc(x)}}</li>`).join("")}}</ul>` : ""}}
      </div>`;
    }} else {{
      notesWrap.innerHTML = "";
    }}
  }}

  function renderRouteGroups() {{
    const devBadge = document.getElementById("route-dev-badge");
    if (DATA.device_label) {{
      devBadge.className = "badge badge-green";
      devBadge.textContent = `단말 연결됨: ${{DATA.device_label}}`;
    }} else {{
      devBadge.className = "badge badge-amber";
      devBadge.textContent = DATA.device_error ? "단말 미연결 (로컬 캐시)" : "로컬 캐시 모드";
    }}
    const dongleBadge = document.getElementById("route-dongle-badge");
    dongleBadge.textContent = DATA.device_dongle_id ? `동글ID: ${{DATA.device_dongle_id}}` : "동글ID: 자동인식 대기";
    const inDongle = document.getElementById("input-dongle-id");
    const inHost = document.getElementById("input-host-ip");
    if (inDongle && document.activeElement !== inDongle) inDongle.value = DATA.device_dongle_id || "";
    if (inHost && document.activeElement !== inHost && DATA.device_label) inHost.value = String(DATA.device_label).split(":")[0];

    const groups = DATA.route_groups || [];
    const q = (document.getElementById("route-search").value || "").trim().toLowerCase();
    const filtered = groups.filter(g => {{
      if (!q) return true;
      return String(g.route || "").toLowerCase().includes(q)
        || String(g.started_at || "").toLowerCase().includes(q)
        || String(g.latest_label || "").toLowerCase().includes(q);
    }});
    document.getElementById("route-count-badge").textContent = `${{filtered.length}} / ${{groups.length}}개 그룹`;

    const tbody = document.getElementById("route-groups-tbody");
    if (!filtered.length) {{
      tbody.innerHTML = '<tr><td colspan="8" class="muted">표시할 주행로그 그룹이 없습니다.</td></tr>';
      return;
    }}
    tbody.innerHTML = filtered.map((g, idx) => {{
      const isAct = g.route && g.route === DATA.active_route;
      const isBusy = g.route && g.route === busyRoute;
      const dt = (g.started_at && g.ended_at) ? `${{g.started_at}}~${{g.ended_at}}` : (g.started_at || "-");
      const dur = Number(g.duration_min || 0) > 0 ? `${{Math.round(g.duration_min)}}분` : "-";
      const segs = `${{g.local_segments || 0}} / ${{g.device_segments || "-"}}`;
      const rep = g.matched_report;
      const repTxt = rep ? `${{rep.score ?? "-"}}점(${{rep.grade ?? "-"}}) ${{Number(rep.distance_km || 0).toFixed(1)}}km` : "-";
      const stTxt = g.analyzed_html ? '<span class="badge badge-green">분석완료</span>'
        : ((g.local_segments || 0) > 0 ? '<span class="badge badge-cyan">로컬보관</span>' : '<span class="badge">단말보관</span>');
      const primaryLabel = isBusy ? "진행중..." : (g.analyzed_html ? (isAct ? "현재 보기" : "리포트 보기") : "동기화·분석");
      const rescanBtn = g.analyzed_html
        ? ` <button type="button" class="btn btn-rescan" data-route="${{esc(g.route)}}" ${{isBusy ? "disabled" : ""}}>재분석</button>`
        : "";
      return `<tr class="${{isAct ? "route-row-active" : ""}}" data-route-row="${{esc(g.route)}}">
        <td class="mono">${{idx + 1}}</td>
        <td class="mono"><strong>${{esc(g.route)}}</strong></td>
        <td class="mono">${{esc(dt)}}${{g.latest_label ? ` <span class="muted">(${{esc(g.latest_label)}})</span>` : ""}}</td>
        <td class="mono">${{dur}}</td>
        <td class="mono">${{segs}}</td>
        <td>${{esc(repTxt)}}</td>
        <td>${{stTxt}}</td>
        <td style="white-space:nowrap;">
          <button type="button" class="btn ${{isAct ? "active-btn" : ""}} btn-run-route" data-route="${{esc(g.route)}}" ${{isBusy ? "disabled" : ""}}>${{primaryLabel}}</button>${{rescanBtn}}
        </td>
      </tr>`;
    }}).join("");

    tbody.querySelectorAll("button.btn-run-route").forEach(btn => {{
      btn.addEventListener("click", (e) => {{
        e.stopPropagation();
        handleRouteSelect(btn.dataset.route, false);
      }});
    }});
    tbody.querySelectorAll("button.btn-rescan").forEach(btn => {{
      btn.addEventListener("click", (e) => {{
        e.stopPropagation();
        handleRouteSelect(btn.dataset.route, true);
      }});
    }});
    tbody.querySelectorAll("tr[data-route-row]").forEach(tr => {{
      tr.addEventListener("click", () => handleRouteSelect(tr.dataset.routeRow, false));
    }});
  }}

  async function refreshRouteGroupsFromApi() {{
    if (!IS_LIVE_API) {{
      setRouteStatus("정적 HTML 파일 모드입니다. 목록 실시간 새로고침/원격 동기화는 `PYTHONPATH=. uv run --no-sync python3 -m openpilot.tools.drive_analyzer serve --open` 실행 후 브라우저에서 이용해 주세요.", "warn");
      return;
    }}
    try {{
      const resp = await fetch("/api/routes");
      const data = await resp.json();
      if (!resp.ok || !data.ok) throw new Error(data.error || "목록 조회 실패");
      DATA.device_label = data.device || "";
      DATA.device_error = data.device_error || "";
      if (data.dongle_id !== undefined) DATA.device_dongle_id = data.dongle_id || "";
      DATA.route_groups = data.groups || [];
      renderRouteGroups();
      setRouteStatus("주행로그 그룹 목록을 최신 상태로 갱신했습니다.", "");
    }} catch (err) {{
      setRouteStatus(`목록 새로고침 오류: ${{err.message}}`, "err");
    }}
  }}

  async function discoverWifiDevice() {{
    if (!IS_LIVE_API) {{
      setRouteStatus("정적 HTML 모드에서는 Wi-Fi 자동인식 API가 비활성입니다. CLI `discover --any` 또는 `serve --open`을 실행하세요.", "warn");
      return;
    }}
    setRouteStatus("동일 Wi-Fi에서 오파 단말 IP 및 동글ID를 자동 탐색 중...", "");
    try {{
      const resp = await fetch("/api/discover", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ any_device: true }}),
      }});
      const data = await resp.json();
      if (!resp.ok || !data.ok) throw new Error(data.error || "자동 인식 실패");
      DATA.device_label = data.device_label || "";
      DATA.device_dongle_id = data.dongle_id || "";
      await refreshRouteGroupsFromApi();
      setRouteStatus(data.message || "Wi-Fi 자동 인식을 완료했습니다.", "");
    }} catch (err) {{
      setRouteStatus(`자동 인식 오류: ${{err.message}}`, "err");
    }}
  }}

  async function saveDongleConfig(clearMode) {{
    if (!IS_LIVE_API) {{
      setRouteStatus("정적 HTML 모드에서는 설정 저장이 비활성입니다. CLI `config --dongle-id <ID>` 또는 `serve --open`을 사용하세요.", "warn");
      return;
    }}
    const dongleId = clearMode ? "" : (document.getElementById("input-dongle-id").value || "").trim();
    const hostIp = clearMode ? "" : (document.getElementById("input-host-ip").value || "").trim();
    setRouteStatus(clearMode ? "동글ID 고정을 초기화하고 재탐색 중..." : "설정 저장 및 연결 확인 중...", "");
    try {{
      const resp = await fetch("/api/config", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ dongle_id: dongleId, host: hostIp, clear_dongle_id: Boolean(clearMode) }}),
      }});
      const data = await resp.json();
      if (!resp.ok || !data.ok) throw new Error(data.error || "설정 저장 실패");
      DATA.device_dongle_id = (data.config && data.config.dongle_id) || "";
      if (clearMode) {{
        await discoverWifiDevice();
      }} else {{
        await refreshRouteGroupsFromApi();
        setRouteStatus(data.message || "동글ID 설정을 저장했습니다.", "");
      }}
    }} catch (err) {{
      setRouteStatus(`설정 오류: ${{err.message}}`, "err");
    }}
  }}

  async function handleRouteSelect(routeName, forceRescan) {{
    if (!routeName || busyRoute) return;
    const g = (DATA.route_groups || []).find(x => x.route === routeName);
    if (!IS_LIVE_API) {{
      if (g && g.analyzed_html && !forceRescan) {{
        const safe = routeName.replace(/[^A-Za-z0-9._-]+/g, "_");
        window.location.href = `../route_${{safe}}/analysis_report.html`;
        return;
      }}
      setRouteStatus(
        `정적 파일 모드에서는 선택 다운로드/분석 API가 비활성입니다. 스튜디오 서버(python3 -m openpilot.tools.drive_analyzer serve --open) 또는 CLI(analyze --route ${{routeName}})를 실행하세요.`,
        "warn"
      );
      return;
    }}

    if (g && g.analyzed_html && !forceRescan) {{
      try {{
        setRouteStatus(`${{routeName}} 분석 리포트를 불러오는 중...`, "");
        const resp = await fetch(`/api/report/${{encodeURIComponent(routeName)}}`);
        if (resp.ok) {{
          const payload = await resp.json();
          applyReportPayload(payload);
          setRouteStatus(`${{routeName}} 리포트를 표시 중입니다.`, "");
          document.getElementById("sec-summary").scrollIntoView({{ behavior: "smooth" }});
          return;
        }}
      }} catch (_e) {{ /* fall through to analyze */ }}
    }}

    busyRoute = routeName;
    renderRouteGroups();
    setRouteStatus(`${{routeName}} 단말 동기화 및 분석 요청 중...`, "");
    try {{
      const startResp = await fetch("/api/analyze", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{ route: routeName, force_rescan: Boolean(forceRescan) }}),
      }});
      const startData = await startResp.json();
      if (!startResp.ok || !startData.ok) throw new Error(startData.error || "분석 시작 실패");
      const jobId = startData.job_id;
      while (true) {{
        await new Promise(r => setTimeout(r, 600));
        const jResp = await fetch(`/api/jobs/${{encodeURIComponent(jobId)}}`);
        const jData = await jResp.json();
        const job = jData.job || {{}};
        setRouteStatus(job.message || `${{routeName}} 처리 중 (${{job.status || ""}})...`, "");
        if (job.status === "done") {{
          const rResp = await fetch(`/api/report/${{encodeURIComponent(routeName)}}`);
          if (!rResp.ok) throw new Error("완료된 리포트 로드 실패");
          const payload = await rResp.json();
          applyReportPayload(payload);
          setRouteStatus(`${{routeName}} 분석이 완료되어 아래에 반영되었습니다.`, "");
          document.getElementById("sec-summary").scrollIntoView({{ behavior: "smooth" }});
          break;
        }}
        if (job.status === "error") {{
          throw new Error(job.error || job.message || "분석 실패");
        }}
      }}
    }} catch (err) {{
      setRouteStatus(`오류 (${{routeName}}): ${{err.message}}`, "err");
    }} finally {{
      busyRoute = "";
      renderRouteGroups();
    }}
  }}

  function applyReportPayload(nextPayload) {{
    DATA = nextPayload;
    SEGMENTS = DATA.segments || [];
    ANOMALIES = DATA.anomalies || [];
    CAT_META = DATA.category_meta || CAT_META;
    selectedSegIdx = Math.max(0, SEGMENTS.findIndex(s => (s.anomalies || []).length > 0));
    if (selectedSegIdx < 0 && SEGMENTS.length > 0) selectedSegIdx = 0;
    focusWindow = null;
    renderAll();
  }}

  function stepSegment(delta) {{
    if (!SEGMENTS.length) return;
    focusWindow = null;
    const next = (selectedSegIdx + delta + SEGMENTS.length) % SEGMENTS.length;
    selectSegment(next);
  }}

  function selectSegment(idx) {{
    if (idx < 0 || idx >= SEGMENTS.length) return;
    selectedSegIdx = idx;
    document.getElementById("sel-long-seg").value = String(idx);
    document.getElementById("sel-lat-seg").value = String(idx);
    renderHeatmap();
    drawLongOverview();
    drawLatOverview();
    drawLongDetail();
    drawLatDetail();
  }}

  // HiDPI canvas setup (use immutable data-height + locked CSS height so Retina dpr never grows layout)
  function setupCanvas(canvas) {{
    const dpr = window.devicePixelRatio || 1;
    const parentW = canvas.parentElement ? canvas.parentElement.clientWidth : 800;
    const w = Math.max(320, Math.floor(parentW || 800));
    const h = Number(canvas.dataset.height) || 240;
    const pw = Math.round(w * dpr);
    const ph = Math.round(h * dpr);
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";
    if (canvas.width !== pw) canvas.width = pw;
    if (canvas.height !== ph) canvas.height = ph;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return {{ ctx, w, h }};
  }}

  function showTooltip(tipEl, canvas, clientX, clientY, htmlContent) {{
    const rect = canvas.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    tipEl.innerHTML = htmlContent;
    tipEl.style.display = "block";
    const tw = tipEl.offsetWidth || 200;
    const th = tipEl.offsetHeight || 90;
    const left = (x + 14 + tw > rect.width) ? Math.max(6, x - tw - 12) : (x + 14);
    const top = Math.max(6, Math.min(rect.height - th - 6, y - 18));
    tipEl.style.left = left + "px";
    tipEl.style.top = top + "px";
  }}

  // 3) Longitudinal Overview Canvas
  const longOvHits = [];
  function drawLongOverview() {{
    const canvas = document.getElementById("canvas-long-overview");
    const {{ ctx, w, h }} = setupCanvas(canvas);
    longOvHits.length = 0;
    if (!SEGMENTS.length) return;
    const pad = {{ l: 44, r: 44, t: 18, b: 28 }};
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    // Grid for acceleration range [-4, +3]
    const aMin = -4.0, aMax = 3.0;
    const yAcc = (v) => pad.t + ph * (1 - (Math.max(aMin, Math.min(aMax, v)) - aMin) / (aMax - aMin));
    const yVel = (v) => pad.t + ph * (1 - Math.max(0, Math.min(140, v)) / 140);

    ctx.strokeStyle = "#1E293B";
    ctx.lineWidth = 1;
    ctx.font = "10px JetBrains Mono, monospace";
    [-4, -2, 0, 2].forEach(tick => {{
      const y = yAcc(tick);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.fillStyle = "#64748B";
      ctx.textAlign = "right";
      ctx.fillText(tick.toFixed(0), pad.l - 6, y + 3);
    }});
    [0, 50, 100, 130].forEach(vt => {{
      const y = yVel(vt);
      ctx.fillStyle = "#38BDF8AA";
      ctx.textAlign = "left";
      ctx.fillText(vt + "k", w - pad.r + 5, y + 3);
    }});

    const slot = pw / SEGMENTS.length;
    const bw = Math.max(6, Math.min(28, slot * 0.56));
    const zeroY = yAcc(0);

    SEGMENTS.forEach((s, i) => {{
      const cx = pad.l + slot * (i + 0.5);
      longOvHits.push({{ x0: pad.l + slot * i, x1: pad.l + slot * (i + 1), idx: i }});
      if (i === selectedSegIdx) {{
        ctx.fillStyle = "rgba(56, 189, 248, 0.12)";
        ctx.fillRect(pad.l + slot * i, pad.t, slot, ph);
      }}
      if (!s.error) {{
        // Max speed bar (background slate-cyan)
        const vy = yVel(s.v_max_kph || 0);
        ctx.fillStyle = "rgba(56, 189, 248, 0.25)";
        ctx.fillRect(cx - bw / 2, vy, bw, pad.t + ph - vy);
        // Accel positive/negative stems
        const yMaxA = yAcc(s.a_ego_max || 0);
        const yMinA = yAcc(s.a_ego_min || 0);
        ctx.fillStyle = "#22C55E";
        ctx.fillRect(cx - bw * 0.22, Math.min(yMaxA, zeroY), bw * 0.44, Math.abs(zeroY - yMaxA));
        ctx.fillStyle = (s.a_ego_min <= -2.5) ? "#EF4444" : "#F59E0B";
        ctx.fillRect(cx - bw * 0.22, zeroY, bw * 0.44, Math.max(2, yMinA - zeroY));
      }}
      ctx.fillStyle = i === selectedSegIdx ? "#38BDF8" : "#94A3B8";
      ctx.textAlign = "center";
      ctx.fillText(shortSeg(s.segment).replace("seg ", "s"), cx, h - 9);
    }});
  }}

  // 4) Longitudinal Detail Canvas (2 synced bands: Speed+Lead & Accel)
  let longHoverX = null;
  function drawLongDetail() {{
    const s = SEGMENTS[selectedSegIdx];
    const metaEl = document.getElementById("long-seg-meta");
    const pillsEl = document.getElementById("long-anom-pills");
    const canvas = document.getElementById("canvas-long-detail");
    const {{ ctx, w, h }} = setupCanvas(canvas);
    if (!s || s.error || !s.telemetry || !(s.telemetry.t || []).length) {{
      metaEl.innerHTML = `<span class="muted">표시할 종방향 시계열 데이터가 없습니다 (${{esc(s ? (s.error || s.segment) : "-")}}).</span>`;
      pillsEl.innerHTML = "";
      return;
    }}
    const tel = s.telemetry;
    const tArr = tel.t;
    const t0 = tArr[0], t1 = Math.max(t0 + 1.0, tArr[tArr.length - 1]);
    const longAnoms = (s.anomalies || []).filter(a => CAT_META[a.category]?.axis === "long");
    metaEl.innerHTML = `
      <span><strong>${{esc(s.segment)}}</strong></span>
      <span>발생일시 <strong class="mono">${{esc(segTimeRange(s))}}</strong></span>
      <span>길이 <strong class="mono">${{(s.duration_s || 0).toFixed(1)}}s</strong></span>
      <span>속도 <strong class="mono">${{(s.v_min_kph || 0).toFixed(0)}}..${{(s.v_max_kph || 0).toFixed(0)}} km/h</strong></span>
      <span>a_ego <strong class="mono">${{(s.a_ego_min || 0).toFixed(2)}} .. +${{(s.a_ego_max || 0).toFixed(2)}} m/s²</strong></span>
      <span>종방향 이상치 <strong class="mono">${{longAnoms.length}}건</strong></span>
    `;
    pillsEl.innerHTML = longAnoms.map(a =>
      `<button type="button" class="anom-pill ${{esc(a.severity)}}" data-t0="${{a.t_start}}" data-t1="${{a.t_end}}">` +
      `${{esc(CAT_META[a.category]?.label || a.category)}} t=${{a.t_start.toFixed(1)}}s (${{a.v_kph.toFixed(0)}}k)</button>`
    ).join("") || '<span class="muted" style="font-size:12px;">이 세그먼트에서 검출된 종방향 이상치가 없습니다.</span>';
    pillsEl.querySelectorAll(".anom-pill").forEach(btn => {{
      btn.addEventListener("click", () => {{
        focusWindow = {{ segName: s.segment, tStart: Number(btn.dataset.t0), tEnd: Number(btn.dataset.t1) }};
        drawLongDetail();
      }});
    }});

    const pad = {{ l: 48, r: 46, t: 14, b: 24 }};
    const pw = w - pad.l - pad.r;
    const gap = 16;
    const hTop = Math.floor((h - pad.t - pad.b - gap) * 0.44);
    const yTop0 = pad.t;
    const yBot0 = pad.t + hTop + gap;
    const hBot = h - pad.b - yBot0;
    const xOf = (t) => pad.l + pw * ((t - t0) / (t1 - t0));

    // Top subplot: v_kph [0..vmax] & lead_d [0..140]
    const vMax = Math.max(40, Math.ceil(Math.max(...tel.v, 30) / 20) * 20);
    const yV = (v) => yTop0 + hTop * (1 - Math.max(0, Math.min(vMax, v)) / vMax);
    const yLd = (d) => yTop0 + hTop * (1 - Math.max(0, Math.min(140, d)) / 140);

    // Bottom subplot: a_ego, a_base, a_target
    const aLo = Math.min(-3.5, Math.floor(Math.min(...tel.ae, ...tel.at, -1.5) * 2) / 2);
    const aHi = Math.max(2.5, Math.ceil(Math.max(...tel.ae, ...tel.at, 1.5) * 2) / 2);
    const yA = (a) => yBot0 + hBot * (1 - (Math.max(aLo, Math.min(aHi, a)) - aLo) / (aHi - aLo));

    // Pedal/Active shading in Top/Bottom
    for (let i = 0; i < tArr.length - 1; i++) {{
      const fl = tel.fl[i] || 0;
      const x1 = xOf(tArr[i]), x2 = xOf(tArr[i + 1]);
      if (fl & 4) {{ // gas
        ctx.fillStyle = "rgba(34, 197, 94, 0.14)";
        ctx.fillRect(x1, yTop0 + hTop - 12, Math.max(1, x2 - x1), 12);
      }}
      if (fl & 8) {{ // brake
        ctx.fillStyle = "rgba(239, 68, 68, 0.24)";
        ctx.fillRect(x1, yTop0, Math.max(1, x2 - x1), hTop);
      }}
    }}

    // Anomaly highlight bands
    longAnoms.forEach(a => {{
      const ax1 = xOf(a.t_start), ax2 = Math.max(ax1 + 4, xOf(a.t_end));
      ctx.fillStyle = a.severity === "high" ? "rgba(239,68,68,0.18)" : "rgba(245,158,11,0.14)";
      ctx.fillRect(ax1, yTop0, ax2 - ax1, hTop);
      ctx.fillRect(ax1, yBot0, ax2 - ax1, hBot);
    }});

    if (focusWindow && focusWindow.segName === s.segment) {{
      const fx1 = xOf(focusWindow.tStart), fx2 = Math.max(fx1 + 4, xOf(focusWindow.tEnd));
      ctx.strokeStyle = "#EF4444";
      ctx.lineWidth = 2;
      ctx.strokeRect(fx1, yTop0, fx2 - fx1, h - pad.t - pad.b);
    }}

    // Axes & grids
    ctx.strokeStyle = "#1E293B";
    ctx.lineWidth = 1;
    ctx.font = "10px JetBrains Mono, monospace";
    [0, vMax * 0.5, vMax].forEach(vt => {{
      const y = yV(vt);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.fillStyle = "#64748B"; ctx.textAlign = "right";
      ctx.fillText(vt.toFixed(0), pad.l - 6, y + 3);
    }});
    [aLo, 0, aHi].forEach(at => {{
      const y = yA(at);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.fillStyle = "#64748B"; ctx.textAlign = "right";
      ctx.fillText(at.toFixed(1), pad.l - 6, y + 3);
    }});
    ctx.fillStyle = "#94A3B8"; ctx.textAlign = "left";
    ctx.fillText("km/h", 6, yTop0 + 10);
    ctx.fillText("m/s²", 6, yBot0 + 10);
    ctx.fillText("lead(m)", w - pad.r + 4, yTop0 + 10);

    // X time ticks
    for (let k = 0; k <= 6; k++) {{
      const tt = t0 + (t1 - t0) * (k / 6);
      const x = xOf(tt);
      ctx.fillStyle = "#64748B"; ctx.textAlign = "center";
      ctx.fillText(tt.toFixed(0) + "s", x, h - 6);
    }}

    const drawSeries = (vals, yFn, color, width, dash) => {{
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < tArr.length; i++) {{
        const v = vals[i];
        if (v === null || v === undefined) {{ started = false; continue; }}
        const x = xOf(tArr[i]), y = yFn(v);
        if (!started) {{ ctx.moveTo(x, y); started = true; }} else ctx.lineTo(x, y);
      }}
      ctx.stroke();
      ctx.restore();
    }};

    drawSeries(tel.ld, yLd, "#94A3B8", 1.2, [4, 3]);
    drawSeries(tel.v, yV, "#38BDF8", 2.0, null);
    drawSeries(tel.ab, yA, "#94A3B8", 1.2, [4, 3]);
    drawSeries(tel.at, yA, "#F59E0B", 1.6, null);
    drawSeries(tel.ae, yA, "#38BDF8", 2.0, null);

    if (longHoverX !== null && longHoverX >= pad.l && longHoverX <= w - pad.r) {{
      ctx.strokeStyle = "#CBD5E166";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(longHoverX, yTop0); ctx.lineTo(longHoverX, h - pad.b); ctx.stroke();
    }}
  }}

  // 5) Lateral Overview Canvas
  const latOvHits = [];
  function drawLatOverview() {{
    const canvas = document.getElementById("canvas-lat-overview");
    const {{ ctx, w, h }} = setupCanvas(canvas);
    latOvHits.length = 0;
    if (!SEGMENTS.length) return;
    const pad = {{ l: 44, r: 44, t: 18, b: 28 }};
    const pw = w - pad.l - pad.r;
    const ph = h - pad.t - pad.b;

    const maxRms = Math.max(12, ...SEGMENTS.map(s => Number(s.curve_steer_rate_rms || 0)));
    const yLat = (v) => pad.t + ph * (1 - Math.min(4.0, Math.max(0, v)) / 4.0);
    const yRms = (v) => pad.t + ph * (1 - Math.min(maxRms, Math.max(0, v)) / maxRms);

    ctx.strokeStyle = "#1E293B";
    ctx.lineWidth = 1;
    ctx.font = "10px JetBrains Mono, monospace";
    [0, 1.5, 3.0, 4.0].forEach(tick => {{
      const y = yLat(tick);
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.fillStyle = "#64748B"; ctx.textAlign = "right";
      ctx.fillText(tick.toFixed(1), pad.l - 6, y + 3);
    }});

    const slot = pw / SEGMENTS.length;
    const bw = Math.max(5, Math.min(18, slot * 0.34));
    SEGMENTS.forEach((s, i) => {{
      const cx = pad.l + slot * (i + 0.5);
      latOvHits.push({{ x0: pad.l + slot * i, x1: pad.l + slot * (i + 1), idx: i }});
      if (i === selectedSegIdx) {{
        ctx.fillStyle = "rgba(56, 189, 248, 0.12)";
        ctx.fillRect(pad.l + slot * i, pad.t, slot, ph);
      }}
      if (!s.error) {{
        const y1 = yLat(s.lat_accel_max || 0);
        ctx.fillStyle = (s.lat_accel_max >= 2.8) ? "#EF4444" : "#38BDF8";
        ctx.fillRect(cx - bw - 1, y1, bw, pad.t + ph - y1);

        const y2 = yRms(s.curve_steer_rate_rms || 0);
        ctx.fillStyle = "#F59E0B";
        ctx.fillRect(cx + 1, y2, bw, pad.t + ph - y2);

        if ((s.steer_sat_frames || 0) > 0) {{
          ctx.fillStyle = "#EF4444";
          ctx.beginPath(); ctx.arc(cx, pad.t + 6, 3.5, 0, Math.PI * 2); ctx.fill();
        }}
      }}
      ctx.fillStyle = i === selectedSegIdx ? "#38BDF8" : "#94A3B8";
      ctx.textAlign = "center";
      ctx.fillText(shortSeg(s.segment).replace("seg ", "s"), cx, h - 9);
    }});
  }}

  // 6) Lateral Detail Canvas (3 bands: lat_accel, steer_angle, lane_offset)
  let latHoverX = null;
  function drawLatDetail() {{
    const s = SEGMENTS[selectedSegIdx];
    const metaEl = document.getElementById("lat-seg-meta");
    const pillsEl = document.getElementById("lat-anom-pills");
    const canvas = document.getElementById("canvas-lat-detail");
    const {{ ctx, w, h }} = setupCanvas(canvas);
    if (!s || s.error || !s.telemetry || !(s.telemetry.t || []).length) {{
      metaEl.innerHTML = `<span class="muted">표시할 횡방향 시계열 데이터가 없습니다 (${{esc(s ? (s.error || s.segment) : "-")}}).</span>`;
      pillsEl.innerHTML = "";
      return;
    }}
    const tel = s.telemetry;
    const tArr = tel.t;
    const t0 = tArr[0], t1 = Math.max(t0 + 1.0, tArr[tArr.length - 1]);
    const latAnoms = (s.anomalies || []).filter(a => CAT_META[a.category]?.axis === "lat");
    metaEl.innerHTML = `
      <span><strong>${{esc(s.segment)}}</strong></span>
      <span>발생일시 <strong class="mono">${{esc(segTimeRange(s))}}</strong></span>
      <span>최대 횡가속 <strong class="mono">${{(s.lat_accel_max || 0).toFixed(2)}} m/s²</strong></span>
      <span>코너 조향 RMS <strong class="mono">${{(s.curve_steer_rate_rms || 0).toFixed(1)}} °/s</strong></span>
      <span>토크 포화 <strong class="mono">${{s.steer_sat_frames || 0}} frames</strong></span>
      <span>횡방향 이상치 <strong class="mono">${{latAnoms.length}}건</strong></span>
    `;
    pillsEl.innerHTML = latAnoms.map(a =>
      `<button type="button" class="anom-pill ${{esc(a.severity)}}" data-t0="${{a.t_start}}" data-t1="${{a.t_end}}">` +
      `${{esc(CAT_META[a.category]?.label || a.category)}} t=${{a.t_start.toFixed(1)}}s (${{a.v_kph.toFixed(0)}}k)</button>`
    ).join("") || '<span class="muted" style="font-size:12px;">이 세그먼트에서 검출된 횡방향 이상치가 없습니다.</span>';
    pillsEl.querySelectorAll(".anom-pill").forEach(btn => {{
      btn.addEventListener("click", () => {{
        focusWindow = {{ segName: s.segment, tStart: Number(btn.dataset.t0), tEnd: Number(btn.dataset.t1) }};
        drawLatDetail();
      }});
    }});

    const pad = {{ l: 48, r: 20, t: 14, b: 24 }};
    const pw = w - pad.l - pad.r;
    const gap = 14;
    const usable = h - pad.t - pad.b - gap * 2;
    const h1 = Math.floor(usable * 0.36), y1 = pad.t;
    const h2 = Math.floor(usable * 0.36), y2 = y1 + h1 + gap;
    const h3 = usable - h1 - h2, y3 = y2 + h2 + gap;
    const xOf = (t) => pad.l + pw * ((t - t0) / (t1 - t0));

    const laAbs = Math.max(2.5, ...tel.la.map(Math.abs), ...tel.dl.map(Math.abs));
    const saAbs = Math.max(30, ...tel.sa.map(Math.abs), ...tel.ds.map(Math.abs));
    const loAbs = 0.65;

    const mapSym = (val, lim, top, bh) => top + bh * (0.5 - Math.max(-lim, Math.min(lim, val)) / (2 * lim));

    // Flags shading (steer_saturated=32, steering_pressed=16, blinker/lc=64)
    for (let i = 0; i < tArr.length - 1; i++) {{
      const fl = tel.fl[i] || 0;
      const x1 = xOf(tArr[i]), x2 = xOf(tArr[i + 1]);
      const bw = Math.max(1, x2 - x1);
      if (fl & 32) {{ ctx.fillStyle = "rgba(239,68,68,0.22)"; ctx.fillRect(x1, y1, bw, h1); }}
      if (fl & 16) {{ ctx.fillStyle = "rgba(245,158,11,0.16)"; ctx.fillRect(x1, y2, bw, h2); }}
      if (fl & 64) {{ ctx.fillStyle = "rgba(148,163,184,0.15)"; ctx.fillRect(x1, y3, bw, h3); }}
    }}

    latAnoms.forEach(a => {{
      const ax1 = xOf(a.t_start), ax2 = Math.max(ax1 + 4, xOf(a.t_end));
      ctx.fillStyle = a.severity === "high" ? "rgba(239,68,68,0.18)" : "rgba(245,158,11,0.13)";
      ctx.fillRect(ax1, y1, ax2 - ax1, h1 + gap + h2 + gap + h3);
    }});

    if (focusWindow && focusWindow.segName === s.segment) {{
      const fx1 = xOf(focusWindow.tStart), fx2 = Math.max(fx1 + 4, xOf(focusWindow.tEnd));
      ctx.strokeStyle = "#EF4444";
      ctx.lineWidth = 2;
      ctx.strokeRect(fx1, y1, fx2 - fx1, h - pad.t - pad.b);
    }}

    // Grid & band labels
    ctx.strokeStyle = "#1E293B";
    ctx.lineWidth = 1;
    ctx.font = "10px JetBrains Mono, monospace";
    [[y1, h1, "m/s²", laAbs], [y2, h2, "deg", saAbs], [y3, h3, "off(m)", loAbs]].forEach(([bt, bh, unit, lim]) => {{
      const ym = bt + bh / 2;
      ctx.beginPath(); ctx.moveTo(pad.l, ym); ctx.lineTo(w - pad.r, ym); ctx.stroke();
      ctx.fillStyle = "#64748B"; ctx.textAlign = "right";
      ctx.fillText(lim.toFixed(lim < 2 ? 2 : 0), pad.l - 6, bt + 10);
      ctx.fillText("0", pad.l - 6, ym + 3);
      ctx.fillText("-" + lim.toFixed(lim < 2 ? 2 : 0), pad.l - 6, bt + bh);
      ctx.fillStyle = "#94A3B8"; ctx.textAlign = "left";
      ctx.fillText(unit, pad.l + 4, bt + 10);
    }});

    for (let k = 0; k <= 6; k++) {{
      const tt = t0 + (t1 - t0) * (k / 6);
      ctx.fillStyle = "#64748B"; ctx.textAlign = "center";
      ctx.fillText(tt.toFixed(0) + "s", xOf(tt), h - 6);
    }}

    const drawBand = (vals, top, bh, lim, color, width, dash) => {{
      ctx.save();
      ctx.strokeStyle = color; ctx.lineWidth = width;
      if (dash) ctx.setLineDash(dash);
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < tArr.length; i++) {{
        const v = vals[i];
        if (v === null || v === undefined) {{ started = false; continue; }}
        const x = xOf(tArr[i]), y = mapSym(v, lim, top, bh);
        if (!started) {{ ctx.moveTo(x, y); started = true; }} else ctx.lineTo(x, y);
      }}
      ctx.stroke();
      ctx.restore();
    }};

    drawBand(tel.dl, y1, h1, laAbs, "#F59E0B", 1.3, [4, 3]);
    drawBand(tel.la, y1, h1, laAbs, "#38BDF8", 2.0, null);
    drawBand(tel.ds, y2, h2, saAbs, "#CBD5E1", 1.3, [4, 3]);
    drawBand(tel.sa, y2, h2, saAbs, "#22C55E", 2.0, null);
    drawBand(tel.lo, y3, h3, loAbs, "#38BDF8", 1.8, null);

    if (latHoverX !== null && latHoverX >= pad.l && latHoverX <= w - pad.r) {{
      ctx.strokeStyle = "#CBD5E166"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(latHoverX, y1); ctx.lineTo(latHoverX, h - pad.b); ctx.stroke();
    }}
  }}

  // Mouse interactivity bindings
  function bindCanvasInteractions() {{
    const cLongOv = document.getElementById("canvas-long-overview");
    const tipLongOv = document.getElementById("tip-long-overview");
    cLongOv.addEventListener("mousemove", (e) => {{
      const rect = cLongOv.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const hit = longOvHits.find(h => x >= h.x0 && x <= h.x1);
      if (!hit) {{ tipLongOv.style.display = "none"; return; }}
      const s = SEGMENTS[hit.idx];
      showTooltip(tipLongOv, cLongOv, e.clientX, e.clientY,
        `<strong>${{esc(s.segment)}}</strong><br>일시: ${{esc(segTimeRange(s))}}<br>속도: ${{(s.v_min_kph||0).toFixed(0)}}..${{(s.v_max_kph||0).toFixed(0)}} km/h<br>` +
        `a_ego: ${{(s.a_ego_min||0).toFixed(2)}}..+${{(s.a_ego_max||0).toFixed(2)}} m/s²<br>이상치: ${{(s.anomalies||[]).length}}건`);
    }});
    cLongOv.addEventListener("mouseleave", () => tipLongOv.style.display = "none");
    cLongOv.addEventListener("click", (e) => {{
      const x = e.clientX - cLongOv.getBoundingClientRect().left;
      const hit = longOvHits.find(h => x >= h.x0 && x <= h.x1);
      if (hit) {{ focusWindow = null; selectSegment(hit.idx); }}
    }});

    const cLatOv = document.getElementById("canvas-lat-overview");
    const tipLatOv = document.getElementById("tip-lat-overview");
    cLatOv.addEventListener("mousemove", (e) => {{
      const x = e.clientX - cLatOv.getBoundingClientRect().left;
      const hit = latOvHits.find(h => x >= h.x0 && x <= h.x1);
      if (!hit) {{ tipLatOv.style.display = "none"; return; }}
      const s = SEGMENTS[hit.idx];
      showTooltip(tipLatOv, cLatOv, e.clientX, e.clientY,
        `<strong>${{esc(s.segment)}}</strong><br>일시: ${{esc(segTimeRange(s))}}<br>최대횡가속: ${{(s.lat_accel_max||0).toFixed(2)}} m/s²<br>` +
        `코너조향RMS: ${{(s.curve_steer_rate_rms||0).toFixed(1)}} °/s<br>토크포화: ${{s.steer_sat_frames||0}} frames`);
    }});
    cLatOv.addEventListener("mouseleave", () => tipLatOv.style.display = "none");
    cLatOv.addEventListener("click", (e) => {{
      const x = e.clientX - cLatOv.getBoundingClientRect().left;
      const hit = latOvHits.find(h => x >= h.x0 && x <= h.x1);
      if (hit) {{ focusWindow = null; selectSegment(hit.idx); }}
    }});

    const bindDetailHover = (canvasId, tipId, padL, padR, isLong) => {{
      const canvas = document.getElementById(canvasId);
      const tip = document.getElementById(tipId);
      canvas.addEventListener("mousemove", (e) => {{
        const s = SEGMENTS[selectedSegIdx];
        const tel = s && s.telemetry;
        if (!tel || !(tel.t || []).length) {{ tip.style.display = "none"; return; }}
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        if (isLong) longHoverX = x; else latHoverX = x;
        const pw = rect.width - padL - padR;
        const t0 = tel.t[0], t1 = Math.max(t0 + 1, tel.t[tel.t.length - 1]);
        const targetT = t0 + Math.max(0, Math.min(1, (x - padL) / Math.max(pw, 1))) * (t1 - t0);
        let best = 0, bestD = 1e9;
        for (let i = 0; i < tel.t.length; i++) {{
          const d = Math.abs(tel.t[i] - targetT);
          if (d < bestD) {{ bestD = d; best = i; }}
        }}
        const fl = tel.fl[best] || 0;
        const wc = fmtWallClock(s, tel.t[best]);
        const tLabel = wc ? `${{wc}} (t=${{tel.t[best].toFixed(2)}}s)` : `t=${{tel.t[best].toFixed(2)}}s`;
        const html = isLong
          ? `${{tLabel}} | v=${{tel.v[best].toFixed(1)}} km/h<br>` +
            `a_ego=${{tel.ae[best].toFixed(2)}} | a_base=${{tel.ab[best].toFixed(2)}} | a_tgt=${{tel.at[best].toFixed(2)}}<br>` +
            `lead=${{tel.ld[best] != null ? tel.ld[best].toFixed(1) + 'm' : '없음'}} | ` +
            `${{(fl&1)?'LONG':'OFF'}} ${{(fl&4)?'GAS':''}} ${{(fl&8)?'BRK':''}}`
          : `${{tLabel}} | v=${{tel.v[best].toFixed(1)}} km/h<br>` +
            `lat=${{tel.la[best].toFixed(2)}} / tgt=${{tel.dl[best].toFixed(2)}} m/s²<br>` +
            `steer=${{tel.sa[best].toFixed(1)}}° / tgt=${{tel.ds[best].toFixed(1)}}°<br>` +
            `offset=${{tel.lo[best] != null ? tel.lo[best].toFixed(2)+'m' : '-'}} ${{(fl&32)?'[SAT]':''}} ${{(fl&16)?'[MAN]':''}}`;
        if (isLong) drawLongDetail(); else drawLatDetail();
        showTooltip(tip, canvas, e.clientX, e.clientY, html);
      }});
      canvas.addEventListener("mouseleave", () => {{
        tip.style.display = "none";
        if (isLong) {{ longHoverX = null; drawLongDetail(); }}
        else {{ latHoverX = null; drawLatDetail(); }}
      }});
    }};
    bindDetailHover("canvas-long-detail", "tip-long-detail", 48, 46, true);
    bindDetailHover("canvas-lat-detail", "tip-lat-detail", 48, 20, false);
  }}

  // 7) Filterable Anomaly Explorer
  function renderExplorerOptions() {{
    const catSel = document.getElementById("filter-cat");
    const prev = catSel.value || "all";
    catSel.innerHTML = '<option value="all">카테고리: 전체</option>';
    Object.keys(CAT_META).forEach(k => {{
      const opt = document.createElement("option");
      opt.value = k;
      opt.textContent = `${{CAT_META[k].label}} (${{k}})`;
      catSel.appendChild(opt);
    }});
    catSel.value = Array.from(catSel.options).some(o => o.value === prev) ? prev : "all";
  }}

  function bindStaticListenersOnce() {{
    ["sel-long-seg", "sel-lat-seg"].forEach(id => {{
      document.getElementById(id).addEventListener("change", (e) => {{
        focusWindow = null;
        selectSegment(Number(e.target.value));
      }});
    }});
    document.getElementById("btn-long-prev").addEventListener("click", () => stepSegment(-1));
    document.getElementById("btn-long-next").addEventListener("click", () => stepSegment(1));
    document.getElementById("btn-lat-prev").addEventListener("click", () => stepSegment(-1));
    document.getElementById("btn-lat-next").addEventListener("click", () => stepSegment(1));

    document.getElementById("route-search").addEventListener("input", renderRouteGroups);
    document.getElementById("btn-routes-refresh").addEventListener("click", refreshRouteGroupsFromApi);
    document.getElementById("btn-wifi-discover").addEventListener("click", discoverWifiDevice);
    document.getElementById("btn-toggle-dongle-cfg").addEventListener("click", () => {{
      const panel = document.getElementById("dongle-config-panel");
      panel.style.display = (panel.style.display === "none" || !panel.style.display) ? "flex" : "none";
    }});
    document.getElementById("btn-save-dongle").addEventListener("click", () => saveDongleConfig(false));
    document.getElementById("btn-reset-dongle").addEventListener("click", () => saveDongleConfig(true));

    document.querySelectorAll("#axis-filter-btns button").forEach(btn => {{
      btn.addEventListener("click", () => {{
        axisFilter = btn.dataset.axis;
        document.querySelectorAll("#axis-filter-btns button").forEach(b => b.classList.remove("active-btn"));
        btn.classList.add("active-btn");
        renderAnomalyTable();
      }});
    }});
    ["filter-sev", "filter-cat", "filter-q"].forEach(id => {{
      document.getElementById(id).addEventListener("input", renderAnomalyTable);
      document.getElementById(id).addEventListener("change", renderAnomalyTable);
    }});
    window.addEventListener("resize", () => {{
      drawLongOverview();
      drawLatOverview();
      drawLongDetail();
      drawLatDetail();
    }});
  }}

  function renderAnomalyTable() {{
    const sev = document.getElementById("filter-sev").value;
    const cat = document.getElementById("filter-cat").value;
    const q = document.getElementById("filter-q").value.trim().toLowerCase();
    const filtered = ANOMALIES.filter(a => {{
      const axis = CAT_META[a.category]?.axis || "long";
      if (axisFilter !== "all" && axis !== axisFilter) return false;
      if (sev !== "all" && a.severity !== sev) return false;
      if (cat !== "all" && a.category !== cat) return false;
      if (q && !(String(a.segment).toLowerCase().includes(q) || String(a.summary).toLowerCase().includes(q) || String(a.category).toLowerCase().includes(q))) return false;
      return true;
    }});
    document.getElementById("filter-count").textContent = `${{filtered.length}} / ${{ANOMALIES.length}}건`;
    const tbody = document.getElementById("anomaly-tbody");
    if (!filtered.length) {{
      tbody.innerHTML = '<tr><td colspan="7" class="muted">조건에 해당하는 이상치가 없습니다.</td></tr>';
      return;
    }}
    tbody.innerHTML = filtered.map((a, idx) => {{
      const axis = CAT_META[a.category]?.axis || "long";
      const axisBadge = axis === "lat" ? '<span class="badge badge-amber">횡방향</span>' : '<span class="badge badge-cyan">종방향</span>';
      const sevBadge = a.severity === "high" ? '<span class="badge badge-red">높음</span>' : '<span class="badge">보통</span>';
      const segObj = SEGMENTS.find(s => s.segment === a.segment);
      const wc = fmtWallClock(segObj, a.t_start);
      const segDt = segObj && segObj.started_at ? `<br><span class="muted mono" style="font-size:10px">${{esc(segObj.started_at.slice(0, 10))}}${{wc ? " " + esc(wc) : ""}}</span>` : "";
      return `<tr data-row="${{idx}}">
        <td>${{axisBadge}}</td>
        <td>${{sevBadge}}</td>
        <td><strong>${{esc(CAT_META[a.category]?.label || a.category)}}</strong><br><span class="muted mono" style="font-size:10px">${{esc(a.category)}}</span></td>
        <td class="mono">${{esc(shortSeg(a.segment))}}${{segDt}}</td>
        <td class="mono">${{Number(a.t_start).toFixed(1)}}~${{Number(a.t_end).toFixed(1)}}s</td>
        <td class="mono">${{Number(a.v_kph).toFixed(1)}} km/h</td>
        <td>${{esc(a.summary)}}</td>
      </tr>`;
    }}).join("");
    tbody.querySelectorAll("tr[data-row]").forEach(tr => {{
      tr.addEventListener("click", () => {{
        const a = filtered[Number(tr.dataset.row)];
        const sIdx = SEGMENTS.findIndex(s => s.segment === a.segment);
        if (sIdx >= 0) {{
          focusWindow = {{ segName: a.segment, tStart: Number(a.t_start), tEnd: Number(a.t_end) }};
          selectSegment(sIdx);
          const axis = CAT_META[a.category]?.axis || "long";
          document.getElementById(axis === "lat" ? "sec-lat" : "sec-long").scrollIntoView({{ behavior: "smooth" }});
        }}
      }});
    }});
  }}

  // 8) Action Plans
  function renderPlans() {{
    const pHost = document.getElementById("param-guides-grid");
    pHost.innerHTML = (DATA.param_guides || []).map(g => `
      <div class="guide-card">
        <div class="guide-head">
          <span class="guide-title mono">${{esc(g.param)}}</span>
          <span class="badge ${{g.priority === '주의' ? 'badge-amber' : 'badge-cyan'}}">${{esc(g.priority)}}</span>
        </div>
        <div class="kv-line"><strong>현재 관찰값:</strong> <span class="mono">${{esc(g.current)}}</span></div>
        <div class="kv-line"><strong>조정 방향:</strong> ${{esc(g.direction)}}</div>
        <div class="kv-line"><strong>적용 조건:</strong> ${{esc(g.condition)}}</div>
        <div class="kv-line"><strong>확인 포인트:</strong> ${{esc(g.caution)}}</div>
      </div>
    `).join("") || '<div class="empty-inline">파라미터 가이드 없음</div>';

    const cHost = document.getElementById("code-plans-grid");
    cHost.innerHTML = (DATA.code_plans || []).map(p => {{
      const isClean = !String(p.status || "").includes("승인 대기");
      const badgeCls = isClean ? "badge-green" : "badge-amber";
      return `<div class="plan-card ${{isClean ? 'clean' : ''}}">
        <div class="guide-head">
          <span class="guide-title">[${{esc(p.id)}}] ${{esc(p.title)}}</span>
          <span class="badge ${{badgeCls}}">${{esc(p.status)}}</span>
        </div>
        <div class="kv-line"><strong>판단 근거:</strong> ${{esc(p.reason)}}</div>
        <div class="kv-line"><strong>격리 조건:</strong> ${{esc(p.target_condition)}}</div>
        <div class="kv-line"><strong>수정 범위:</strong> <code class="mono">${{esc(p.change_scope)}}</code></div>
        <div class="kv-line"><strong>ver-1.0 회귀 금지:</strong> ${{esc(p.forbidden_regression)}}</div>
        <div class="cmd-box">${{esc(p.verify_cmd)}}</div>
      </div>`;
    }}).join("");
  }}

  function renderAll() {{
    renderRouteGroups();
    renderHeaderAndSummary();
    renderAnomalyBars();
    populateSelectors();
    renderHeatmap();
    renderExplorerOptions();
    renderAnomalyTable();
    renderPlans();
    drawLongOverview();
    drawLatOverview();
    drawLongDetail();
    drawLatDetail();
  }}

  // Initial mount
  bindStaticListenersOnce();
  bindCanvasInteractions();
  renderAll();
}})();
</script>
</body>
</html>
"""

---
name: drive-log-review
description: 오파 단말 Wi-Fi 주행로그 일괄 수집, 1차 이상치 스캔, ver-1.0 무회귀 검증 기반 마이너 튜닝 워크플로우. 사용자가 "최근 주행 로그 가져와서 분석해줘", "로그 동기화해서 문제 찾아줘", "ver-1.0 유지하면서 다듬어줘" 등을 요청할 때 사용한다.
---

# Drive Log Review & ver-1.0 Regression-Safe Tuning

## 핵심 원칙
1. **`ver-1.0` 기준선 보존**: Git 태그 `ver-1.0` (`738f018ef`)을 항상 무회귀 기준선으로 삼는다.
2. **승인 전 코드 수정 금지**: 로그 분석 및 개선 계획 제시 단계에서는 `openpilot/selfdrive/` 런타임 코드를 절대 수정하지 않는다. 사용자 승인 후에만 편집·테스트·커밋한다.
3. **고립된 마이너 개선만 허용**: 특정 속도대/조건(예: `v >= 55 km/h` 원거리)의 문제만 격리해 수정하며, 비대상 속도대(`0-25` 서행 정체 등)는 변경률 `0%`를 목표로 한다.

## 1단계: 주행로그 그룹 조회 → 선택 동기화 + 1차 이상치 스캔
- 브라우저 HTML UI 안에서 최근 **주행로그 그룹 목록**을 보고 클릭 한 번으로 다운로드·분석하려면 로컬 웹 스튜디오를 실행한다:
  ```bash
  PYTHONPATH=. uv run --no-sync python3 -m openpilot.tools.drive_analyzer serve --port 8765 --open
  ```
- CLI에서 최근 **주행로그 그룹(루트 단위)** 목록과 매칭된 주행리포트를 확인할 수도 있다:
  ```bash
  PYTHONPATH=. uv run --no-sync python3 -m openpilot.tools.drive_analyzer routes --limit 15
  ```
- 사용자가 선택한 번호 또는 루트명만 단말에서 내려받아 분석하고 전용 HTML 리포트(`scans/route_<루트>/analysis_report.html`)를 생성한다:
  ```bash
  PYTHONPATH=. uv run --no-sync python3 -m openpilot.tools.drive_analyzer analyze --route <번호 또는 루트명>
  ```
- 또는 최신 N개 루트를 한 번에 동기화·스캔한다:
  ```bash
  PYTHONPATH=. uv run --no-sync python3 -m openpilot.tools.drive_analyzer run --latest-routes 2
  ```
- 이미 로컬에 받아둔 폴더/파일만 분석할 경우:
  ```bash
  PYTHONPATH=. uv run --no-sync python3 -m openpilot.tools.drive_analyzer scan --dir <경로...>
  ```
- **동글 ID 자동 인식 및 수동 입력**:
  - HTML 스튜디오 상단 **`[Wi-Fi 자동인식]`** 버튼(또는 CLI `discover --any`)을 실행하면 동일 Wi-Fi에서 단말 IP와 해당 단말의 `dongle_id`를 자동으로 인식해 `config.json`에 저장한다.
  - 특정 `dongle_id`를 수동 지정/변경해야 할 때는 HTML 상단 **`[동글ID 수동설정]`** 패널 또는 아래 CLI를 사용한다(초기화는 `--clear-dongle-id`):
    ```bash
    PYTHONPATH=. uv run --no-sync python3 -m openpilot.tools.drive_analyzer config --dongle-id <dongle_id>
    PYTHONPATH=. uv run --no-sync python3 -m openpilot.tools.drive_analyzer discover --any
    ```
- 또는 수동 IP 고정 설정 (`--dongle-id` 미지정 시 해당 IP에서 `dongle_id` 자동 인식):
  ```bash
  PYTHONPATH=. uv run --no-sync python3 -m openpilot.tools.drive_analyzer config --host <단말IP>
  ```

## 2단계: 2차 상세 분석 및 HTML 시각화 리포트 제공 (Antigravity)
- `run` 또는 `scan` 실행 시 `.analysis/drive_analyzer/scans/<timestamp>/analysis_report.html` 시각 리포트와 `scan_summary.json`, `scan_report.md`가 자동 생성된다.
  - 기존 JSON에 AI 상세 코멘트를 추가해 HTML만 재생성할 수도 있다:
    ```bash
    PYTHONPATH=. uv run --no-sync python3 -m openpilot.tools.drive_analyzer report --summary <scan_summary.json> [--notes-json <notes.json>] [--open]
    ```
- 사용자에게 응답할 때는 텍스트 나열에 그치지 말고, 생성된 `analysis_report.html` 경로를 클릭 가능한 `file://` 링크(필요 시 아티팩트 복사 포함)로 제공해 도식화된 **전체 요약 → 종방향(가감속) 세그먼트 그래프 → 횡방향(조향/코너링) 세그먼트 그래프 → 파라미터 튜닝 가이드 → `ver-1.0` 무회귀 개발 계획**을 바로 확인할 수 있게 한다.
  - 종방향: `auto_hard_brake`, `manual_brake_override`, `low_speed_surge`, `mid_speed_surge`, `far_lead_late_decel`
  - 조향(횡방향): `manual_steer_override`, `steer_pingpong`(와리가리), `curve_lat_overshoot`, `lane_center_bias` 및 세그먼트별 `curve_steer_rate_rms`(코너 조향각속도 RMS)
- 커브 감속 속도 파라미터(`AutoCurveSpeedFactor` vs `MapTurnSpeedFactor`) 조정 후에는 특히 기존 코너링 와리가리(`steer_pingpong`, `curve_steer_rate_rms`)와 횡가속 포화(`curve_lat_overshoot`) 변화 여부를 반드시 함께 비교한다.
- `.analysis/drive_analyzer/reports/`에 동기화된 `drive_reports` JSON(점수·감점·급정거 포함)도 HTML 종합 요약에 함께 표시된다.

## 3단계: 개선 계획 + `ver-1.0` A/B 회귀 영향도 제시 (승인 게이트)
- 문제를 확인했으면 **코드 커밋 전에** 개선안을 설계하고 `regress` 게이트로 `ver-1.0` 대비 종방향·조향 영향도를 수치화한다:
  ```bash
  PYTHONPATH=. uv run --no-sync python3 -m openpilot.tools.drive_analyzer regress \
    --baseline-ref ver-1.0 \
    --mode both \
    --target-dir <이상 세그먼트 경로> \
    --golden-dir .analysis/drive_analyzer/golden_v1.0 \
    --forbid-band 0-25 \
    --forbid-lat-band straight
  ```
- 사용자에게 아래 3가지를 제시하고 **승인**을 받은 뒤에야 실제 반영한다:
  1. 이상치 구간 정밀 원인 분석 (타임스탬프/센서/목표가속 수치)
  2. 격리된 수정 범위 (어느 파일의 어떤 조건만 바뀌는지)
  3. `ver-1.0` 대비 A/B 리플레이 영향도 표 (문제 구간 개선 수치 + `golden_v1.0` 속도대별 변경률)

## 4단계: 승인 후 구현 · 검증 · 커밋
- 관련 단위 테스트(`pytest`)와 `drive_analyzer regress --baseline-ref ver-1.0` 통과 확인 후 커밋/푸시한다.
- 언제든 실차 주행 결과가 마음에 들지 않으면 `ver-1.0` 태그 기준으로 해당 변경만 즉시 원복한다.

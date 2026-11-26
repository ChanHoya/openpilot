"""ver-1.0 A/B replay regression gate comparing baseline vs candidate longitudinal & lateral helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import inspect
from pathlib import Path
import subprocess
import sys
import types
from typing import Any, Sequence

from . import DEFAULT_BASELINE_REF, REPO_ROOT
from .scanner import FrameSample, discover_rlogs, load_segment_frames

SPEED_BANDS: tuple[tuple[str, float, float], ...] = (
  ("0-25", 0.0, 25.0),
  ("25-55", 25.0, 55.0),
  ("55-85", 55.0, 85.0),
  ("85+", 85.0, 1e6),
)
LATERAL_BANDS: tuple[tuple[str, float, float], ...] = (
  ("straight", 0.0, 0.5),
  ("mild_curve", 0.5, 1.3),
  ("sharp_curve", 1.3, 1e6),
)
_HUMAN_LONG_REL = "openpilot/selfdrive/controls/lib/human_longitudinal.py"
_DRIVE_HELPERS_REL = "openpilot/selfdrive/controls/lib/drive_helpers.py"


@dataclass
class BandStats:
  band: str
  active_frames: int = 0
  changed_frames: int = 0
  ramp_changed_frames: int = 0
  max_abs_delta: float = 0.0
  sum_delta: float = 0.0
  samples: list[dict[str, Any]] = field(default_factory=list)

  @property
  def changed_pct(self) -> float:
    return (100.0 * self.changed_frames / self.active_frames) if self.active_frames > 0 else 0.0

  @property
  def mean_delta(self) -> float:
    return (self.sum_delta / self.changed_frames) if self.changed_frames > 0 else 0.0

  def to_dict(self) -> dict[str, Any]:
    return {
      "band": self.band,
      "active_frames": self.active_frames,
      "changed_frames": self.changed_frames,
      "changed_pct": round(self.changed_pct, 3),
      "ramp_changed_frames": self.ramp_changed_frames,
      "max_abs_delta": round(self.max_abs_delta, 4),
      "mean_delta": round(self.mean_delta, 4),
      "samples": self.samples,
    }


@dataclass
class DatasetComparison:
  label: str
  segment_count: int
  active_frames: int
  changed_frames: int
  ramp_changed_frames: int
  max_abs_delta: float
  bands: dict[str, BandStats]
  domain: str = "long"

  @property
  def changed_pct(self) -> float:
    return (100.0 * self.changed_frames / self.active_frames) if self.active_frames > 0 else 0.0

  def to_dict(self) -> dict[str, Any]:
    return {
      "label": self.label,
      "domain": self.domain,
      "segment_count": self.segment_count,
      "active_frames": self.active_frames,
      "changed_frames": self.changed_frames,
      "changed_pct": round(self.changed_pct, 3),
      "ramp_changed_frames": self.ramp_changed_frames,
      "max_abs_delta": round(self.max_abs_delta, 4),
      "bands": {k: v.to_dict() for k, v in self.bands.items()},
    }


@dataclass
class RegressionReport:
  baseline_ref: str
  passed: bool
  reasons: list[str]
  datasets: list[DatasetComparison]

  def to_dict(self) -> dict[str, Any]:
    return {
      "baseline_ref": self.baseline_ref,
      "passed": self.passed,
      "reasons": list(self.reasons),
      "datasets": [d.to_dict() for d in self.datasets],
    }


def load_module_from_git_ref(git_ref: str, rel_path: str, repo_root: Path = REPO_ROOT) -> types.ModuleType:
  src = subprocess.check_output(
    ["git", "-C", str(repo_root), "show", f"{git_ref}:{rel_path}"],
    text=True,
  )
  safe_ref = "".join(c if c.isalnum() else "_" for c in git_ref)
  safe_rel = "".join(c if c.isalnum() else "_" for c in Path(rel_path).stem)
  mod_name = f"_{safe_rel}_ref_{safe_ref}"
  mod = types.ModuleType(mod_name)
  sys.modules[mod_name] = mod
  exec(compile(src, f"{mod_name}.py", "exec"), mod.__dict__)
  return mod


def load_shaper_from_git_ref(git_ref: str, repo_root: Path = REPO_ROOT) -> types.ModuleType:
  return load_module_from_git_ref(git_ref, _HUMAN_LONG_REL, repo_root=repo_root)


def load_drive_helpers_from_git_ref(git_ref: str, repo_root: Path = REPO_ROOT) -> types.ModuleType:
  return load_module_from_git_ref(git_ref, _DRIVE_HELPERS_REL, repo_root=repo_root)


def _accepted_param_filter(fn: Any) -> tuple[bool, set[str]]:
  sig = inspect.signature(fn)
  has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
  return has_var_kw, set(sig.parameters.keys())


def _accepted_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
  has_var_kw, names = _accepted_param_filter(fn)
  if has_var_kw:
    return kwargs
  return {k: v for k, v in kwargs.items() if k in names}


def _band_for_speed(v_kph: float) -> str:
  for name, lo, hi in SPEED_BANDS:
    if lo <= v_kph < hi:
      return name
  return SPEED_BANDS[-1][0]


def compare_segment_frames(
  baseline_mod: Any,
  current_mod: Any,
  segments: Sequence[tuple[str, dict[str, Any], Sequence[FrameSample]]],
  *,
  dataset_label: str,
  delta_threshold: float = 0.02,
) -> DatasetComparison:
  bands = {name: BandStats(band=name) for name, _, _ in SPEED_BANDS}
  total_active = 0
  total_changed = 0
  total_ramp_changed = 0
  overall_max_delta = 0.0

  base_ramp_fn = baseline_mod.compute_accel_ramp_targets
  curr_ramp_fn = current_mod.compute_accel_ramp_targets

  for seg_name, meta, frames in segments:
    sh_base = baseline_mod.HumanLongitudinalShaper()
    sh_curr = current_mod.HumanLongitudinalShaper()
    stop_dist = float(meta.get("stop_dist", 5.0))
    base_var_kw, base_shape_params = _accepted_param_filter(sh_base.shape)
    curr_var_kw, curr_shape_params = _accepted_param_filter(sh_curr.shape)

    for idx, f in enumerate(frames):
      dt = 0.05 if idx == 0 else max(0.01, min(0.20, f.t - frames[idx - 1].t))
      reset = (not f.long_active) or f.gas_pressed or f.brake_pressed

      common_kwargs = dict(
        a_mpc=f.a_base,
        v_target_mpc=f.v_target_now,
        j_target_mpc=f.j_target_now,
        v_ego=f.v_kph / 3.6,
        a_ego=f.a_ego,
        pitch=0.0,
        action_t=0.25,
        lead_status=f.lead_status,
        d_rel=f.lead_d,
        v_rel=f.lead_v_rel,
        v_lead=f.lead_v,
        a_lead=f.lead_a,
        t_follow=f.t_follow,
        comfort_brake=2.5,
        stop_dist=stop_dist,
        should_stop=f.should_stop,
        fcw=f.fcw,
        reset_state=reset,
        mpc_source=f.mpc_source,
        lead_radar=f.lead_radar,
        lead_prob=f.lead_prob,
      )
      b_kwargs = common_kwargs if base_var_kw else {k: v for k, v in common_kwargs.items() if k in base_shape_params}
      c_kwargs = common_kwargs if curr_var_kw else {k: v for k, v in common_kwargs.items() if k in curr_shape_params}
      out_b = sh_base.shape(**b_kwargs)
      out_c = sh_curr.shape(**c_kwargs)

      if reset:
        continue

      band = _band_for_speed(f.v_kph)
      bs = bands[band]
      bs.active_frames += 1
      total_active += 1

      ramp_kwargs = dict(
        d_rel=f.lead_d,
        v_rel=f.lead_v_rel,
        v_lead=f.lead_v,
        a_lead_filt=f.lead_a,
        t_follow=f.t_follow,
        comfort_brake=2.5,
        stop_dist=stop_dist,
        lead_radar=f.lead_radar,
        lead_prob=f.lead_prob,
      )
      r_b = base_ramp_fn(f.v_kph / 3.6, 1.8, f.lead_status, **_accepted_kwargs(base_ramp_fn, ramp_kwargs))
      r_c = curr_ramp_fn(f.v_kph / 3.6, 1.8, f.lead_status, **_accepted_kwargs(curr_ramp_fn, ramp_kwargs))
      if abs(r_b[0] - r_c[0]) > delta_threshold:
        bs.ramp_changed_frames += 1
        total_ramp_changed += 1

      delta = float(out_c.a_target - out_b.a_target)
      abs_d = abs(delta)
      if abs_d > bs.max_abs_delta:
        bs.max_abs_delta = abs_d
      if abs_d > overall_max_delta:
        overall_max_delta = abs_d

      if abs_d > delta_threshold:
        bs.changed_frames += 1
        bs.sum_delta += delta
        total_changed += 1
        if len(bs.samples) < 5:
          bs.samples.append({
            "segment": seg_name,
            "t": round(f.t, 2),
            "v_kph": round(f.v_kph, 1),
            "lead_d": round(f.lead_d, 1),
            "lead_vr": round(f.lead_v_rel, 1),
            "baseline_a": round(out_b.a_target, 3),
            "candidate_a": round(out_c.a_target, 3),
          })

  return DatasetComparison(
    label=dataset_label,
    segment_count=len(segments),
    active_frames=total_active,
    changed_frames=total_changed,
    ramp_changed_frames=total_ramp_changed,
    max_abs_delta=overall_max_delta,
    bands=bands,
  )


def _band_for_lateral(lat_mag: float) -> str:
  for name, lo, hi in LATERAL_BANDS:
    if lo <= lat_mag < hi:
      return name
  return LATERAL_BANDS[-1][0]


def _make_lane_lines(f: FrameSample):
  if f.lane_y_left is None or f.lane_y_right is None:
    return None, None
  ll = [
    types.SimpleNamespace(y=[0.0]),
    types.SimpleNamespace(y=[f.lane_y_left]),
    types.SimpleNamespace(y=[f.lane_y_right]),
    types.SimpleNamespace(y=[0.0]),
  ]
  probs = [0.0, f.lane_prob_left, f.lane_prob_right, 0.0]
  return ll, probs


def compare_lateral_frames(
  baseline_mod: Any,
  current_mod: Any,
  segments: Sequence[tuple[str, dict[str, Any], Sequence[FrameSample]]],
  *,
  dataset_label: str,
  curv_delta_threshold: float = 1e-4,
  smooth_delta_threshold: float = 0.005,
) -> DatasetComparison:
  bands = {name: BandStats(band=name) for name, _, _ in LATERAL_BANDS}
  total_active = 0
  total_changed = 0
  total_smooth_changed = 0
  overall_max_delta = 0.0

  b_curv_fn = baseline_mod.compute_curve_inner_curvature_assist
  c_curv_fn = current_mod.compute_curve_inner_curvature_assist
  b_smooth_fn = baseline_mod.compute_curve_entry_smooth_factor
  c_smooth_fn = current_mod.compute_curve_entry_smooth_factor

  for seg_name, meta, frames in segments:
    bias_percent = float(meta.get("curve_inner_bias", 40.0))
    for f in frames:
      if not f.lat_active or f.steering_pressed or f.blinker_active or f.lane_changing:
        continue
      if f.v_kph < 18.0 or len(f.model_t) < 10:
        continue
      v_ego = f.v_kph / 3.6
      ll, lprobs = _make_lane_lines(f)
      lat_mag = max(abs(f.lat_accel), abs(f.desired_lat_accel))
      band = _band_for_lateral(lat_mag)
      bs = bands[band]
      bs.active_frames += 1
      total_active += 1

      b_curv = float(b_curv_fn(
        f.desired_curvature, f.model_yaws, f.model_yaw_rates, f.model_t, v_ego, 0.15, bias_percent,
        **_accepted_kwargs(b_curv_fn, {"lane_lines": ll, "lane_line_probs": lprobs, "desire": 0, "lane_change_prob": 0.0}),
      ))
      c_curv = float(c_curv_fn(
        f.desired_curvature, f.model_yaws, f.model_yaw_rates, f.model_t, v_ego, 0.15, bias_percent,
        **_accepted_kwargs(c_curv_fn, {"lane_lines": ll, "lane_line_probs": lprobs, "desire": 0, "lane_change_prob": 0.0}),
      ))
      b_sf = float(b_smooth_fn(f.model_yaws, f.model_yaw_rates, f.model_t, v_ego, 0.15, bias_percent))
      c_sf = float(c_smooth_fn(f.model_yaws, f.model_yaw_rates, f.model_t, v_ego, 0.15, bias_percent))

      d_curv = c_curv - b_curv
      d_lat_a = d_curv * (v_ego ** 2)
      abs_lat_a = abs(d_lat_a)
      if abs_lat_a > bs.max_abs_delta:
        bs.max_abs_delta = abs_lat_a
      if abs_lat_a > overall_max_delta:
        overall_max_delta = abs_lat_a

      smooth_changed = abs(c_sf - b_sf) > smooth_delta_threshold
      if smooth_changed:
        bs.ramp_changed_frames += 1
        total_smooth_changed += 1

      if abs(d_curv) > curv_delta_threshold or smooth_changed:
        bs.changed_frames += 1
        bs.sum_delta += d_lat_a
        total_changed += 1
        if len(bs.samples) < 5:
          bs.samples.append({
            "segment": seg_name,
            "t": round(f.t, 2),
            "v_kph": round(f.v_kph, 1),
            "lat_accel": round(f.lat_accel, 2),
            "d_lat_accel": round(d_lat_a, 3),
            "d_smooth": round(c_sf - b_sf, 3),
          })

  return DatasetComparison(
    label=dataset_label,
    domain="lat",
    segment_count=len(segments),
    active_frames=total_active,
    changed_frames=total_changed,
    ramp_changed_frames=total_smooth_changed,
    max_abs_delta=overall_max_delta,
    bands=bands,
  )


def evaluate_regression_gate(
  golden: DatasetComparison | None,
  *,
  forbid_bands: Sequence[str] = (),
  max_golden_change_pct: float = 0.5,
  max_golden_abs_delta: float = 0.15,
) -> tuple[bool, list[str]]:
  if golden is None or golden.active_frames == 0:
    return True, []

  kind_label = "구간대" if golden.domain == "lat" else "속도대"
  domain_tag = "[조향] " if golden.domain == "lat" else "[종방향] "
  reasons: list[str] = []
  for band_name in forbid_bands:
    bs = golden.bands.get(band_name)
    if bs and (bs.changed_frames > 0 or bs.ramp_changed_frames > 0):
      reasons.append(
        f"{domain_tag}금지 {kind_label} '{band_name}'에서 변경 발생 (changed={bs.changed_frames}, sub={bs.ramp_changed_frames}, max|Δ|={bs.max_abs_delta:.3f})"
      )

  if golden.changed_pct > max_golden_change_pct:
    reasons.append(
      f"{domain_tag}골든 코퍼스 전체 변경률 {golden.changed_pct:.2f}% > 허용 한도 {max_golden_change_pct:.2f}%"
    )
  if golden.max_abs_delta > max_golden_abs_delta and golden.changed_frames > 0:
    reasons.append(
      f"{domain_tag}골든 코퍼스 최대 출력차 {golden.max_abs_delta:.3f} > 허용 한도 {max_golden_abs_delta:.3f}"
    )
  return (len(reasons) == 0), reasons


def _load_dataset(paths: Sequence[Path | str]) -> list[tuple[str, dict[str, Any], list[FrameSample]]]:
  rlogs = discover_rlogs(paths)
  loaded: list[tuple[str, dict[str, Any], list[FrameSample]]] = []
  for p in rlogs:
    meta, frames = load_segment_frames(p)
    loaded.append((p.name, meta, frames))
  return loaded


def run_regression(
  *,
  baseline_ref: str = DEFAULT_BASELINE_REF,
  target_paths: Sequence[Path | str] = (),
  golden_paths: Sequence[Path | str] = (),
  mode: str = "both",
  forbid_bands: Sequence[str] = ("0-25",),
  forbid_lat_bands: Sequence[str] = ("straight",),
  max_golden_change_pct: float = 0.5,
  max_golden_abs_delta: float = 0.15,
  candidate_mod: Any = None,
  candidate_lat_mod: Any = None,
) -> RegressionReport:
  datasets: list[DatasetComparison] = []
  reasons: list[str] = []
  passed = True

  target_segs = _load_dataset(target_paths) if target_paths else []
  golden_segs = _load_dataset(golden_paths) if golden_paths else []

  if mode in ("both", "long"):
    baseline_long = load_shaper_from_git_ref(baseline_ref)
    if candidate_mod is None:
      import openpilot.selfdrive.controls.lib.human_longitudinal as curr_long_mod
      candidate_mod = curr_long_mod
    golden_long: DatasetComparison | None = None
    if target_segs:
      datasets.append(compare_segment_frames(baseline_long, candidate_mod, target_segs, dataset_label="target"))
    if golden_segs:
      golden_long = compare_segment_frames(baseline_long, candidate_mod, golden_segs, dataset_label="golden")
      datasets.append(golden_long)
    ok_long, r_long = evaluate_regression_gate(
      golden_long,
      forbid_bands=forbid_bands,
      max_golden_change_pct=max_golden_change_pct,
      max_golden_abs_delta=max_golden_abs_delta,
    )
    passed = passed and ok_long
    reasons.extend(r_long)

  if mode in ("both", "lat"):
    baseline_lat = load_drive_helpers_from_git_ref(baseline_ref)
    if candidate_lat_mod is None:
      import openpilot.selfdrive.controls.lib.drive_helpers as curr_lat_mod
      candidate_lat_mod = curr_lat_mod
    golden_lat: DatasetComparison | None = None
    if target_segs:
      datasets.append(compare_lateral_frames(baseline_lat, candidate_lat_mod, target_segs, dataset_label="target"))
    if golden_segs:
      golden_lat = compare_lateral_frames(baseline_lat, candidate_lat_mod, golden_segs, dataset_label="golden")
      datasets.append(golden_lat)
    ok_lat, r_lat = evaluate_regression_gate(
      golden_lat,
      forbid_bands=forbid_lat_bands,
      max_golden_change_pct=max_golden_change_pct,
      max_golden_abs_delta=max_golden_abs_delta,
    )
    passed = passed and ok_lat
    reasons.extend(r_lat)

  return RegressionReport(
    baseline_ref=baseline_ref,
    passed=passed,
    reasons=reasons,
    datasets=datasets,
  )


def render_regression_markdown(report: RegressionReport) -> str:
  status = "PASS" if report.passed else "FAIL"
  lines = [
    f"# `{report.baseline_ref}` 대비 A/B 회귀 검증 ({status})",
    "",
  ]
  if report.reasons:
    lines.append("## 실패 사유")
    for r in report.reasons:
      lines.append(f"- {r}")
    lines.append("")

  for ds in report.datasets:
    if ds.domain == "lat":
      lines.append(
        f"## {ds.label} [조향] ({ds.segment_count}세그먼트 / 활성 {ds.active_frames}프레임 / "
        f"변경 {ds.changed_frames}프레임 ({ds.changed_pct:.2f}%), max|Δlat_a|={ds.max_abs_delta:.3f})"
      )
      lines.append("| 조향 구간 (횡가속) | 활성 프레임 | 곡률 변경 | smooth 변경 | 변경률(%) | 최대 |Δlat_a| | 평균 Δlat_a |")
      lines.append("|---|---:|---:|---:|---:|---:|---:|")
      band_defs = LATERAL_BANDS
    else:
      lines.append(
        f"## {ds.label} [종방향] ({ds.segment_count}세그먼트 / 활성 {ds.active_frames}프레임 / "
        f"변경 {ds.changed_frames}프레임 ({ds.changed_pct:.2f}%), max|Δ|={ds.max_abs_delta:.3f})"
      )
      lines.append("| 속도대 (km/h) | 활성 프레임 | shape 변경 | ramp 변경 | 변경률(%) | 최대 |Δa| | 평균 Δa |")
      lines.append("|---|---:|---:|---:|---:|---:|---:|")
      band_defs = SPEED_BANDS
    for name, _, _ in band_defs:
      bs = ds.bands[name]
      lines.append(
        f"| {name} | {bs.active_frames} | {bs.changed_frames} | {bs.ramp_changed_frames} | "
        f"{bs.changed_pct:.2f}% | {bs.max_abs_delta:.3f} | {bs.mean_delta:+.3f} |"
      )
    lines.append("")
  return "\n".join(lines)

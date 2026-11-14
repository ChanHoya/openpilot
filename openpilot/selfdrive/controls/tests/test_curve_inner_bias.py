import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.drive_helpers import (
  compute_curve_entry_smooth_factor,
  compute_curve_inner_curvature_assist,
  read_curve_inner_bias,
)
from openpilot.selfdrive.controls.lib.lane_planner_2 import LanePlanner
from openpilot.selfdrive.modeld.constants import ModelConstants, Plan


def _make_trajectory(k_now: float, k_preview: float, v_ego: float):
  t_idxs = np.asarray(ModelConstants.T_IDXS, dtype=float)
  blend = np.clip((t_idxs - 0.35) / 0.35, 0.0, 1.0)
  k_profile = k_now + (k_preview - k_now) * blend
  yaw_rates = k_profile * v_ego
  yaws = np.zeros_like(t_idxs)
  for i in range(1, len(t_idxs)):
    dt = t_idxs[i] - t_idxs[i - 1]
    yaws[i] = yaws[i - 1] + 0.5 * (yaw_rates[i - 1] + yaw_rates[i]) * dt
  return yaws, yaw_rates, t_idxs


def _make_lane_lines(y_center: float, width: float = 3.2):
  n = ModelConstants.IDX_N
  half_w = 0.5 * width
  lines = np.zeros((4, n), dtype=float)
  lines[1, :] = y_center - half_w
  lines[2, :] = y_center + half_w
  probs = np.array([0.1, 0.9, 0.9, 0.1], dtype=float)
  return lines, probs


def test_bias_zero_is_identity():
  v_ego = 18.0
  yaws, yaw_rates, t_idxs = _make_trajectory(0.004, 0.012, v_ego)
  lines, probs = _make_lane_lines(y_center=0.35)
  assert compute_curve_entry_smooth_factor(yaws, yaw_rates, t_idxs, v_ego, 0.3, 0.0) == 1.0
  assert compute_curve_inner_curvature_assist(
    0.004, yaws, yaw_rates, t_idxs, v_ego, 0.3, 0.0,
    lane_lines=lines, lane_line_probs=probs,
  ) == pytest.approx(0.004)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_sharp_curve_entry_guides_inner_and_prevents_outer_drift(sign):
  v_ego = 16.0
  base_curv = sign * 0.004
  preview_curv = sign * 0.011
  yaws, yaw_rates, t_idxs = _make_trajectory(base_curv, preview_curv, v_ego)

  centered_lines, probs = _make_lane_lines(y_center=0.0)
  # Drifted toward outside of curve (left in right curve => +y_center; right in left curve => -y_center)
  outer_lines, _ = _make_lane_lines(y_center=sign * 0.30)

  curv_30 = compute_curve_inner_curvature_assist(
    base_curv, yaws, yaw_rates, t_idxs, v_ego, 0.3, 30.0,
    lane_lines=centered_lines, lane_line_probs=probs,
  )
  curv_80 = compute_curve_inner_curvature_assist(
    base_curv, yaws, yaw_rates, t_idxs, v_ego, 0.3, 80.0,
    lane_lines=centered_lines, lane_line_probs=probs,
  )
  curv_outer = compute_curve_inner_curvature_assist(
    base_curv, yaws, yaw_rates, t_idxs, v_ego, 0.3, 30.0,
    lane_lines=outer_lines, lane_line_probs=probs,
  )

  # Must pull strictly further into the curve than base_curv, scaling smoothly with strength
  assert sign * (curv_30 - base_curv) > 0.0
  assert sign * (curv_80 - curv_30) > 0.0
  # Drifting toward the outside must trigger stronger inner recovery than when centered
  assert sign * (curv_outer - curv_30) > 0.0


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_inner_overshoot_is_damped_at_apex_exit(sign):
  v_ego = 16.0
  base_curv = sign * 0.010
  # Apex/exit: preview curvature already easing compared to current curvature
  preview_curv = sign * 0.008
  yaws, yaw_rates, t_idxs = _make_trajectory(base_curv, preview_curv, v_ego)

  # Already cut well past target inner offset toward the inner lane line
  too_inside_lines, probs = _make_lane_lines(y_center=-sign * 0.40)
  curv_damped = compute_curve_inner_curvature_assist(
    base_curv, yaws, yaw_rates, t_idxs, v_ego, 0.3, 50.0,
    lane_lines=too_inside_lines, lane_line_probs=probs,
  )
  # Should gently ease curvature (less magnitude toward inside) rather than pushing harder
  assert sign * (curv_damped - base_curv) < 0.0


def test_suppressed_on_straight_low_speed_and_lane_change():
  v_ego = 18.0
  yaws_s, rates_s, t_idxs = _make_trajectory(0.0002, 0.0003, v_ego)
  lines_offset, probs = _make_lane_lines(y_center=0.25)
  assert compute_curve_inner_curvature_assist(
    0.0002, yaws_s, rates_s, t_idxs, v_ego, 0.3, 50.0,
    lane_lines=lines_offset, lane_line_probs=probs,
  ) == pytest.approx(0.0002)

  yaws_c, rates_c, t_idxs = _make_trajectory(0.005, 0.012, v_ego)
  assert compute_curve_inner_curvature_assist(
    0.005, yaws_c, rates_c, t_idxs, 4.0, 0.3, 50.0,
    lane_lines=lines_offset, lane_line_probs=probs,
  ) == pytest.approx(0.005)
  assert compute_curve_inner_curvature_assist(
    0.005, yaws_c, rates_c, t_idxs, v_ego, 0.3, 50.0,
    lane_lines=lines_offset, lane_line_probs=probs, desire=1,
  ) == pytest.approx(0.005)
  assert compute_curve_inner_curvature_assist(
    0.005, yaws_c, rates_c, t_idxs, v_ego, 0.3, 50.0,
    lane_lines=lines_offset, lane_line_probs=probs, lane_change_prob=0.5,
  ) == pytest.approx(0.005)


def test_entry_smooth_factor_reduces_lag_on_entry_only():
  v_ego = 18.0
  yaws_in, rates_in, t_idxs = _make_trajectory(0.003, 0.011, v_ego)
  f30 = compute_curve_entry_smooth_factor(yaws_in, rates_in, t_idxs, v_ego, 0.3, 30.0)
  f80 = compute_curve_entry_smooth_factor(yaws_in, rates_in, t_idxs, v_ego, 0.3, 80.0)
  assert 0.55 <= f80 < f30 < 1.0

  yaws_out, rates_out, _ = _make_trajectory(0.011, 0.005, v_ego)
  assert compute_curve_entry_smooth_factor(yaws_out, rates_out, t_idxs, v_ego, 0.3, 80.0) == 1.0


def test_read_curve_inner_bias_fallback_and_clamping(tmp_path):
  class UnknownKey(Exception):
    pass

  (tmp_path / "CurveInnerBias").write_text("45\n", encoding="utf-8")

  unrebuilt = NS(
    get_int=lambda k: (_ for _ in ()).throw(UnknownKey(k)),
    get_param_path=lambda: str(tmp_path),
  )
  assert read_curve_inner_bias(unrebuilt) == 45.0

  clamped_hi = NS(get_int=lambda k: 180)
  clamped_lo = NS(get_int=lambda k: -20)
  assert read_curve_inner_bias(clamped_hi) == 100.0
  assert read_curve_inner_bias(clamped_lo) == 0.0

  missing = NS(get_int=lambda k: (_ for _ in ()).throw(UnknownKey(k)))
  assert read_curve_inner_bias(missing) == 30.0


def test_lane_planner_curve_inner_bias_direction_and_responsiveness():
  def run_steps(bias_val: int, curve_speed: float, steps: int = 8) -> float:
    lp = LanePlanner()
    lp.params = NS(get_int=lambda k: bias_val if k == "CurveInnerBias" else 0, get_float=lambda k: 0.0)
    lp.lll_y = np.full(ModelConstants.IDX_N, -1.6)
    lp.rll_y = np.full(ModelConstants.IDX_N, 1.6)
    lp.lll_prob = 0.9
    lp.rll_prob = 0.9
    lp.lll_std = 0.05
    lp.rll_std = 0.05
    lp.lane_width_left = 3.2
    lp.lane_width_right = 3.2
    lp.lanefull_mode = False
    path_t = np.asarray(ModelConstants.T_IDXS, dtype=float)
    out = None
    for _ in range(steps):
      path_xyz = np.zeros((ModelConstants.IDX_N, 3), dtype=float)
      path_xyz[:, 0] = ModelConstants.X_IDXS
      out, _ = lp.get_d_path(NS(), 16.0, path_t, path_xyz, curve_speed)
    return float(out[0, 1])

  assert run_steps(0, 60.0) == pytest.approx(0.0)
  assert run_steps(50, 200.0) == pytest.approx(0.0)
  right_y = run_steps(50, 60.0)
  left_y = run_steps(50, -60.0)
  assert right_y > 0.08
  assert left_y < -0.08
  assert math.isclose(right_y, -left_y, rel_tol=1e-6)


def test_modeld_action_and_smoothing_respect_curve_inner_bias():
  modeld_path = Path(__file__).parents[2] / "modeld/modeld.py"
  helpers_path = Path(__file__).parents[1] / "lib/drive_helpers.py"
  ns = {
    "os": __import__("os"),
    "math": math,
    "np": np,
    "ModelConstants": ModelConstants,
    "Plan": Plan,
    "log": NS(ModelDataV2=NS(Action=NS)),
    "DT_MDL": 0.05,
    "LONG_SMOOTH_SECONDS": 0.3,
    "MIN_LAT_CONTROL_SPEED": 0.3,
    "MIN_SPEED": 1.0,
  }
  needed = {
    "CURVE_INNER_BIAS_KEY", "CURVE_INNER_BIAS_DEFAULT",
    "get_accel_from_plan", "smooth_value", "curv_from_psis", "get_curvature_from_plan",
    "read_curve_inner_bias", "_preview_curvature_from_plan", "_near_lane_y", "_lane_center_and_width",
    "compute_curve_entry_smooth_factor", "compute_curve_inner_curvature_assist",
    "get_lat_smooth_seconds_dynamic", "get_action_from_model",
  }
  for src in (helpers_path, modeld_path):
    tree = ast.parse(src.read_text(encoding="utf-8"))
    body = [
      node for node in tree.body
      if (isinstance(node, ast.FunctionDef) and node.name in needed) or
      (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in needed for t in node.targets))
    ]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(src), "exec"), ns)

  v_ego = 16.0
  yaws, yaw_rates, _ = _make_trajectory(0.004, 0.012, v_ego)
  plan = np.zeros((1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH))
  plan[0, :, Plan.VELOCITY][:, 0] = v_ego
  plan[0, :, Plan.T_FROM_CURRENT_EULER][:, 2] = yaws
  plan[0, :, Plan.ORIENTATION_RATE][:, 2] = yaw_rates
  plan_stds = np.zeros((1, ModelConstants.IDX_N, ModelConstants.PLAN_WIDTH, 2))
  plan_stds[0, 10, Plan.POSITION, 1] = 0.25

  lane_lines = np.zeros((1, 4, ModelConstants.IDX_N, 2), dtype=float)
  lane_lines[0, 1, :, 0] = -1.35
  lane_lines[0, 2, :, 0] = 1.85
  lane_probs = np.zeros((1, 8), dtype=float)
  lane_probs[0, 1::2] = [0.1, 0.9, 0.9, 0.1]

  output = {
    "plan": plan,
    "plan_stds": plan_stds,
    "lane_lines": lane_lines,
    "lane_lines_prob": lane_probs,
  }
  prev = NS(desiredCurvature=0.004, desiredAcceleration=0.0, desiredVelocity=v_ego)

  s0, _, _ = ns["get_lat_smooth_seconds_dynamic"](output, 0.13, v_ego=v_ego, lat_action_t=0.3, curve_inner_bias=0.0)
  s30, _, _ = ns["get_lat_smooth_seconds_dynamic"](output, 0.13, v_ego=v_ego, lat_action_t=0.3, curve_inner_bias=30.0)
  s80, _, _ = ns["get_lat_smooth_seconds_dynamic"](output, 0.13, v_ego=v_ego, lat_action_t=0.3, curve_inner_bias=80.0)
  assert s80 < s30 < s0

  a0 = ns["get_action_from_model"](output, prev, 0.3, 0.4, v_ego, s0, 0.05, curve_inner_bias=0.0)
  a30 = ns["get_action_from_model"](output, prev, 0.3, 0.4, v_ego, s30, 0.05, curve_inner_bias=30.0)
  a80 = ns["get_action_from_model"](output, prev, 0.3, 0.4, v_ego, s80, 0.05, curve_inner_bias=80.0)
  assert a80.desiredCurvature > a30.desiredCurvature > a0.desiredCurvature

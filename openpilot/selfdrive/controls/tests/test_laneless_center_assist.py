from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.drive_helpers import (
  compute_laneless_center_assist,
  read_laneless_center_adjust,
)
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.modeld import get_action_from_model


def _make_lane_lines(y_center: float, width: float = 3.2):
  n = ModelConstants.IDX_N
  half_w = 0.5 * width
  lines = np.zeros((4, n), dtype=float)
  lines[1, :] = y_center - half_w
  lines[2, :] = y_center + half_w
  probs = np.array([0.1, 0.9, 0.9, 0.1], dtype=float)
  return lines, probs


def test_adjust_zero_is_identity():
  v_ego = 20.0
  base_curv = 0.0002
  lines, probs = _make_lane_lines(y_center=0.30)
  assert compute_laneless_center_assist(
    base_curv, v_ego, 0.0,
    lane_lines=lines, lane_line_probs=probs,
  ) == pytest.approx(base_curv)


def test_deadzone_preserves_base_curvature():
  v_ego = 20.0
  base_curv = 0.0001
  # Inside 12cm deadzone: +0.08m and -0.08m
  for y_c in [0.0, 0.05, -0.05, 0.11, -0.11]:
    lines, probs = _make_lane_lines(y_center=y_c)
    assert compute_laneless_center_assist(
      base_curv, v_ego, 10.0,
      lane_lines=lines, lane_line_probs=probs,
    ) == pytest.approx(base_curv)


def test_left_drift_steers_right():
  v_ego = 20.0
  base_curv = 0.0
  # Vehicle biased to the left -> lane center is to the right (+y_center)
  lines, probs = _make_lane_lines(y_center=0.25)
  res = compute_laneless_center_assist(
    base_curv, v_ego, 10.0,
    lane_lines=lines, lane_line_probs=probs,
  )
  # Must command positive curvature (steer right toward center)
  assert res > base_curv


def test_right_drift_steers_left():
  v_ego = 20.0
  base_curv = 0.0
  # Vehicle biased to the right -> lane center is to the left (-y_center)
  lines, probs = _make_lane_lines(y_center=-0.25)
  res = compute_laneless_center_assist(
    base_curv, v_ego, 10.0,
    lane_lines=lines, lane_line_probs=probs,
  )
  # Must command negative curvature (steer left toward center)
  assert res < base_curv


def test_scaling_with_adjust_percent():
  v_ego = 20.0
  base_curv = 0.0
  lines, probs = _make_lane_lines(y_center=0.30)
  curv_10 = compute_laneless_center_assist(
    base_curv, v_ego, 10.0,
    lane_lines=lines, lane_line_probs=probs,
  )
  curv_20 = compute_laneless_center_assist(
    base_curv, v_ego, 20.0,
    lane_lines=lines, lane_line_probs=probs,
  )
  assert curv_10 > base_curv
  assert curv_20 > curv_10


def test_suppressed_on_sharp_curve():
  v_ego = 20.0
  # Sharp curve: curv_mag = 0.005 (radius 200m)
  sharp_curv = 0.005
  lines, probs = _make_lane_lines(y_center=0.30)
  res = compute_laneless_center_assist(
    sharp_curv, v_ego, 10.0,
    lane_lines=lines, lane_line_probs=probs,
  )
  assert res == pytest.approx(sharp_curv)


def test_suppressed_during_lane_change():
  v_ego = 20.0
  base_curv = 0.0
  lines, probs = _make_lane_lines(y_center=0.30)
  # Active desire
  res_desire = compute_laneless_center_assist(
    base_curv, v_ego, 10.0,
    lane_lines=lines, lane_line_probs=probs,
    desire=int(log.Desire.laneChangeRight),
  )
  assert res_desire == pytest.approx(base_curv)

  # High lane change probability
  res_prob = compute_laneless_center_assist(
    base_curv, v_ego, 10.0,
    lane_lines=lines, lane_line_probs=probs,
    lane_change_prob=0.35,
  )
  assert res_prob == pytest.approx(base_curv)


def test_suppressed_at_low_speed():
  base_curv = 0.0
  lines, probs = _make_lane_lines(y_center=0.30)
  res_slow = compute_laneless_center_assist(
    base_curv, 4.0, 10.0,
    lane_lines=lines, lane_line_probs=probs,
  )
  assert res_slow == pytest.approx(base_curv)


def test_suppressed_on_low_confidence_lanes():
  v_ego = 20.0
  base_curv = 0.0
  lines, _ = _make_lane_lines(y_center=0.30)
  # Low probability lines
  low_probs = np.array([0.05, 0.1, 0.1, 0.05], dtype=float)
  res = compute_laneless_center_assist(
    base_curv, v_ego, 10.0,
    lane_lines=lines, lane_line_probs=low_probs,
  )
  assert res == pytest.approx(base_curv)


def test_read_laneless_center_adjust(tmp_path: Path):
  # Default when None
  assert read_laneless_center_adjust(None) == 10.0

  # Get int from params
  params_int = NS(get_int=lambda k: 15 if k == "LanelessCenterAdjust" else 0)
  assert read_laneless_center_adjust(params_int) == 15.0

  # Fallback to param path file
  (tmp_path / "LanelessCenterAdjust").write_text("20\n", encoding="utf-8")
  params_file = NS(
    get_int=lambda _k: (_ for _ in ()).throw(RuntimeError("no Cython key")),
    get_param_path=lambda: str(tmp_path),
  )
  assert read_laneless_center_adjust(params_file) == 20.0

  # Fallback to default on invalid content
  (tmp_path / "LanelessCenterAdjust").write_text("invalid_val\n", encoding="utf-8")
  assert read_laneless_center_adjust(params_file) == 10.0


def test_modeld_get_action_integration():
  v_ego = 20.0
  prev_action = log.ModelDataV2.Action(
    desiredCurvature=0.0,
    desiredAcceleration=0.0,
    shouldStop=False,
    desiredVelocity=20.0,
  )
  plan = np.zeros((ModelConstants.IDX_N, 15), dtype=np.float32)
  lines, probs = _make_lane_lines(y_center=0.30)
  model_output = {
    'plan': plan.reshape(1, ModelConstants.IDX_N, 15),
    'action': np.zeros((1, 2), dtype=np.float32),
    'lane_lines': lines.reshape(1, 4, ModelConstants.IDX_N, 1),
    'lane_lines_prob': np.repeat(probs.reshape(1, 4), 2, axis=1),
  }

  action_adj_0 = get_action_from_model(
    model_output, prev_action, 0.3, 0.2, v_ego, 0.0, 0.3,
    laneless_center_adjust=0.0,
  )
  action_adj_10 = get_action_from_model(
    model_output, prev_action, 0.3, 0.2, v_ego, 0.0, 0.3,
    laneless_center_adjust=10.0,
  )

  # With 10% laneless center adjust and vehicle to the left (y_center=+0.30),
  # desiredCurvature must be positive to steer right toward center
  assert action_adj_0.desiredCurvature == pytest.approx(0.0)
  assert action_adj_10.desiredCurvature > 0.0

"""Third-person VR Mode driving scene inspired by carrot-vision3d v2.3.

Renders an elevated chase-camera view with:
- 3-tier sky / ground / asphalt road surface (day & night palettes) + moving road seams
- 4-lane hysteresis-gated lane lines + road edges + driving path ribbon
- Ego Genesis-style sedan sprite with soft shadow, brake glow, blinkers, and blindspot floor zones
- Stabilized L1/L2 lead vehicles with distance pills & road chevron + radar-fused adjacent cars
"""
from dataclasses import dataclass
import math
import time

import numpy as np
import pyray as rl

from openpilot.selfdrive.ui.onroad.view_3d_camera import ChaseCamera, CameraPose
from openpilot.selfdrive.ui.onroad.view_3d_scene import View3dScene, C3_SIZES, C4_SIZES
from openpilot.selfdrive.ui.onroad import view_3d_draw
from openpilot.selfdrive.ui.onroad.view_vr_tracks import VrTrackManager, VrVehicle
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.shader_polygon import draw_polygon_solid
from openpilot.system.ui.lib.text_measure import measure_text_cached


# Elevated 3rd-person chase perspective matching carrot-vision3d framing
VR_CHASE_POSE = CameraPose(
  back_m=10.8,
  height_m=4.15,
  pitch_deg=9.5,
  focal_scale=0.58,
  horizon_frac=0.35,
)

# Color palettes modeled directly on carrot-vision3d v2.3 THEMES
VR_PALETTE = {
  True: {  # Night (dark)
    'sky_top': (10, 13, 17, 255),
    'sky_horizon': (14, 18, 24, 255),
    'ground': (17, 21, 26, 255),
    'road': (28, 32, 38, 255),
    'road_seam': (40, 46, 56, 120),
    'lane_ego': (228, 232, 236, 245),
    'lane_adj': (200, 208, 218, 180),
    'edge': (138, 91, 87, 225),
    'path_engaged': (48, 128, 255, 165),
    'path_idle': (141, 150, 163, 115),
    'accent': (48, 128, 255, 255),
    'warn': (255, 159, 10, 255),
    'danger': (255, 69, 58, 255),
    'shadow_alpha': 0.55,
  },
  False: {  # Day (light)
    'sky_top': (236, 239, 243, 255),
    'sky_horizon': (226, 230, 236, 255),
    'ground': (214, 220, 226, 255),
    'road': (201, 207, 214, 255),
    'road_seam': (176, 184, 194, 110),
    'lane_ego': (255, 255, 255, 250),
    'lane_adj': (248, 250, 253, 195),
    'edge': (183, 131, 126, 230),
    'path_engaged': (37, 99, 235, 145),
    'path_idle': (124, 135, 148, 105),
    'accent': (37, 99, 235, 255),
    'warn': (205, 162, 112, 255),
    'danger': (215, 0, 21, 255),
    'shadow_alpha': 0.34,
  },
}

VR_MIN_DRAW_M = 12.0
VR_MAX_DRAW_M = 120.0
VR_LANE_BEHIND_M = 9.0
VR_ROAD_FALLBACK_MARGIN_M = 1.35
VR_SEAM_SPACING_M = 12.0


def _ema_alpha(dt: float, tau: float) -> float:
  if tau <= 0.0:
    return 1.0
  dt_eff = dt if dt > 0.0 else 0.05
  return float(1.0 - math.exp(-dt_eff / tau))


def _c(rgba, alpha: float = 1.0) -> rl.Color:
  return rl.Color(int(rgba[0]), int(rgba[1]), int(rgba[2]), int(np.clip(rgba[3] * alpha, 0, 255)))


def draw_vr_background(rect: rl.Rectangle, is_night: bool, horizon_frac: float = VR_CHASE_POSE.horizon_frac) -> None:
  """Render clean 3-tier sky + soft horizon fog + ground backdrop for VR Mode."""
  pal = VR_PALETTE[is_night]
  road_y = int(np.clip(rect.y + rect.height * horizon_frac, rect.y + rect.height * 0.25, rect.y + rect.height * 0.55))
  sky_h = max(1, int(road_y - rect.y))
  ground_h = max(1, int(rect.height - sky_h))

  rl.draw_rectangle_gradient_v(
    int(rect.x), int(rect.y), int(rect.width), sky_h,
    _c(pal['sky_top']), _c(pal['sky_horizon']),
  )
  rl.draw_rectangle_gradient_v(
    int(rect.x), road_y, int(rect.width), ground_h,
    _c(pal['sky_horizon']), _c(pal['ground']),
  )


@dataclass
class _LaneGateState:
  on: list[bool]
  level: list[float]
  ready: bool = False


class ViewVrScene:
  """Self-contained 3rd-person VR scene shared by C3 and C4 renderers."""

  def __init__(self, is_compact: bool = False):
    self._is_compact = is_compact
    self.camera = ChaseCamera()
    self.tracks = VrTrackManager()
    self._last_t: float | None = None
    self._smooth_ground_z: float | None = None
    self._smooth_horizon_frac: float = VR_CHASE_POSE.horizon_frac
    self.horizon_frac: float = VR_CHASE_POSE.horizon_frac
    self.horizon_y: float = 0.0
    self._travel_m: float = 0.0
    self._lane_gate = _LaneGateState(on=[False, False, False, False], level=[0.0, 0.0, 0.0, 0.0])
    self._smooth_path_y: np.ndarray | None = None
    self._smooth_path_z: np.ndarray | None = None
    self._smooth_path_max: float | None = None
    self._brake_level: float = 0.0
    self._bs_level: dict[str, float] = {'left': 0.0, 'right': 0.0}
    self._blink_since: float = 0.0
    self._tl_scene: View3dScene | None = None

  def _dt(self, now: float | None) -> tuple[float, float]:
    now_t = time.monotonic() if now is None else float(now)
    dt = 0.0 if self._last_t is None else float(np.clip(now_t - self._last_t, 0.0, 0.5))
    self._last_t = now_t
    return dt, now_t

  def _update_lane_gate(self, lanes: list, probs: np.ndarray, road_edges: list | None,
                        road_edge_stds: np.ndarray | None, dt: float) -> list[float]:
    """Hysteresis lane gate ported from carrot-vision3d (prevents adjacent lane flicker)."""
    def _ref_y(pts: np.ndarray | None) -> float | None:
      if pts is None or pts.shape[0] < 2:
        return None
      return 0.5 * (float(np.interp(8.0, pts[:, 0], pts[:, 1])) + float(np.interp(28.0, pts[:, 0], pts[:, 1])))

    ys = [_ref_y(lanes[i].raw_points) if i < len(lanes) else None for i in range(4)]
    center_y = 0.5 * (ys[1] + ys[2]) if (ys[1] is not None and ys[2] is not None) else 0.0

    edge_left = edge_right = None
    if road_edges:
      for i, edge in enumerate(road_edges[:2]):
        std = float(road_edge_stds[i]) if road_edge_stds is not None and i < len(road_edge_stds) else 1.0
        if std > 1.8:
          continue
        ey = _ref_y(getattr(edge, 'raw_points', None))
        if ey is None:
          continue
        if ey < center_y:
          edge_left = ey if edge_left is None else max(edge_left, ey)
        else:
          edge_right = ey if edge_right is None else min(edge_right, ey)

    def _want(i: int, is_ego: bool) -> bool:
      if ys[i] is None or i >= len(probs):
        return False
      p = float(probs[i])
      show_p = 0.35 if is_ego else 0.50
      hide_p = 0.20 if is_ego else 0.30
      return p >= (hide_p if self._lane_gate.on[i] else show_p)

    want = [False, False, False, False]
    want[1] = _want(1, True) and not (edge_left is not None and ys[1] < edge_left - 0.5)
    want[2] = _want(2, True) and not (edge_right is not None and ys[2] > edge_right + 0.5)
    if want[1] and _want(0, False) and ys[0] is not None and ys[1] is not None:
      w01 = abs(ys[1] - ys[0])
      want[0] = (2.3 <= w01 <= 4.8) and not (edge_left is not None and ys[0] < edge_left + 0.45)
    if want[2] and _want(3, False) and ys[3] is not None and ys[2] is not None:
      w23 = abs(ys[3] - ys[2])
      want[3] = (2.3 <= w23 <= 4.8) and not (edge_right is not None and ys[3] > edge_right - 0.45)

    step = (dt / 0.35) if (self._lane_gate.ready and dt > 0.0) else 1.0
    for i in range(4):
      self._lane_gate.on[i] = want[i]
      target = 1.0 if want[i] else 0.0
      diff = target - self._lane_gate.level[i]
      self._lane_gate.level[i] += math.copysign(min(step, abs(diff)), diff) if abs(diff) > 1e-4 else diff
    self._lane_gate.ready = True
    return self._lane_gate.level

  def _extend_line_behind(self, pts: np.ndarray, max_dist: float) -> np.ndarray:
    """Resample a 3D road curve from -VR_LANE_BEHIND_M to max_dist for smooth 3rd-person framing."""
    xs_src = pts[:, 0].astype(np.float64)
    ys_src = pts[:, 1].astype(np.float64)
    zs_src = pts[:, 2].astype(np.float64)
    if xs_src.size < 2:
      return np.empty((0, 3), dtype=np.float64)
    end_x = min(max_dist, float(xs_src[-1]))
    if end_x <= 1.0:
      return np.empty((0, 3), dtype=np.float64)
    xs = np.concatenate((
      np.linspace(-VR_LANE_BEHIND_M, 0.0, 5, endpoint=False),
      np.linspace(0.0, min(30.0, end_x), 14, endpoint=False),
      np.linspace(min(30.0, end_x), end_x, 12),
    ))
    k_slope = min(3, xs_src.size - 1)
    slope_y = (ys_src[k_slope] - ys_src[0]) / max(0.01, xs_src[k_slope] - xs_src[0])
    ys = np.interp(xs, xs_src, ys_src)
    behind = xs < xs_src[0]
    ys[behind] = ys_src[0] + slope_y * (xs[behind] - xs_src[0])
    zs = np.interp(xs, xs_src, zs_src)
    zs[behind] = zs_src[0]
    return np.stack((xs, ys, zs), axis=1)

  def _project_strip(self, left_pts: np.ndarray, right_pts: np.ndarray) -> np.ndarray:
    if left_pts.shape[0] < 2 or right_pts.shape[0] < 2:
      return np.empty((0, 2), dtype=np.float32)
    l_xy, l_d = self.camera.project(left_pts)
    r_xy, r_d = self.camera.project(right_pts)
    margin = 0.7 * self.camera._w
    ok = (
      (l_d > 0.4) & (r_d > 0.4)
      & np.isfinite(l_xy).all(axis=1) & np.isfinite(r_xy).all(axis=1)
      & (l_xy[:, 1] >= self.camera._y0 - margin) & (l_xy[:, 1] <= self.camera._y0 + self.camera._h + margin)
    )
    l_xy, r_xy = l_xy[ok], r_xy[ok]
    if len(l_xy) < 2:
      return np.empty((0, 2), dtype=np.float32)
    return np.concatenate((l_xy, r_xy[::-1])).astype(np.float32)

  def _draw_ribbon_3d(self, curve: np.ndarray, half_w: float, z_off: float, color: rl.Color) -> None:
    if curve.shape[0] < 2:
      return
    dist_mult = 1.0 + 0.018 * np.maximum(0.0, curve[:, 0] - 12.0)
    hw = half_w * dist_mult
    left = curve.copy()
    left[:, 1] -= hw
    left[:, 2] += z_off
    right = curve.copy()
    right[:, 1] += hw
    right[:, 2] += z_off
    poly = self._project_strip(left, right)
    if poly.size:
      draw_polygon_solid(poly, color)

  def _road_boundary_curve(self, side: str, lanes: list, road_edges: list | None,
                           road_edge_stds: np.ndarray | None, path: np.ndarray,
                           max_dist: float, ground_z: float) -> np.ndarray:
    is_left = side == 'left'
    edge_idx = 0 if is_left else 1
    if road_edges and edge_idx < len(road_edges):
      std = float(road_edge_stds[edge_idx]) if road_edge_stds is not None and edge_idx < len(road_edge_stds) else 1.0
      raw = getattr(road_edges[edge_idx], 'raw_points', None)
      if raw is not None and raw.shape[0] >= 2 and std < 2.4:
        return self._extend_line_behind(raw, max_dist)

    sign = -1.0 if is_left else 1.0
    outer_idx = 0 if is_left else 3
    ego_idx = 1 if is_left else 2
    if outer_idx < len(lanes) and self._lane_gate.on[outer_idx] and lanes[outer_idx].raw_points.shape[0] >= 2:
      c = self._extend_line_behind(lanes[outer_idx].raw_points, max_dist)
      c[:, 1] += sign * VR_ROAD_FALLBACK_MARGIN_M
      return c
    if ego_idx < len(lanes) and self._lane_gate.on[ego_idx] and lanes[ego_idx].raw_points.shape[0] >= 2:
      c = self._extend_line_behind(lanes[ego_idx].raw_points, max_dist)
      c[:, 1] += sign * (3.4 + VR_ROAD_FALLBACK_MARGIN_M)
      return c
    c = self._extend_line_behind(path, max_dist)
    c[:, 1] += sign * 6.5
    c[:, 2] += ground_z
    return c

  def _stabilize_path(self, path: np.ndarray, path_max: float, v_ego: float, dt: float) -> np.ndarray:
    stationary = v_ego < 0.35
    if self._smooth_path_max is None:
      self._smooth_path_max = path_max
    else:
      tau = 0.85 if stationary else 0.22
      self._smooth_path_max += (path_max - self._smooth_path_max) * _ema_alpha(dt, tau)
    smax = max(2.5, float(self._smooth_path_max))
    xs = np.linspace(1.6, smax, 28, dtype=np.float64)
    ty = np.interp(xs, path[:, 0], path[:, 1])
    tz = np.interp(xs, path[:, 0], path[:, 2])
    if self._smooth_path_y is None or self._smooth_path_z is None:
      self._smooth_path_y = ty.copy()
      self._smooth_path_z = tz.copy()
    else:
      alpha = _ema_alpha(dt, 0.85 if stationary else 0.14)
      self._smooth_path_y += (ty - self._smooth_path_y) * alpha
      self._smooth_path_z += (tz - self._smooth_path_z) * alpha
    return np.stack((xs, self._smooth_path_y, self._smooth_path_z), axis=1)

  def _draw_blindspot_zones(self, car_state, ground_z: float, is_night: bool, dt: float, now_t: float) -> None:
    pal = VR_PALETTE[is_night]
    for side, bs_attr, bl_attr, y_in, y_out in (
      ('left', 'leftBlindspot', 'leftBlinker', -2.15, -4.65),
      ('right', 'rightBlindspot', 'rightBlinker', 2.15, 4.65),
    ):
      active = bool(getattr(car_state, bs_attr, False)) if car_state is not None else False
      blinker = bool(getattr(car_state, bl_attr, False)) if car_state is not None else False
      lvl = self._bs_level[side]
      lvl += ((1.0 if active else 0.0) - lvl) * _ema_alpha(dt, 0.16)
      self._bs_level[side] = lvl
      if lvl <= 0.02:
        continue
      danger = active and blinker
      pulse = (0.65 + 0.35 * abs(math.sin(now_t * 6.5))) if danger else 1.0
      base_rgb = pal['danger'] if danger else pal['warn']
      alpha = (0.38 if danger else 0.24) * lvl * pulse

      left_pts = np.array([[-4.2, y_in, ground_z], [2.2, y_in, ground_z]], dtype=np.float64)
      right_pts = np.array([[-4.2, y_out, ground_z], [2.2, y_out, ground_z]], dtype=np.float64)
      if y_in > y_out:
        left_pts, right_pts = right_pts, left_pts
      poly = self._project_strip(left_pts, right_pts)
      if poly.size:
        draw_polygon_solid(poly, _c(base_rgb, alpha))

  def _draw_lead_chevron(self, d_rel: float, y_dev: float, ground_z: float, color: rl.Color) -> None:
    """Blue forward-pointing road chevron right behind L1 lead vehicle."""
    x_tip = max(0.5, d_rel - 1.55)
    x_base = max(0.2, x_tip - 1.35)
    pts_3d = np.array([
      [x_tip, y_dev, ground_z],
      [x_base, y_dev - 0.85, ground_z],
      [x_base, y_dev + 0.85, ground_z],
    ], dtype=np.float64)
    xy, depth = self.camera.project(pts_3d)
    if np.all(depth > 0.4) and np.isfinite(xy).all():
      rl.draw_triangle(
        rl.Vector2(float(xy[0, 0]), float(xy[0, 1])),
        rl.Vector2(float(xy[1, 0]), float(xy[1, 1])),
        rl.Vector2(float(xy[2, 0]), float(xy[2, 1])),
        color,
      )
      rl.draw_triangle(
        rl.Vector2(float(xy[0, 0]), float(xy[0, 1])),
        rl.Vector2(float(xy[2, 0]), float(xy[2, 1])),
        rl.Vector2(float(xy[1, 0]), float(xy[1, 1])),
        color,
      )

  def _draw_distance_pill(self, cx: float, top_y: float, text: str, is_night: bool,
                          danger: bool, warning: bool, is_primary: bool) -> None:
    pal = VR_PALETTE[is_night]
    font = gui_app.font(FontWeight.BOLD)
    font_size = (20 if is_primary else 15) if self._is_compact else (30 if is_primary else 22)
    sz = measure_text_cached(font, text, font_size)
    pad_x = 18.0 if not self._is_compact else 11.0
    pad_y = 8.0 if not self._is_compact else 5.0
    w = sz.x + 2.0 * pad_x
    h = sz.y + 2.0 * pad_y
    rx = cx - w * 0.5
    ry = top_y - h - (10.0 if not self._is_compact else 6.0)

    if danger:
      bg = _c(pal['danger'], 0.94)
    elif warning:
      bg = rl.Color(230, 125, 10, 240)
    elif is_primary:
      bg = _c(pal['accent'], 0.94)
    else:
      bg = rl.Color(42, 68, 105, 200) if is_night else rl.Color(95, 118, 148, 200)

    rl.draw_rectangle_rounded(rl.Rectangle(rx, ry, w, h), 0.95, 10, bg)
    rl.draw_text_ex(font, text, rl.Vector2(rx + pad_x, ry + pad_y), font_size, 0, rl.Color(255, 255, 255, 255))

  def _draw_ego_vehicle(self, car_state, ground_z: float, is_night: bool, dt: float, now_t: float) -> None:
    """Draw 3rd-person Genesis-style ego sedan at bottom-center with brake glow & blinkers."""
    pal = VR_PALETTE[is_night]
    box = self.camera.car_box(-2.4, 0.0, ground_z, max_width_frac=0.34, scale_mult=1.18)
    if box is None:
      return
    cx, cy, vw, vh = box

    tex = None
    try:
      tex = gui_app.texture('images/ego_car.png')
    except Exception:
      tex = None
    tex_w = getattr(tex, 'width', 0) if tex is not None else 0
    tex_h = getattr(tex, 'height', 0) if tex is not None else 0
    has_tex = isinstance(tex_w, (int, float)) and tex_w > 0 and isinstance(tex_h, (int, float)) and tex_h > 0

    draw_w = float(vw)
    draw_h = float(vw * (float(tex_h) / float(tex_w))) if has_tex else float(vh)
    ground_y = min(self.camera._y0 + self.camera._h * 0.87, cy + vh * 0.50)
    car_cy = ground_y - draw_h * 0.48

    # Soft dark contact shadow under ego tires/body
    sh_a = pal['shadow_alpha']
    for i in range(6):
      t = i / 5.0
      rx = draw_w * (0.58 - 0.16 * t)
      ry = max(draw_h * (0.11 - 0.05 * t), 2.0)
      rl.draw_ellipse(int(cx), int(ground_y - draw_h * 0.015), rx, ry,
                      rl.Color(0, 0, 0, int((8 + 18 * t) * sh_a)))

    # Brake state smoothing (brakeLights or active deceleration)
    brake_on = False
    if car_state is not None:
      brake_on = bool(getattr(car_state, 'brakeLights', False)) or bool(getattr(car_state, 'brakePressed', False))
      if not brake_on and float(getattr(car_state, 'aEgo', 0.0)) < -0.55:
        brake_on = True
    self._brake_level += ((1.0 if brake_on else 0.0) - self._brake_level) * _ema_alpha(dt, 0.08)

    # Red ground reflection behind ego bumper when braking (matching carrot-vision3d)
    if self._brake_level > 0.02:
      bl = self._brake_level
      glow_y = ground_y + draw_h * 0.10
      for i in range(5):
        t = i / 4.0
        rx = draw_w * (0.56 - 0.24 * t)
        ry = max(draw_h * (0.14 - 0.07 * t), 2.0)
        rl.draw_ellipse(int(cx), int(glow_y), rx, ry,
                        rl.Color(255, 65, 55, int((6 + 14 * t) * bl)))

    if has_tex:
      tint = rl.Color(236, 240, 248, 255) if is_night else rl.Color(255, 255, 255, 255)
      rl.draw_texture_pro(
        tex,
        rl.Rectangle(0.0, 0.0, float(tex_w), float(tex_h)),
        rl.Rectangle(cx - draw_w * 0.5, car_cy - draw_h * 0.5, draw_w, draw_h),
        rl.Vector2(0.0, 0.0), 0.0, tint,
      )
    else:
      view_3d_draw.draw_vehicle_rear(cx, car_cy, draw_w, draw_h, is_night, style='ego', braking=brake_on, glow=False)

    # Taillight + CHMSL bloom aligned to Genesis G80 twin-line rear lamps & top rear glass
    if self._brake_level > 0.05:
      bl = self._brake_level
      chmsl_y = car_cy - draw_h * 0.305
      tail_y = car_cy + draw_h * 0.115
      rl.draw_ellipse(int(cx), int(chmsl_y), draw_w * 0.11, max(draw_h * 0.014, 1.5),
                      rl.Color(255, 50, 40, int(195 * bl)))
      for side_x in (-0.33, 0.33):
        rl.draw_ellipse(int(cx + draw_w * side_x), int(tail_y), draw_w * 0.135, max(draw_h * 0.022, 2.0),
                        rl.Color(255, 40, 35, int(185 * bl)))

    # Amber turn-signal pulse
    left_bl = bool(getattr(car_state, 'leftBlinker', False)) if car_state is not None else False
    right_bl = bool(getattr(car_state, 'rightBlinker', False)) if car_state is not None else False
    if left_bl or right_bl:
      if self._blink_since <= 0.0:
        self._blink_since = now_t
      blink_on = ((now_t - self._blink_since) % 0.72) < 0.38
      if blink_on:
        bl_y = car_cy + draw_h * 0.125
        if left_bl:
          rl.draw_ellipse(int(cx - draw_w * 0.37), int(bl_y), draw_w * 0.09, max(draw_h * 0.025, 2.0),
                          rl.Color(255, 168, 20, 220))
        if right_bl:
          rl.draw_ellipse(int(cx + draw_w * 0.37), int(bl_y), draw_w * 0.09, max(draw_h * 0.025, 2.0),
                          rl.Color(255, 168, 20, 220))
    else:
      self._blink_since = 0.0

  def draw(self, renderer, sm, rect: rl.Rectangle, is_night: bool, now: float | None = None) -> None:
    path = renderer._path.raw_points
    lanes = renderer._lane_lines
    if path.shape[0] < 2 or len(lanes) == 0 or lanes[0].raw_points.shape[0] == 0:
      return

    dt, now_t = self._dt(now)
    pal = VR_PALETTE[is_night]

    raw_ground_z = float(renderer._path_offset_z)
    if self._smooth_ground_z is None:
      self._smooth_ground_z = raw_ground_z
    else:
      self._smooth_ground_z += (raw_ground_z - self._smooth_ground_z) * _ema_alpha(dt, 0.50)
    ground_z = self._smooth_ground_z

    car_state = sm['carState'] if getattr(sm, 'valid', {}).get('carState', False) else None
    v_ego = float(getattr(car_state, 'vEgo', 0.0)) if car_state is not None else 0.0
    if v_ego > 0.2 and dt > 0.0:
      self._travel_m = (self._travel_m + v_ego * dt) % (VR_SEAM_SPACING_M * 100.0)

    # Gentle lateral chase-camera follow on curves
    if v_ego < 0.4 or float(path[-1, 0]) < 12.0:
      yaw = 0.0
    else:
      lat_25 = float(np.interp(25.0, path[:, 0], path[:, 1]))
      yaw = 0.35 * math.atan2(lat_25, 28.0)

    self.horizon_frac = VR_CHASE_POSE.horizon_frac
    self.horizon_y = float(rect.y + self.horizon_frac * rect.height)
    self.camera.update(rect, ground_z, 0.0, yaw, dt, horizon_frac=self.horizon_frac, pose=VR_CHASE_POSE)

    probs = renderer._lane_line_probs
    road_edges = getattr(renderer, '_road_edges', None)
    road_edge_stds = getattr(renderer, '_road_edge_stds', None)
    max_dist = float(np.clip(lanes[0].raw_points[-1, 0], VR_MIN_DRAW_M, VR_MAX_DRAW_M))

    lane_levels = self._update_lane_gate(lanes, probs, road_edges, road_edge_stds, dt)

    # 1. Asphalt Road Surface Ribbon (distinct from outer ground plane)
    left_bound = self._road_boundary_curve('left', lanes, road_edges, road_edge_stds, path, max_dist, ground_z)
    right_bound = self._road_boundary_curve('right', lanes, road_edges, road_edge_stds, path, max_dist, ground_z)
    if left_bound.shape[0] >= 2 and right_bound.shape[0] >= 2 and left_bound.shape[0] == right_bound.shape[0]:
      road_poly = self._project_strip(left_bound, right_bound)
      if road_poly.size:
        draw_polygon_solid(road_poly, _c(pal['road']))

      # Subtle moving asphalt seams (12m spacing) for forward speed sensation
      seam_phase = self._travel_m % VR_SEAM_SPACING_M
      seam_x = VR_SEAM_SPACING_M - seam_phase
      while seam_x < min(max_dist, 95.0):
        fade = 1.0 - float(np.clip((seam_x - 45.0) / 50.0, 0.0, 1.0))
        if fade > 0.05:
          ly = float(np.interp(seam_x, left_bound[:, 0], left_bound[:, 1]))
          ry = float(np.interp(seam_x, right_bound[:, 0], right_bound[:, 1]))
          lz = float(np.interp(seam_x, left_bound[:, 0], left_bound[:, 2]))
          rz = float(np.interp(seam_x, right_bound[:, 0], right_bound[:, 2]))
          p1 = self.camera.project_point(seam_x, ly, lz)
          p2 = self.camera.project_point(seam_x, ry, rz)
          if p1 is not None and p2 is not None:
            rl.draw_line_ex(rl.Vector2(p1[0], p1[1]), rl.Vector2(p2[0], p2[1]), 1.4, _c(pal['road_seam'], fade))
        seam_x += VR_SEAM_SPACING_M

    # 2. Road Edges (outer reddish-brown boundaries)
    if road_edges:
      for i, edge in enumerate(road_edges[:2]):
        std = float(road_edge_stds[i]) if road_edge_stds is not None and i < len(road_edge_stds) else 1.0
        raw = getattr(edge, 'raw_points', None)
        if raw is None or raw.shape[0] < 2 or std > 2.5:
          continue
        conf = float(np.clip((2.5 - std) / 1.8, 0.3, 1.0))
        curve = self._extend_line_behind(raw, max_dist)
        self._draw_ribbon_3d(curve, 0.14, 0.0, _c(pal['edge'], conf))

    # 3. 4 Lane Lines with smooth hysteresis fade
    for i in range(min(4, len(lanes))):
      lvl = lane_levels[i]
      if lvl <= 0.02 or lanes[i].raw_points.shape[0] < 2:
        continue
      is_ego = i in (1, 2)
      col = pal['lane_ego'] if is_ego else pal['lane_adj']
      hw = 0.095 if is_ego else 0.075
      curve = self._extend_line_behind(lanes[i].raw_points, max_dist)
      self._draw_ribbon_3d(curve, hw, 0.0, _c(col, lvl))

    # 4. Radar + Vision Vehicle Fusion (compute before path so path stops cleanly behind L1)
    started_frame = getattr(ui_state, 'started_frame', 0)
    vehicles = self.tracks.update(sm, v_ego, path, road_edges, road_edge_stds, dt, now_t, started_frame=started_frame)
    lead_one: VrVehicle | None = next((v for v in vehicles if v.tag == 'L1'), None)

    # 5. Driving Path Ribbon (ends 3m before L1 lead car)
    raw_path_max = float(np.clip(path[-1, 0], VR_MIN_DRAW_M, VR_MAX_DRAW_M))
    if lead_one is not None:
      raw_path_max = min(raw_path_max, max(lead_one.d_rel - 3.0, 2.5))
    stab_path = self._stabilize_path(path, raw_path_max, v_ego, dt)
    sd_state = sm['selfdriveState'] if getattr(sm, 'valid', {}).get('selfdriveState', False) else None
    engaged = bool(getattr(sd_state, 'enabled', False)) if sd_state is not None else False
    path_col = _c(pal['path_engaged']) if engaged else _c(pal['path_idle'])
    self._draw_ribbon_3d(stab_path, 0.92, ground_z, path_col)

    # 6. Blindspot Warning Floor Zones beside Ego
    self._draw_blindspot_zones(car_state, ground_z, is_night, dt, now_t)

    # 7. Vehicles (already sorted far -> near)
    for veh in vehicles:
      z_rel = float(np.interp(veh.d_rel, path[:, 0], path[:, 2]))
      if veh.tag == 'L1':
        self._draw_lead_chevron(veh.d_rel, veh.y_dev, z_rel + ground_z, _c(pal['accent'], veh.opacity * 0.92))

      dist_boost = 1.0 if veh.d_rel <= 22.0 else (1.0 + min(0.95, (veh.d_rel - 22.0) / 38.0 * 0.85))
      scale_mult = (1.35 if veh.tag == 'ADJ' else 1.48) * dist_boost
      box = self.camera.car_box(veh.d_rel, veh.y_dev, z_rel + ground_z, max_width_frac=0.38, scale_mult=scale_mult)
      if box is None:
        continue
      sx, sy, vw, vh = box
      style = 'adj' if veh.tag == 'ADJ' else ('lead_two' if veh.tag == 'L2' else 'lead')
      view_3d_draw.draw_vehicle_rear(sx, sy, vw, vh, is_night, style=style, braking=veh.braking, alpha=veh.opacity, glow=False)

      if veh.tag == 'L1' and veh.opacity > 0.35:
        ttc = -veh.d_rel / veh.v_rel if veh.v_rel < -0.5 else 999.0
        pill_text = f"{veh.disp_d:.1f} m"
        self._draw_distance_pill(sx, sy - vh * 0.50, pill_text, is_night, ttc < 2.5, ttc < 4.5, True)

    # 8. Ego Vehicle (bottom-center foreground)
    self._draw_ego_vehicle(car_state, ground_z, is_night, dt, now_t)

    # 9. Traffic Light Badge (reuses existing View3dScene indicator even if 3D View wasn't visited first)
    tl_scene = getattr(renderer, '_view_3d_scene', None)
    if tl_scene is None:
      if self._tl_scene is None:
        self._tl_scene = View3dScene(C4_SIZES if self._is_compact else C3_SIZES)
      tl_scene = self._tl_scene
    tl_scene.draw_traffic_light_badge(rect, sm, dt)

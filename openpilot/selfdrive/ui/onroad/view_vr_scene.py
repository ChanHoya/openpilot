"""Third-person VR Mode driving scene inspired by carrot-vision3d v2.3.

Renders an elevated chase-camera view with:
- 3-tier sky / ground / asphalt road surface (day & night palettes) + moving road seams
- 4-lane hysteresis-gated lane lines + road edges + driving path ribbon
- Ego Genesis-style sedan sprite with soft shadow, brake glow, blinkers, and blindspot floor zones
- Stabilized L1/L2 lead vehicles with distance pills & road chevron + radar-fused adjacent cars
"""
from dataclasses import dataclass
import colorsys
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
from openpilot.system.ui.lib.shader_polygon import draw_polygon, draw_polygon_solid, Gradient
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
    'road_warn': (112, 34, 38, 255),
    'road_seam': (40, 46, 56, 120),
    'road_seam_warn': (155, 62, 66, 130),
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
    'road_warn': (244, 176, 176, 255),
    'road_seam': (176, 184, 194, 110),
    'road_seam_warn': (218, 136, 136, 120),
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
VR_PATH_RAINBOW_WAVE_M = 20.0
VR_PATH_RAINBOW_SAT = 0.90
VR_PATH_RAINBOW_LIGHT_DAY = 0.62
VR_PATH_RAINBOW_LIGHT_NIGHT = 0.56
VR_PATH_IDLE_ALPHA = 0.60


def _ema_alpha(dt: float, tau: float) -> float:
  if tau <= 0.0:
    return 1.0
  dt_eff = dt if dt > 0.0 else 0.05
  return float(1.0 - math.exp(-dt_eff / tau))


def _c(rgba, alpha: float = 1.0) -> rl.Color:
  return rl.Color(int(rgba[0]), int(rgba[1]), int(rgba[2]), int(np.clip(rgba[3] * alpha, 0, 255)))


def _mix_rgba(c0, c1, t: float) -> tuple[int, int, int, int]:
  t = float(np.clip(t, 0.0, 1.0))
  return (
    int(round(c0[0] + (c1[0] - c0[0]) * t)),
    int(round(c0[1] + (c1[1] - c0[1]) * t)),
    int(round(c0[2] + (c1[2] - c0[2]) * t)),
    int(round(c0[3] + (c1[3] - c0[3]) * t)),
  )


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
    self._engaged_level: float = 0.0
    self._lc_level: float = 0.0
    self._bs_level: dict[str, float] = {'left': 0.0, 'right': 0.0}
    self._blink_since: float = 0.0
    self._tl_scene: View3dScene | None = None
    self._cam_state: dict = {}
    self._bump_state: dict = {}
    self._ego_screen_box: tuple[float, float, float, float, float] | None = None
    self._last_speed_cam_sign: tuple[float, float, float, int, float, bool] | None = None
    self._last_speed_bump_dist: float = 0.0
    self._last_tl_overhead: tuple[str, float, int] | None = None
    self._last_road_color: tuple[int, int, int, int] | None = None
    self._last_vehicle_textures: list[tuple[str, str]] = []
    self._last_bsd_boxes: list[tuple[str, float, float, float, float]] = []
    self._last_vehicle_boxes: list[tuple[str, float, float, float, float, float]] = []

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

  def _draw_vr_path_ribbon(self, rect: rl.Rectangle, curve: np.ndarray, half_w: float, z_off: float,
                           is_night: bool, engaged: bool, lane_changing: bool,
                           dt: float, now_t: float) -> None:
    target_eng = 1.0 if engaged else 0.0
    if dt <= 0.0:
      self._engaged_level = target_eng
    else:
      self._engaged_level += (target_eng - self._engaged_level) * _ema_alpha(dt, 0.35)

    target_lc = 1.0 if (engaged and lane_changing) else 0.0
    if dt <= 0.0:
      self._lc_level = target_lc
    else:
      self._lc_level += (target_lc - self._lc_level) * _ema_alpha(dt, 0.25)

    if curve.shape[0] < 2:
      return

    pal = VR_PALETTE[is_night]
    dist_mult = 1.0 + 0.018 * np.maximum(0.0, curve[:, 0] - 12.0)
    hw = half_w * dist_mult
    left = curve.copy()
    left[:, 1] -= hw
    left[:, 2] += z_off
    right = curve.copy()
    right[:, 1] += hw
    right[:, 2] += z_off
    l_xy, l_d = self.camera.project(left)
    r_xy, r_d = self.camera.project(right)
    margin = 0.7 * self.camera._w
    ok = (
      (l_d > 0.4) & (r_d > 0.4)
      & np.isfinite(l_xy).all(axis=1) & np.isfinite(r_xy).all(axis=1)
      & (l_xy[:, 1] >= self.camera._y0 - margin) & (l_xy[:, 1] <= self.camera._y0 + self.camera._h + margin)
    )
    xs = curve[:, 0][ok]
    l_xy, r_xy = l_xy[ok], r_xy[ok]
    if len(l_xy) < 2:
      return

    poly = np.concatenate((l_xy, r_xy[::-1])).astype(np.float32)
    x_start = float(xs[0])
    x_end = float(xs[-1])
    y_bot_rl = float(max(l_xy[0, 1], r_xy[0, 1]))
    y_top_rl = float(min(l_xy[-1, 1], r_xy[-1, 1]))
    rect_h = max(1.0, float(rect.height))
    gl_y_bot = float(np.clip(1.0 - (y_bot_rl - float(rect.y)) / rect_h, 0.0, 1.0))
    gl_y_top = float(np.clip(1.0 - (y_top_rl - float(rect.y)) / rect_h, 0.0, 1.0))

    eng = self._engaged_level
    lc = self._lc_level
    idle_rgb = pal['path_idle'][:3]
    flow_offset = self._travel_m + (now_t * 14.0 * lc if lc > 0.01 else 0.0)
    stop_count = 12
    stop_colors: list[rl.Color] = []
    stop_positions: list[float] = []

    for k in range(stop_count):
      s = k / float(stop_count - 1)
      x_dist = x_start + s * (x_end - x_start)
      dist_fade = 0.78 * ((1.0 - s) ** 1.15) + 0.14
      tip_fade = min(1.0, max(0.25, (1.0 - s) / 0.14))
      alpha_scale = (VR_PATH_IDLE_ALPHA + (1.0 - VR_PATH_IDLE_ALPHA) * eng) * dist_fade * tip_fade

      if eng > 0.01:
        hue = (0.86 + (x_dist + flow_offset) / VR_PATH_RAINBOW_WAVE_M) % 1.0
        sat = min(0.98, VR_PATH_RAINBOW_SAT + 0.08 * lc)
        light = (VR_PATH_RAINBOW_LIGHT_NIGHT if is_night else VR_PATH_RAINBOW_LIGHT_DAY) + 0.04 * lc
        rr, gg, bb = colorsys.hls_to_rgb(hue, light, sat)
        r = int(round(idle_rgb[0] + (rr * 255.0 - idle_rgb[0]) * eng))
        g = int(round(idle_rgb[1] + (gg * 255.0 - idle_rgb[1]) * eng))
        b = int(round(idle_rgb[2] + (bb * 255.0 - idle_rgb[2]) * eng))
        base_a = (210.0 if is_night else 185.0) + 35.0 * lc
      else:
        r, g, b = idle_rgb
        base_a = float(pal['path_idle'][3])

      alpha = int(np.clip(base_a * alpha_scale, 0.0, 255.0))
      stop_colors.append(rl.Color(r, g, b, alpha))
      stop_positions.append(s)

    if gl_y_top > gl_y_bot + 1e-4:
      grad = Gradient(
        start=(0.0, gl_y_top),
        end=(0.0, gl_y_bot),
        colors=stop_colors,
        stops=stop_positions,
      )
      draw_polygon(rect, poly, gradient=grad)
    else:
      draw_polygon_solid(poly, stop_colors[0])

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
    xs = np.linspace(-1.2, smax, 30, dtype=np.float64)
    px = path[:, 0].astype(np.float64)
    py = path[:, 1].astype(np.float64)
    pz = path[:, 2].astype(np.float64)
    ty = np.interp(xs, px, py)
    tz = np.interp(xs, px, pz)
    behind = xs < px[0]
    if np.any(behind):
      k_slope = min(3, px.size - 1)
      slope_y = (py[k_slope] - py[0]) / max(0.01, px[k_slope] - px[0])
      ty[behind] = py[0] + slope_y * (xs[behind] - px[0])
      tz[behind] = pz[0]
    if self._smooth_path_y is None or self._smooth_path_z is None or self._smooth_path_y.shape != ty.shape:
      self._smooth_path_y = ty.copy()
      self._smooth_path_z = tz.copy()
    else:
      alpha = _ema_alpha(dt, 0.85 if stationary else 0.14)
      self._smooth_path_y += (ty - self._smooth_path_y) * alpha
      self._smooth_path_z += (tz - self._smooth_path_z) * alpha
    return np.stack((xs, self._smooth_path_y, self._smooth_path_z), axis=1)

  def _draw_blindspot_zones(self, car_state, ground_z: float, is_night: bool, dt: float, now_t: float) -> None:
    pal = VR_PALETTE[is_night]
    for side, bs_attr, bl_attr, y_in, y_out, y_car in (
      ('left', 'leftBlindspot', 'leftBlinker', -2.15, -4.65, -3.35),
      ('right', 'rightBlindspot', 'rightBlinker', 2.15, 4.65, 3.35),
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

      left_pts = np.array([[-7.5, y_in, ground_z], [-1.2, y_in, ground_z]], dtype=np.float64)
      right_pts = np.array([[-7.5, y_out, ground_z], [-1.2, y_out, ground_z]], dtype=np.float64)
      if y_in > y_out:
        left_pts, right_pts = right_pts, left_pts
      poly = self._project_strip(left_pts, right_pts)
      if poly.size:
        draw_polygon_solid(poly, _c(base_rgb, alpha))

      if lvl > 0.05:
        bsd_box = self.camera.car_box(-2.4, y_car, ground_z, max_width_frac=0.56, scale_mult=1.88)
        if bsd_box is not None:
          bsx, _, bvw, bvh = bsd_box
          side_sign = 1.0 if side == 'right' else -1.0
          bsx += side_sign * bvw * 0.11
          bsx = float(np.clip(bsx, self.camera._x0 + bvw * 0.48, self.camera._x0 + self.camera._w - bvw * 0.48))
          # Place the BSD car behind ego (lower on screen) in the rear-quarter blindspot zone.
          bsd_ground_y = self.camera._y0 + self.camera._h * 1.095
          bsy = bsd_ground_y - bvh * 0.50
          self._last_bsd_boxes.append((side, float(bsx), float(bsy), float(bvw), float(bvh)))
          if danger:
            blink = 0.28 + 0.72 * (0.5 + 0.5 * math.sin(now_t * 14.0))
          else:
            blink = 0.40 + 0.56 * (0.5 + 0.5 * math.sin(now_t * 7.5))
          tex_name = 'vr_bsd_car_right.png' if side == 'right' else 'vr_bsd_car_left.png'
          self._draw_vr_vehicle_sprite(bsx, bsy, bvw, bvh, tex_name, is_night, 'adj', danger, lvl * blink)

  def _draw_vr_vehicle_sprite(self, cx: float, cy: float, vw: float, vh: float, tex_name: str,
                              is_night: bool, style: str, braking: bool, alpha: float) -> float:
    """Draw carrot-vision3d perspective-matched car sprite (fallback to procedural draw_vehicle_rear)."""
    self._last_vehicle_textures.append((style, tex_name))
    tex = None
    try:
      tex = gui_app.texture(f'images/{tex_name}')
    except Exception:
      tex = None
    tex_w = getattr(tex, 'width', 0) if tex is not None else 0
    tex_h = getattr(tex, 'height', 0) if tex is not None else 0
    has_tex = isinstance(tex_w, (int, float)) and tex_w > 0 and isinstance(tex_h, (int, float)) and tex_h > 0

    if not has_tex:
      view_3d_draw.draw_vehicle_rear(cx, cy, vw, vh, is_night, style=style, braking=braking, alpha=alpha, glow=False)
      return float(cy - vh * 0.50)

    pal = VR_PALETTE[is_night]
    draw_w = float(vw)
    draw_h = float(vw * (float(tex_h) / float(tex_w)))
    ground_y = float(cy + vh * 0.50)
    draw_cy = ground_y - draw_h * 0.50

    sh_a = pal['shadow_alpha'] * alpha
    for i in range(4):
      t = i / 3.0
      rx = draw_w * (0.52 - 0.14 * t)
      ry = max(draw_h * (0.10 - 0.04 * t), 1.5)
      rl.draw_ellipse(int(round(cx)), int(round(ground_y - draw_h * 0.02)), rx, ry,
                      rl.Color(0, 0, 0, int(np.clip((10 + 18 * t) * sh_a, 0, 255))))

    base_rgb = (220, 228, 240) if is_night else (255, 255, 255)
    tint_a = int(np.clip(round(255.0 * alpha), 0, 255))
    rl.draw_texture_pro(
      tex,
      rl.Rectangle(0.0, 0.0, float(tex_w), float(tex_h)),
      rl.Rectangle(cx - draw_w * 0.50, draw_cy - draw_h * 0.50, draw_w, draw_h),
      rl.Vector2(0.0, 0.0), 0.0, rl.Color(base_rgb[0], base_rgb[1], base_rgb[2], tint_a),
    )

    if braking:
      tail_y = draw_cy + draw_h * 0.14
      glow_a = int(np.clip(round(170.0 * alpha), 0, 255))
      for side_x in (-0.30, 0.30):
        rl.draw_ellipse(int(round(cx + draw_w * side_x)), int(round(tail_y)),
                        max(draw_w * 0.12, 2.0), max(draw_h * 0.032, 1.5),
                        rl.Color(255, 45, 38, glow_a))

    return float(draw_cy - draw_h * 0.50)

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
    box = self.camera.car_box(-2.4, 0.0, ground_z, max_width_frac=0.40, scale_mult=1.34)
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
    ground_y = self.camera._y0 + self.camera._h * 0.965
    car_cy = ground_y - draw_h * 0.48
    self._ego_screen_box = (float(cx), float(car_cy), draw_w, draw_h, float(ground_y))

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
      glow_y = min(self.camera._y0 + self.camera._h - 6.0, ground_y + draw_h * 0.04)
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

  def _resolve_speed_camera(self, sm, v_ego: float, now_t: float) -> tuple[int, float, bool]:
    return view_3d_draw.resolve_speed_camera_state(sm, v_ego, now_t, self._cam_state)

  def _resolve_speed_bump(self, sm, v_ego: float, now_t: float) -> float:
    valid = getattr(sm, 'valid', {})
    cs = sm['carState'] if valid.get('carState', False) else None
    cm = sm['carrotMan'] if valid.get('carrotMan', False) else None
    bump_offset = 10.0
    raw_bump = None
    if cs is not None:
      d_cs = float(getattr(cs, 'speedBumpDistance', 0.0) or 0.0)
      if bump_offset + 0.5 < d_cs <= 130.0:
        raw_bump = d_cs - bump_offset
    if raw_bump is None and cm is not None and int(float(getattr(cm, 'xSpdType', 0) or 0)) == 22:
      d_cm = float(getattr(cm, 'xSpdDist', 0.0) or 0.0)
      if bump_offset + 0.5 < d_cm <= 130.0:
        raw_bump = d_cm - bump_offset

    if raw_bump is None:
      self._bump_state.clear()
      return 0.0

    prev_raw = float(self._bump_state.get('raw_dist', 0.0) or 0.0)
    prev_t = float(self._bump_state.get('last_t', 0.0) or 0.0)
    if abs(raw_bump - prev_raw) > 0.05 or prev_t <= 0.0:
      self._bump_state['raw_dist'] = raw_bump
      self._bump_state['est_dist'] = raw_bump
      self._bump_state['last_t'] = now_t
    else:
      dt_b = float(np.clip(now_t - prev_t, 0.0, 0.25))
      self._bump_state['est_dist'] = max(0.5, float(self._bump_state.get('est_dist', raw_bump)) - max(0.0, v_ego) * dt_b)
      self._bump_state['last_t'] = now_t
    return float(self._bump_state['est_dist'])

  def _draw_speed_camera_sign(self, rect: rl.Rectangle, path: np.ndarray, ground_z: float,
                              limit_kph: int, cam_dist: float, overspeed: bool, is_night: bool,
                              min_cx: float | None = None) -> None:
    if limit_kph <= 0 or not (0.5 <= cam_dist <= 500.0):
      self._last_speed_cam_sign = None
      return

    d_proj = float(min(cam_dist, min(120.0, float(path[-1, 0]))))
    sy = float(np.interp(d_proj, path[:, 0], path[:, 1])) + 2.6
    sz = float(np.interp(d_proj, path[:, 0], path[:, 2])) + ground_z - 7.4
    p_over = self.camera.project_point(d_proj, sy, sz)

    if self._is_compact:
      sign_r = float(np.clip(520.0 / max(8.5, cam_dist), 14.0, 56.0))
    else:
      sign_r = float(np.clip(820.0 / max(8.5, cam_dist), 22.0, 92.0))

    if p_over is not None:
      cx = float(p_over[0])
      cy = float(p_over[1])
    else:
      cx = float(rect.x + rect.width * 0.56)
      cy = float(rect.y + rect.height * 0.22)
    left_lim = max(rect.x + rect.width * 0.18, min_cx) if min_cx is not None else (rect.x + rect.width * 0.18)
    cx = float(np.clip(cx, left_lim, rect.x + rect.width * 0.86))
    cy_max = max(rect.y + sign_r + 12.0, rect.y + rect.height * 0.27)
    cy = float(np.clip(cy, rect.y + sign_r + 10.0, cy_max))

    self._last_speed_cam_sign = (cx, cy, sign_r, int(limit_kph), float(cam_dist), bool(overspeed))
    view_3d_draw.draw_speed_camera_sign(cx, cy, sign_r, limit_kph, cam_dist, overspeed, is_night, is_compact=self._is_compact)

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

    cam_limit, cam_dist, cam_active = self._resolve_speed_camera(sm, v_ego, now_t)
    overspeed = cam_active and cam_limit > 0 and (v_ego * 3.6) > cam_limit * 1.10
    if overspeed:
      warn_pulse = 0.25 + 0.75 * (0.5 + 0.5 * math.sin(now_t * 6.5))
      road_rgba = _mix_rgba(pal['road'], pal['road_warn'], warn_pulse)
      seam_rgba = _mix_rgba(pal['road_seam'], pal['road_seam_warn'], warn_pulse)
    else:
      road_rgba = pal['road']
      seam_rgba = pal['road_seam']
    self._last_road_color = road_rgba

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
        draw_polygon_solid(road_poly, _c(road_rgba))

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
            rl.draw_line_ex(rl.Vector2(p1[0], p1[1]), rl.Vector2(p2[0], p2[1]), 1.4, _c(seam_rgba, fade))
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
    m_v2 = sm['modelV2'] if getattr(sm, 'valid', {}).get('modelV2', False) else None
    meta = getattr(m_v2, 'meta', None) if m_v2 is not None else None
    lc_raw = str(getattr(meta, 'laneChangeState', 'off')).lower() if meta is not None else 'off'
    lane_changing = lc_raw in ('prelanechange', 'lanechangestarting', 'lanechangefinishing', '1', '2', '3')
    self._draw_vr_path_ribbon(rect, stab_path, 0.92, ground_z, is_night, engaged, lane_changing, dt, now_t)

    # 5.5 Approaching Speed Bump on VR road surface
    bump_dist = self._resolve_speed_bump(sm, v_ego, now_t)
    self._last_speed_bump_dist = 0.0
    if 0.5 <= bump_dist <= 120.0 and float(path[-1, 0]) >= 2.0:
      d_b = min(bump_dist, float(path[-1, 0]))
      by = float(np.interp(d_b, path[:, 0], path[:, 1]))
      bz = float(np.interp(d_b, path[:, 0], path[:, 2])) + ground_z
      pl = self.camera.project_point(d_b, by - 1.65, bz)
      pr = self.camera.project_point(d_b, by + 1.65, bz)
      if pl is not None and pr is not None:
        bcx = 0.5 * (float(pl[0]) + float(pr[0]))
        bcy = 0.5 * (float(pl[1]) + float(pr[1]))
        bw = float(np.clip(abs(float(pr[0]) - float(pl[0])), 44.0, rect.width * 0.90))
        if rect.y < bcy <= rect.y + rect.height + 30.0:
          view_3d_draw.draw_road_speed_bump_texture(bcx, bcy, bw, bump_dist, is_night)
          self._last_speed_bump_dist = float(bump_dist)

    # 5.6 Red/Yellow Traffic Stop Line on VR road surface (overhead lamp drawn at step 9)
    resolve_tl = getattr(renderer, '_resolve_traffic_light', None)
    tl_lamp, tl_dist, tl_remain = resolve_tl(sm, path, v_ego, now_t) if callable(resolve_tl) else ('', 0.0, 0)
    self._last_tl_overhead = None
    tl_draw_params: tuple[float, float, float] | None = None
    if tl_lamp and 0.5 <= tl_dist <= 130.0 and float(path[-1, 0]) >= 2.0:
      d_tl = min(tl_dist, float(path[-1, 0]))
      sy_tl = float(np.interp(d_tl, path[:, 0], path[:, 1]))
      sz_tl = float(np.interp(d_tl, path[:, 0], path[:, 2])) + ground_z
      p_rl = self.camera.project_point(d_tl, sy_tl - 1.65, sz_tl)
      p_rr = self.camera.project_point(d_tl, sy_tl + 1.65, sz_tl)
      if tl_lamp in ('red', 'yellow') and p_rl is not None and p_rr is not None and 1.5 <= tl_dist <= 120.0:
        line_y = 0.5 * (float(p_rl[1]) + float(p_rr[1]))
        if rect.y < line_y <= rect.y + rect.height:
          rl.draw_line_ex(rl.Vector2(float(p_rl[0]), float(p_rl[1])),
                          rl.Vector2(float(p_rr[0]), float(p_rr[1])), 5.0, rl.Color(255, 65, 65, 220))
      p_over = self.camera.project_point(d_tl, sy_tl, sz_tl - 6.8)
      raw_tl_w = abs(float(p_rr[0]) - float(p_rl[0])) * 0.65 if (p_rl is not None and p_rr is not None) else (1650.0 / max(8.0, tl_dist))
      tl_w = float(np.clip(raw_tl_w, 75.0 if self._is_compact else 105.0, 165.0 if self._is_compact else 245.0))
      tl_cx = float(np.clip(p_over[0] if p_over is not None else rect.x + rect.width * 0.5, rect.x + rect.width * 0.22, rect.x + rect.width * 0.78))
      tl_cy = float(np.clip(p_over[1] if p_over is not None else rect.y + rect.height * 0.21, rect.y + rect.height * 0.14, rect.y + rect.height * 0.29))
      tl_draw_params = (tl_cx, tl_cy, tl_w)

    # 6. Blindspot Warning Floor Zones beside Ego
    self._last_vehicle_textures.clear()
    self._last_bsd_boxes.clear()
    self._last_vehicle_boxes.clear()
    self._draw_blindspot_zones(car_state, ground_z, is_night, dt, now_t)

    # 7. Vehicles (already sorted far -> near)
    for veh in vehicles:
      z_rel = float(np.interp(veh.d_rel, path[:, 0], path[:, 2]))
      if veh.tag == 'L1':
        self._draw_lead_chevron(veh.d_rel, veh.y_dev, z_rel + ground_z, _c(pal['accent'], veh.opacity * 0.92))

      near_scale = float(np.interp(veh.d_rel, [2.0, 14.0], [0.74, 1.0]))
      dist_boost = 1.0 if veh.d_rel <= 22.0 else (1.0 + min(0.95, (veh.d_rel - 22.0) / 38.0 * 0.85))
      if veh.tag == 'ADJ' and veh.d_rel < 18.0 and abs(veh.y_dev) > 2.2:
        tex_name = 'vr_bsd_car_right.png' if veh.y_dev > 0.0 else 'vr_bsd_car_left.png'
        scale_mult = 1.44 * near_scale * dist_boost
      elif veh.tag == 'L1':
        tex_name = 'vr_lead_car.png'
        scale_mult = 1.36 * near_scale * dist_boost
      else:
        tex_name = 'vr_adj_car.png'
        scale_mult = (1.28 if veh.tag == 'ADJ' else 1.36) * near_scale * dist_boost

      box = self.camera.car_box(veh.d_rel, veh.y_dev, z_rel + ground_z, max_width_frac=0.34, scale_mult=scale_mult)
      if box is None:
        continue
      sx, sy, vw, vh = box
      self._last_vehicle_boxes.append((veh.tag, float(veh.d_rel), float(sx), float(sy), float(vw), float(vh)))
      style = 'adj' if veh.tag == 'ADJ' else ('lead_two' if veh.tag == 'L2' else 'lead')
      top_y = self._draw_vr_vehicle_sprite(sx, sy, vw, vh, tex_name, is_night, style, veh.braking, veh.opacity)

      if veh.tag == 'L1' and veh.opacity > 0.35:
        ttc = -veh.d_rel / veh.v_rel if veh.v_rel < -0.5 else 999.0
        pill_text = f"{veh.disp_d:.1f} m"
        self._draw_distance_pill(sx, top_y, pill_text, is_night, ttc < 2.5, ttc < 4.5, True)

    # 8. Ego Vehicle (bottom-center foreground)
    self._draw_ego_vehicle(car_state, ground_z, is_night, dt, now_t)

    # 9. Approaching Overhead 4-Lamp Traffic Light & Speed Camera Sign
    min_cam_cx = None
    if tl_draw_params is not None:
      tl_cx, tl_cy, tl_w = tl_draw_params
      view_3d_draw.draw_overhead_traffic_light_4lamp(tl_cx, tl_cy, tl_w, tl_lamp, tl_dist, tl_remain, is_night)
      self._last_tl_overhead = (tl_lamp, float(tl_dist), int(tl_remain))
      min_cam_cx = tl_cx + tl_w * 0.5 + 48.0
    self._draw_speed_camera_sign(rect, path, ground_z, cam_limit, cam_dist, overspeed, is_night, min_cx=min_cam_cx)

    # 10. Traffic Light Badge fallback only when overhead 4-lamp was not drawn
    if tl_draw_params is None:
      tl_scene = getattr(renderer, '_view_3d_scene', None)
      if tl_scene is None:
        if self._tl_scene is None:
          self._tl_scene = View3dScene(C4_SIZES if self._is_compact else C3_SIZES)
        tl_scene = self._tl_scene
      tl_scene.draw_traffic_light_badge(rect, sm, dt)

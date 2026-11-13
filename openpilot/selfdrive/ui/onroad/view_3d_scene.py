"""3D view scene composition shared by the C3 and C4 model renderers."""
from dataclasses import dataclass
import math
import time

import numpy as np

from openpilot.selfdrive.ui.onroad.view_3d_camera import ChaseCamera, zoom_target, EGO_REAR_X_M
from openpilot.selfdrive.ui.onroad import view_3d_draw
from openpilot.system.ui.lib.shader_polygon import draw_polygon_solid

MIN_DRAW_DISTANCE = 10.0
MAX_DRAW_DISTANCE = 100.0
MAX_LEAD_DISTANCE = 130.0
MAX_ADJACENT_DISTANCE = 90.0


@dataclass(frozen=True)
class SceneSizes:
  wheel_radius_frac: float   # outer wheel radius / view height
  wheel_drop_frac: float     # wheel centre below the bottom edge / view height
  chip_min_font: int
  chip_max_font: int
  ego_lane_half_m: float = 0.075
  adj_lane_half_m: float = 0.05
  path_half_m: float = 0.9


C3_SIZES = SceneSizes(wheel_radius_frac=0.104, wheel_drop_frac=0.023, chip_min_font=14, chip_max_font=26)
C4_SIZES = SceneSizes(wheel_radius_frac=0.175, wheel_drop_frac=0.042, chip_min_font=9, chip_max_font=13)


def _lead(radar_state, name):
  lead = getattr(radar_state, name, None) if radar_state is not None else None
  return lead if lead is not None and getattr(lead, 'status', False) else None


def _path_lateral(path: np.ndarray, distance: float) -> float:
  if path.shape[0] < 2:
    return 0.0
  return float(np.interp(distance, path[:, 0], path[:, 1]))


def _path_z(path: np.ndarray, distance: float) -> float:
  if path.shape[0] < 2:
    return 0.0
  return float(np.interp(distance, path[:, 0], path[:, 2]))


def _max_idx(xs: np.ndarray, distance: float) -> int:
  idx = np.where(xs <= distance)[0]
  return int(idx[-1]) if idx.size else 0


class View3dScene:
  def __init__(self, sizes: SceneSizes):
    self.sizes = sizes
    self.camera = ChaseCamera()
    self._last_t: float | None = None

  def _dt(self, now: float | None) -> float:
    now = time.monotonic() if now is None else now
    dt = 0.0 if self._last_t is None else float(np.clip(now - self._last_t, 0.0, 0.5))
    self._last_t = now
    return dt

  def draw(self, renderer, sm, rect, is_night: bool, now: float | None = None) -> None:
    path = renderer._path.raw_points
    lanes = renderer._lane_lines
    if path.shape[0] < 2 or len(lanes) == 0 or lanes[0].raw_points.shape[0] == 0:
      return
    ground_z = float(renderer._path_offset_z)
    car_state = sm['carState'] if sm.valid.get('carState', False) else None
    radar_state = sm['radarState'] if sm.valid.get('radarState', False) else None
    v_ego = float(getattr(car_state, 'vEgo', 0.0)) if car_state is not None else 0.0
    steer = float(getattr(car_state, 'steeringAngleDeg', 0.0)) if car_state is not None else 0.0
    left_bs = bool(getattr(car_state, 'leftBlindspot', False))
    right_bs = bool(getattr(car_state, 'rightBlindspot', False))
    lead_one = _lead(radar_state, 'leadOne')
    lead_d = float(lead_one.dRel) if lead_one is not None else None

    cam = self.camera
    curve_lat = _path_lateral(path, 30.0)
    yaw = 0.5 * math.atan2(_path_lateral(path, 20.0), 20.0 + 4.0)
    cam.update(rect, ground_z, zoom_target(v_ego, lead_d, curve_lat), yaw, self._dt(now))

    lane_zero = lanes[0].raw_points
    max_distance = float(np.clip(lane_zero[-1, 0], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE))
    max_idx = _max_idx(lane_zero[:, 0], max_distance)
    pal = view_3d_draw.PALETTE[is_night]

    # 1. Planned path, cut just short of the lead car.
    path_max = float(np.clip(path[-1, 0], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE))
    if lead_d is not None:
      path_max = min(path_max, max(lead_d - 1.0, 2.0))
    p_idx = _max_idx(path[:, 0], path_max)
    poly = cam.ribbon(path, self.sizes.path_half_m, ground_z, p_idx, path_max)
    if poly.size:
      draw_polygon_solid(poly, view_3d_draw._c(*pal['path']))

    # 2. Lane lines: adjacent first, ego lanes on top with green/white/orange state.
    probs = renderer._lane_line_probs
    for i in (0, 3, 1, 2):
      if i >= len(lanes) or i >= len(probs) or probs[i] < view_3d_draw.LANE_MIN_PROB:
        continue
      prob = float(probs[i])
      half = self.sizes.ego_lane_half_m if i in (1, 2) else self.sizes.adj_lane_half_m
      poly = cam.ribbon(lanes[i].raw_points, half, 0.0, max_idx, max_distance)
      if poly.size:
        blocked = view_3d_draw.lane_blocked(car_state, i)
        draw_polygon_solid(poly, view_3d_draw.lane_color(i, prob, is_night, blocked, alpha=min(1.0, 0.55 + prob)))

    # 3. Vehicles far to near so closer cars overlap farther ones.
    vehicles = []
    if radar_state is not None:
      for lead, tag in ((lead_one, 'L1'), (_lead(radar_state, 'leadTwo'), 'L2')):
        if lead is not None and 0.5 < float(lead.dRel) <= MAX_LEAD_DISTANCE:
          if tag == 'L2' and lead_one is not None and abs(float(lead.dRel) - float(lead_one.dRel)) < 3.0:
            continue
          vehicles.append((float(lead.dRel), float(lead.yRel), float(getattr(lead, 'vRel', 0.0)), tag))
      for name in ('leadsLeft', 'leadsRight'):
        for lead in (getattr(radar_state, name, None) or []):
          if getattr(lead, 'status', False) and 1.0 < float(lead.dRel) <= MAX_ADJACENT_DISTANCE:
            vehicles.append((float(lead.dRel), float(lead.yRel), 0.0, 'ADJ'))
    for d_rel, y_rel, v_rel, tag in sorted(vehicles, key=lambda v: -v[0]):
      box = cam.car_box(d_rel, -y_rel, _path_z(path, d_rel) + ground_z)
      if box is None:
        continue
      x, y, vw, vh = box
      style = 'adj' if tag == 'ADJ' else 'lead'
      view_3d_draw.draw_vehicle_rear(x, y, vw, vh, is_night, style=style, braking=v_rel < -0.8 and tag != 'ADJ')
      if tag in ('L1', 'L2'):
        v_kph = v_rel * 3.6
        text = f"{tag} {d_rel:.0f}m" + (f"  {v_kph:+.0f}km/h" if tag == 'L1' and abs(v_kph) >= 1.0 else "")
        ttc = -d_rel / v_rel if v_rel < -0.5 else 999.0
        view_3d_draw.draw_metric_chip(x, y - vh * 0.5, text, vw, is_night, tag == 'L1', ttc < 2.5, ttc < 4.5,
                                      self.sizes.chip_min_font, self.sizes.chip_max_font)

    # 4. Ego: wheel in the cruise view, car with halo in the wide view.
    self.draw_ego(rect, is_night, steer, left_bs, right_bs)

  def draw_ego(self, rect, is_night: bool, steer_deg: float, left_bs: bool, right_bs: bool) -> None:
    cam = self.camera
    zoom = cam.zoom
    wheel_alpha = float(np.clip(1.0 - zoom * 1.6, 0.0, 1.0))
    car_alpha = float(np.clip((zoom - 0.25) / 0.45, 0.0, 1.0))
    if car_alpha > 0.0:
      box = cam.ego_box()
      if box is not None:
        x, y, vw, vh = box
        view_3d_draw.draw_ego_halo(x, y + vh * 0.42, vw, is_night, left_bs, right_bs, car_alpha)
        view_3d_draw.draw_vehicle_rear(x, y, vw, vh, is_night, style='ego', alpha=car_alpha)
    if wheel_alpha > 0.0:
      h = float(rect.height)
      steer = float(np.clip(-steer_deg, -75.0, 75.0))
      view_3d_draw.draw_steering_wheel(float(rect.x + rect.width / 2), float(rect.y + h * (1.0 + self.sizes.wheel_drop_frac)),
                                       h * self.sizes.wheel_radius_frac, steer, is_night, left_bs, right_bs, wheel_alpha)


# Backward compatibility aliases
FsdScene = View3dScene

__all__ = ['View3dScene', 'FsdScene', 'C3_SIZES', 'C4_SIZES', 'EGO_REAR_X_M', 'draw_polygon_solid']

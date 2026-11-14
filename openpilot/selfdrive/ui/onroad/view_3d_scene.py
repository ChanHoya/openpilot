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
  adj_lane_half_m: float = 0.06
  road_edge_half_m: float = 0.070
  path_half_m: float = 0.80


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

    # 1. Driving Path (center vehicle-width ribbon, clean without cluttering inner stripes)
    path_max = float(np.clip(path[-1, 0], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE))
    if lead_d is not None:
      path_max = min(path_max, max(lead_d - 1.0, 2.0))
    p_idx = _max_idx(path[:, 0], path_max)

    selfdrive_state = sm['selfdriveState'] if sm.valid.get('selfdriveState', False) else None
    long_active = bool(getattr(selfdrive_state, 'enabled', False)) if selfdrive_state is not None else False
    lp = sm['longitudinalPlan'] if sm.valid.get('longitudinalPlan', False) else None
    accel = float(lp.accels[0]) if lp is not None and len(getattr(lp, 'accels', [])) > 0 else 0.0
    controls_state = sm['controlsState'] if sm.valid.get('controlsState', False) else None
    active_lane = (bool(getattr(controls_state, 'activeLaneLine', False)) if controls_state is not None
                   else bool(getattr(renderer, '_carrot_active_lane_line', False)))

    if active_lane:
      show_color = getattr(renderer, '_carrot_show_path_color_lane', getattr(renderer, '_carrot_show_path_color', 13))
    else:
      show_color = getattr(renderer, '_carrot_show_path_color_normal', getattr(renderer, '_carrot_show_path_color', 13))
    if not long_active:
      show_color = getattr(renderer, '_carrot_show_path_color_cruise_off', show_color)

    if show_color >= 20:
      if long_active:
        show_color = 13
        if lead_one is not None and getattr(lead_one, 'status', False):
          if abs(accel) < 0.5:
            show_color = 12
          elif accel >= 0.5:
            show_color = 11
          else:
            show_color = 10
      else:
        show_color = 19

    carrot_colors = getattr(renderer, '_carrot_colors', None)
    if carrot_colors and len(carrot_colors) > (show_color % 10):
      path_fill = carrot_colors[show_color % 10]
    else:
      pal = view_3d_draw.PALETTE[is_night]
      path_fill = view_3d_draw._c(0, 180, 50, 150) if is_night else view_3d_draw._c(*pal['path'])

    half_m = self.sizes.path_half_m
    poly_path = cam.ribbon(path, half_m, ground_z, p_idx, path_max, y_shift=0.0)
    if poly_path.size:
      draw_polygon_solid(poly_path, path_fill)

    # 2. Tire Trajectories flanking left & right edges of the driving path
    probs = renderer._lane_line_probs
    model = sm['modelV2'] if sm.valid.get('modelV2', False) else None
    left_y0 = right_y0 = None
    p_left = p_right = 0.0
    if model is not None and len(getattr(model, 'laneLines', [])) >= 3 and len(getattr(model, 'laneLineProbs', [])) >= 3:
      ly, ry = model.laneLines[1].y, model.laneLines[2].y
      if len(ly) > 0 and len(ry) > 0:
        left_y0, right_y0 = float(ly[0]), float(ry[0])
        p_left, p_right = float(model.laneLineProbs[1]), float(model.laneLineProbs[2])
    elif len(lanes) >= 3 and len(probs) >= 3 and lanes[1].raw_points.shape[0] > 0 and lanes[2].raw_points.shape[0] > 0:
      left_y0 = float(lanes[1].raw_points[0, 1])
      right_y0 = float(lanes[2].raw_points[0, 1])
      p_left, p_right = float(probs[1]), float(probs[2])

    if left_y0 is not None and right_y0 is not None:
      # Device frame: -Y is left, +Y is right. Distance to left lane is -left_y0, to right lane is +right_y0.
      left_m = -left_y0 if left_y0 <= 0.0 or right_y0 >= 0.0 else left_y0
      right_m = right_y0 if right_y0 >= 0.0 or left_y0 <= 0.0 else -right_y0
      if p_left < 0.3 and p_right > 0.3:
        left_m = 3.0 - right_m
      elif p_right < 0.3 and p_left > 0.3:
        right_m = 3.0 - left_m
      total = max(0.5, left_m + right_m)
      drift = (right_m - left_m) / 2.0
      limit = max(0.1, total / 2.0 - 0.9)
      danger = abs(drift) / limit

      green = view_3d_draw._c(0, 200, 0, 210)
      warn = abs(drift) >= 0.5
      pulse = warn and (int(time.monotonic() * 1000) % 800 < 400)
      if warn:
        side = view_3d_draw._c(255, 0, 0, 255) if pulse else view_3d_draw._c(255, 100, 0, 210)
        col_l = side if drift > 0.0 else green
        col_r = side if drift < 0.0 else green
      elif drift > 0.02:
        col_l, col_r = view_3d_draw.tire_grad_color(danger), green
      elif drift < -0.02:
        col_l, col_r = green, view_3d_draw.tire_grad_color(danger)
      else:
        col_l = col_r = green

      tire_half = 0.12
      tire_offset = half_m + tire_half
      # Screen left = -Y, screen right = +Y.
      for y_shift, col in ((-tire_offset, col_l), (tire_offset, col_r)):
        poly = cam.ribbon(path, tire_half, ground_z, p_idx, path_max, y_shift=y_shift)
        if poly.size:
          draw_polygon_solid(poly, col)

    # 3. Recognized Ego Lane Lines (status colored) outside the tire trajectories
    for i in (1, 2):
      if i >= len(lanes) or i >= len(probs) or probs[i] < view_3d_draw.LANE_MIN_PROB:
        continue
      prob = float(probs[i])
      poly = cam.ribbon(lanes[i].raw_points, self.sizes.ego_lane_half_m, 0.0, max_idx, max_distance)
      if poly.size:
        blocked = view_3d_draw.lane_blocked(car_state, i)
        draw_polygon_solid(poly, view_3d_draw.lane_color(i, prob, is_night, blocked, alpha=min(1.0, 0.70 + prob)))

    # 4. Outermost Road Edges outside the lane lines
    road_edges = getattr(renderer, '_road_edges', None)
    road_edge_stds = getattr(renderer, '_road_edge_stds', None)
    if road_edges:
      for i, edge in enumerate(road_edges):
        raw = getattr(edge, 'raw_points', None)
        if raw is None or raw.shape[0] == 0:
          continue
        std = float(road_edge_stds[i]) if road_edge_stds is not None and i < len(road_edge_stds) else 1.0
        if std >= view_3d_draw.ROAD_EDGE_MAX_STD:
          continue
        edge_max_idx = _max_idx(raw[:, 0], max_distance)
        poly = cam.ribbon(raw, self.sizes.road_edge_half_m, 0.0, edge_max_idx, max_distance)
        if poly.size:
          draw_polygon_solid(poly, view_3d_draw.road_edge_color(std, is_night))

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

    # 4. Ego: wheel glides into cockpit, car with halo in the wide view.
    self.draw_ego(rect, is_night, steer, left_bs, right_bs, ground_z)

  def draw_ego(self, rect, is_night: bool, steer_deg: float, left_bs: bool, right_bs: bool, ground_z: float = 1.22) -> None:
    """Consistent windshield cockpit frame with steering wheel on the left driver side."""
    h = float(rect.height)
    wheel_x = float(rect.x + rect.width * 0.40)
    wheel_y = float(rect.y + h * (1.0 + self.sizes.wheel_drop_frac))
    wheel_r = h * self.sizes.wheel_radius_frac
    view_3d_draw.draw_windshield_frame(rect, wheel_x, is_night, left_bs, right_bs)
    steer = float(np.clip(-steer_deg, -75.0, 75.0))
    view_3d_draw.draw_steering_wheel(wheel_x, wheel_y, wheel_r, steer, is_night, left_bs, right_bs, 1.0)


# Backward compatibility aliases
FsdScene = View3dScene

__all__ = ['View3dScene', 'FsdScene', 'C3_SIZES', 'C4_SIZES', 'EGO_REAR_X_M', 'draw_polygon_solid']

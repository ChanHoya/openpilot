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

    # 1. Planned path (AR-View parity: Carrot two-quad ribbon with center & border lines)
    path_max = float(np.clip(path[-1, 0], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE))
    if lead_d is not None:
      path_max = min(path_max, max(lead_d - 1.0, 2.0))
    p_idx = _max_idx(path[:, 0], path_max)

    # 1A. Tire Trajectory (AR-View parity: lane-departure danger colored bands at +-0.9m)
    model = sm['modelV2'] if sm.valid.get('modelV2', False) else None
    if model is not None and len(getattr(model, 'laneLines', [])) >= 3 and len(getattr(model, 'laneLineProbs', [])) >= 3:
      probs = model.laneLineProbs
      left_y, right_y = model.laneLines[1].y, model.laneLines[2].y
      if len(left_y) > 0 and len(right_y) > 0:
        left_m = float(left_y[0])
        right_m = -float(right_y[0])
        if probs[1] < 0.3 and probs[2] > 0.3:
          left_m = 3.0 - right_m
        elif probs[2] < 0.3 and probs[1] > 0.3:
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

        tire_half = 0.16
        for y_shift, col in ((0.9, col_l), (-0.9, col_r)):
          poly = cam.ribbon(path, tire_half, ground_z, p_idx, path_max, y_shift=y_shift)
          if poly.size:
            draw_polygon_solid(poly, col)

    # 1B. Driving Path (AR-View parity: Carrot two-quad ribbon with center & border lines)
    brake_valid = bool(getattr(car_state, 'brakeLights', False)) if car_state is not None else False
    selfdrive_state = sm['selfdriveState'] if sm.valid.get('selfdriveState', False) else None
    long_active = bool(getattr(selfdrive_state, 'enabled', False)) if selfdrive_state is not None else False
    lp = sm['longitudinalPlan'] if sm.valid.get('longitudinalPlan', False) else None
    accel = float(lp.accels[0]) if lp is not None and len(getattr(lp, 'accels', [])) > 0 else 0.0

    show_color = getattr(renderer, '_carrot_show_path_color', 13)
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
      path_fill = view_3d_draw._c(0, 180, 50, 160) if is_night else view_3d_draw._c(*pal['path'])

    border_color = view_3d_draw._c(255, 40, 40, 240) if brake_valid else view_3d_draw._c(255, 255, 255, 240)

    half_m = self.sizes.path_half_m
    # Left and right quads
    poly_l = cam.ribbon(path, half_m * 0.48, ground_z, p_idx, path_max, y_shift=half_m * 0.50)
    if poly_l.size:
      draw_polygon_solid(poly_l, path_fill)
    poly_r = cam.ribbon(path, half_m * 0.48, ground_z, p_idx, path_max, y_shift=-half_m * 0.50)
    if poly_r.size:
      draw_polygon_solid(poly_r, path_fill)

    # Center divider line & outer borders
    poly_c = cam.ribbon(path, 0.025, ground_z, p_idx, path_max, y_shift=0.0)
    if poly_c.size:
      draw_polygon_solid(poly_c, border_color)
    poly_el = cam.ribbon(path, 0.025, ground_z, p_idx, path_max, y_shift=half_m)
    if poly_el.size:
      draw_polygon_solid(poly_el, border_color)
    poly_er = cam.ribbon(path, 0.025, ground_z, p_idx, path_max, y_shift=-half_m)
    if poly_er.size:
      draw_polygon_solid(poly_er, border_color)

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

    # 4. Ego: wheel glides into cockpit, car with halo in the wide view.
    self.draw_ego(rect, is_night, steer, left_bs, right_bs, ground_z)

  def draw_ego(self, rect, is_night: bool, steer_deg: float, left_bs: bool, right_bs: bool, ground_z: float = 1.22) -> None:
    cam = self.camera
    zoom = cam.zoom
    car_alpha = float(np.clip((zoom - 0.20) / 0.45, 0.0, 1.0))

    h = float(rect.height)
    start_x = float(rect.x + rect.width / 2)
    start_y = float(rect.y + h * (1.0 + self.sizes.wheel_drop_frac))
    start_r = h * self.sizes.wheel_radius_frac

    # Driver cockpit 3D position inside the vehicle:
    # EGO_REAR_X_M is rear bumper (-2.6m). Driver seat wheel is about +1.4m ahead (-1.2m),
    # left-hand drive side y=-0.32m, height 0.72m above ground.
    cockpit_3d = (EGO_REAR_X_M + 1.4, -0.32, ground_z - 0.72)
    cockpit_pt = cam.project_point(*cockpit_3d)
    cockpit_scale = cam.scale_at(*cockpit_3d)

    if cockpit_pt is not None and cockpit_scale > 0.0:
      target_x, target_y = cockpit_pt
      target_r = max(cockpit_scale * 0.18, 12.0)
    else:
      target_x, target_y = start_x, float(rect.y + h * 0.82)
      target_r = max(start_r * 0.25, 15.0)

    # Smoothstep interpolation from cruise bottom to cockpit interior
    t = float(np.clip(zoom / 0.80, 0.0, 1.0))
    t_smooth = t * t * (3.0 - 2.0 * t)
    cur_x = start_x + (target_x - start_x) * t_smooth
    cur_y = start_y + (target_y - start_y) * t_smooth
    cur_r = start_r + (target_r - start_r) * t_smooth

    # Wheel alpha: starts solid 1.0 in cruise, stays fully visible as it glides up into cockpit,
    # then softly integrates once settled inside
    wheel_alpha = float(np.clip(1.0 - (zoom - 0.70) / 0.30, 0.35, 1.0))

    # 1. Halo under ego car (when car is appearing)
    if car_alpha > 0.0:
      box = cam.ego_box()
      if box is not None:
        x, y, vw, vh = box
        view_3d_draw.draw_ego_halo(x, y + vh * 0.42, vw, is_night, left_bs, right_bs, car_alpha)

    # 2. Steering wheel glides into cockpit
    steer = float(np.clip(-steer_deg, -75.0, 75.0))
    view_3d_draw.draw_steering_wheel(cur_x, cur_y, cur_r, steer, is_night, left_bs, right_bs, wheel_alpha)

    # 3. 3D Ego vehicle rear on top (its tinted rear windshield settles over the steering wheel)
    if car_alpha > 0.0:
      box = cam.ego_box()
      if box is not None:
        x, y, vw, vh = box
        view_3d_draw.draw_vehicle_rear(x, y, vw, vh, is_night, style='ego', alpha=car_alpha)


# Backward compatibility aliases
FsdScene = View3dScene

__all__ = ['View3dScene', 'FsdScene', 'C3_SIZES', 'C4_SIZES', 'EGO_REAR_X_M', 'draw_polygon_solid']

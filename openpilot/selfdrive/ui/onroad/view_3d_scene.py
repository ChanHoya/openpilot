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


def _ema_alpha(dt: float, tau: float) -> float:
  if tau <= 0.0:
    return 1.0
  dt_eff = dt if dt > 0.0 else 0.05
  return float(1.0 - math.exp(-dt_eff / tau))


class View3dScene:
  def __init__(self, sizes: SceneSizes):
    self.sizes = sizes
    self.camera = ChaseCamera()
    self._last_t: float | None = None
    self._smooth_ground_z: float | None = None
    self._smooth_path_max: float | None = None
    self._smooth_drift: float | None = None
    self._smooth_path_grid_y: np.ndarray | None = None
    self._smooth_path_grid_z: np.ndarray | None = None
    self._vehicle_tracks: dict[str, dict[str, float]] = {}

  def _dt(self, now: float | None) -> tuple[float, float]:
    now_t = time.monotonic() if now is None else float(now)
    dt = 0.0 if self._last_t is None else float(np.clip(now_t - self._last_t, 0.0, 0.5))
    self._last_t = now_t
    return dt, now_t

  def _stabilize_vehicle(self, key: str, d_rel: float, y_rel: float, v_rel: float, z_rel: float,
                         v_ego: float, dt: float, now_t: float) -> tuple[float, float, float, float, int]:
    tr = self._vehicle_tracks.get(key)
    stationary = v_ego < 0.3 and abs(v_rel) < 0.5
    if tr is None or (now_t - tr['t']) > 1.5 or abs(d_rel - tr['d']) > 5.0 or abs(y_rel - tr['y']) > 2.5:
      v_init = 0.0 if (stationary and abs(v_rel) < 0.4) else v_rel
      z_init = 0.0 if v_ego < 0.3 else z_rel
      disp_d = int(round(d_rel))
      self._vehicle_tracks[key] = {
        'd': d_rel, 'y': y_rel, 'v': v_init, 'z': z_init, 'disp_d': float(disp_d), 't': now_t,
      }
      return d_rel, y_rel, v_init, z_init, disp_d

    if stationary:
      if abs(d_rel - tr['d']) >= 0.55:
        tr['d'] += (d_rel - tr['d']) * _ema_alpha(dt, 0.75)
      if abs(y_rel - tr['y']) >= 0.25:
        tr['y'] += (y_rel - tr['y']) * _ema_alpha(dt, 0.85)
    else:
      tau_d = 0.45 if abs(d_rel - tr['d']) < 0.08 else 0.18
      tr['d'] += (d_rel - tr['d']) * _ema_alpha(dt, tau_d)
      tr['y'] += (y_rel - tr['y']) * _ema_alpha(dt, 0.25)

    tr['v'] += (v_rel - tr['v']) * _ema_alpha(dt, 0.30)
    z_target = 0.0 if v_ego < 0.3 else z_rel
    tr['z'] += (z_target - tr['z']) * _ema_alpha(dt, 0.60)
    tr['t'] = now_t

    v_out = 0.0 if (stationary and abs(tr['v']) < 0.4) else tr['v']
    if abs(tr['d'] - tr['disp_d']) >= 0.65:
      tr['disp_d'] = float(round(tr['d']))
    return tr['d'], tr['y'], v_out, tr['z'], int(tr['disp_d'])

  def _stabilize_path(self, path: np.ndarray, lanes: list, path_max: float, v_ego: float, dt: float) -> tuple[np.ndarray, float]:
    stationary = v_ego < 0.3
    if self._smooth_path_max is None:
      self._smooth_path_max = path_max
    elif not stationary or abs(path_max - self._smooth_path_max) >= 0.40:
      tau_len = 1.0 if stationary else 0.22
      self._smooth_path_max += (path_max - self._smooth_path_max) * _ema_alpha(dt, tau_len)
    smooth_max = float(max(self._smooth_path_max, 2.0))

    x_start = max(0.0, float(path[0, 0]))
    if smooth_max <= x_start + 0.5:
      smooth_max = x_start + 0.5
    xs = np.linspace(x_start, smooth_max, 33, dtype=np.float64)

    path_x = path[:, 0].astype(np.float64)
    path_y = path[:, 1].astype(np.float64)
    path_z = path[:, 2].astype(np.float64)
    path_end_x = float(path_x[-1])

    target_y = np.interp(xs, path_x, path_y)
    target_z = np.interp(xs, path_x, path_z)

    # Extend stopped/collapsed model path out to smooth_max cleanly using lane midpoint if available.
    ext_mask = xs > path_end_x
    if np.any(ext_mask) and len(lanes) >= 3:
      l1 = getattr(lanes[1], 'raw_points', None)
      l2 = getattr(lanes[2], 'raw_points', None)
      if (l1 is not None and l2 is not None and l1.shape[0] >= 2 and l2.shape[0] >= 2
          and l1[-1, 0] > path_end_x and l2[-1, 0] > path_end_x):
        xs_ext = xs[ext_mask]
        mid_ext = 0.5 * (np.interp(xs_ext, l1[:, 0], l1[:, 1]) + np.interp(xs_ext, l2[:, 0], l2[:, 1]))
        mid_end = 0.5 * (np.interp(path_end_x, l1[:, 0], l1[:, 1]) + np.interp(path_end_x, l2[:, 0], l2[:, 1]))
        target_y[ext_mask] = path_y[-1] + (mid_ext - mid_end)

    if self._smooth_path_grid_y is None or self._smooth_path_grid_z is None:
      self._smooth_path_grid_y = target_y.copy()
      self._smooth_path_grid_z = target_z.copy()
    elif stationary:
      max_dy = float(np.max(np.abs(target_y - self._smooth_path_grid_y)))
      max_dz = float(np.max(np.abs(target_z - self._smooth_path_grid_z)))
      if max_dy >= 0.18:
        self._smooth_path_grid_y += (target_y - self._smooth_path_grid_y) * _ema_alpha(dt, 0.90)
      if max_dz >= 0.18:
        self._smooth_path_grid_z += (target_z - self._smooth_path_grid_z) * _ema_alpha(dt, 0.90)
    else:
      alpha_p = _ema_alpha(dt, 0.12)
      self._smooth_path_grid_y += (target_y - self._smooth_path_grid_y) * alpha_p
      self._smooth_path_grid_z += (target_z - self._smooth_path_grid_z) * alpha_p

    stab_path = np.stack((xs, self._smooth_path_grid_y, self._smooth_path_grid_z), axis=1).astype(np.float32)
    return stab_path, smooth_max

  def draw(self, renderer, sm, rect, is_night: bool, now: float | None = None) -> None:
    path = renderer._path.raw_points
    lanes = renderer._lane_lines
    if path.shape[0] < 2 or len(lanes) == 0 or lanes[0].raw_points.shape[0] == 0:
      return
    dt, now_t = self._dt(now)
    raw_ground_z = float(renderer._path_offset_z)
    if self._smooth_ground_z is None:
      self._smooth_ground_z = raw_ground_z
    else:
      self._smooth_ground_z += (raw_ground_z - self._smooth_ground_z) * _ema_alpha(dt, 0.50)
    ground_z = self._smooth_ground_z

    car_state = sm['carState'] if sm.valid.get('carState', False) else None
    radar_state = sm['radarState'] if sm.valid.get('radarState', False) else None
    v_ego = float(getattr(car_state, 'vEgo', 0.0)) if car_state is not None else 0.0
    steer = float(getattr(car_state, 'steeringAngleDeg', 0.0)) if car_state is not None else 0.0
    left_bs = bool(getattr(car_state, 'leftBlindspot', False))
    right_bs = bool(getattr(car_state, 'rightBlindspot', False))
    lead_one = _lead(radar_state, 'leadOne')

    # Prune stale vehicle tracks before updating active detections
    for k in [k for k, tr in self._vehicle_tracks.items() if now_t - tr['t'] > 2.0]:
      del self._vehicle_tracks[k]

    stab_lead_d = None
    if lead_one is not None and 0.5 < float(lead_one.dRel) <= MAX_LEAD_DISTANCE:
      raw_d = float(lead_one.dRel)
      stab_lead_d, _, _, _, _ = self._stabilize_vehicle(
        'L1', raw_d, float(lead_one.yRel), float(getattr(lead_one, 'vRel', 0.0)),
        _path_z(path, raw_d), v_ego, dt, now_t,
      )
    lead_d = stab_lead_d

    cam = self.camera
    curve_lat = _path_lateral(path, 30.0)
    if v_ego < 0.4 or float(path[-1, 0]) < 12.0:
      yaw = 0.0
    else:
      yaw = 0.5 * math.atan2(_path_lateral(path, 20.0), 20.0 + 4.0)
    cam.update(rect, ground_z, zoom_target(v_ego, lead_d, curve_lat), yaw, dt)

    lane_zero = lanes[0].raw_points
    max_distance = float(np.clip(lane_zero[-1, 0], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE))
    max_idx = _max_idx(lane_zero[:, 0], max_distance)

    # 1. Driving Path (center vehicle-width ribbon, clean and stabilized when stopped)
    raw_path_max = float(np.clip(path[-1, 0], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE))
    if lead_d is not None:
      raw_path_max = min(raw_path_max, max(lead_d - 1.0, 2.0))
    stab_path, path_max = self._stabilize_path(path, lanes, raw_path_max, v_ego, dt)
    p_idx = len(stab_path) - 1

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
    poly_path = cam.ribbon(stab_path, half_m, ground_z, p_idx, path_max, y_shift=0.0)
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
      raw_drift = (right_m - left_m) / 2.0
      if self._smooth_drift is None:
        self._smooth_drift = raw_drift
      else:
        self._smooth_drift += (raw_drift - self._smooth_drift) * _ema_alpha(dt, 0.80 if v_ego < 0.3 else 0.25)
      drift = self._smooth_drift
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
        poly = cam.ribbon(stab_path, tire_half, ground_z, p_idx, path_max, y_shift=y_shift)
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

    # 5. Vehicles far to near so closer cars overlap farther ones (with stationary noise filtering).
    vehicles = []
    if radar_state is not None:
      for lead, tag in ((lead_one, 'L1'), (_lead(radar_state, 'leadTwo'), 'L2')):
        if lead is not None and 0.5 < float(lead.dRel) <= MAX_LEAD_DISTANCE:
          if tag == 'L2' and lead_one is not None and abs(float(lead.dRel) - float(lead_one.dRel)) < 3.0:
            continue
          if tag == 'L1' and 'L1' in self._vehicle_tracks:
            tr = self._vehicle_tracks['L1']
            v_l1 = 0.0 if (v_ego < 0.3 and abs(tr['v']) < 0.4) else tr['v']
            vehicles.append((tr['d'], tr['y'], v_l1, tr['z'], int(tr['disp_d']), 'L1'))
          else:
            raw_d = float(lead.dRel)
            sd, sy, sv, sz, disp_d = self._stabilize_vehicle(
              tag, raw_d, float(lead.yRel), float(getattr(lead, 'vRel', 0.0)),
              _path_z(path, raw_d), v_ego, dt, now_t,
            )
            vehicles.append((sd, sy, sv, sz, disp_d, tag))
      for name in ('leadsLeft', 'leadsRight'):
        for idx_adj, lead in enumerate(getattr(radar_state, name, None) or []):
          if getattr(lead, 'status', False) and 1.0 < float(lead.dRel) <= MAX_ADJACENT_DISTANCE:
            raw_d = float(lead.dRel)
            sd, sy, sv, sz, disp_d = self._stabilize_vehicle(
              f'ADJ_{name}_{idx_adj}', raw_d, float(lead.yRel), 0.0,
              _path_z(path, raw_d), v_ego, dt, now_t,
            )
            vehicles.append((sd, sy, sv, sz, disp_d, 'ADJ'))
    for d_rel, y_rel, v_rel, z_rel, disp_d, tag in sorted(vehicles, key=lambda v: -v[0]):
      box = cam.car_box(d_rel, -y_rel, z_rel + ground_z)
      if box is None:
        continue
      x, y, vw, vh = box
      style = 'adj' if tag == 'ADJ' else 'lead'
      view_3d_draw.draw_vehicle_rear(x, y, vw, vh, is_night, style=style, braking=v_rel < -0.8 and tag != 'ADJ')
      if tag in ('L1', 'L2'):
        v_kph = v_rel * 3.6
        text = f"{tag} {disp_d}m" + (f"  {v_kph:+.0f}km/h" if tag == 'L1' and abs(v_kph) >= 1.0 else "")
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

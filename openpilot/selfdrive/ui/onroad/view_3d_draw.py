"""Shared 3D-view drawing for C3/C4: vehicles, lanes, ego halo and wheel."""
import math

import numpy as np
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib import text_measure

LANE_MIN_PROB = 0.30
LANE_GOOD_PROB = 0.50


def _c(r: int, g: int, b: int, a: int = 255, alpha: float = 1.0) -> rl.Color:
  return rl.Color(int(r), int(g), int(b), int(np.clip(a * alpha, 0, 255)))


def _mix(c0, c1, t: float, alpha: float = 1.0) -> rl.Color:
  t = float(np.clip(t, 0.0, 1.0))
  return _c(*(c0[i] + (c1[i] - c0[i]) * t for i in range(3)), a=c0[3] + (c1[3] - c0[3]) * t, alpha=alpha)


def tire_grad_color(d: float, alpha: float = 1.0) -> rl.Color:
  """danger 0(center) -> 1(touching lane): green -> yellow -> red multi-stop."""
  d = max(0.0, min(1.5, float(d)))
  if d <= 0.10:
    r, g = 0.0, 200.0
  elif d <= 0.60:
    f = (d - 0.10) / 0.50
    r, g = f * 255.0, 200.0
  elif d <= 1.00:
    f = (d - 0.60) / 0.40
    r, g = 255.0, 200.0 - f * 200.0
  else:
    r, g = 255.0, 0.0
  return _c(int(r), int(g), 0, 220, alpha=alpha)


# Palettes keep vehicles readable on the dark night tarmac and the light day tarmac.
PALETTE = {
  True: {   # night
    'path': (0, 190, 255, 120), 'path_edge': (0, 225, 255, 210),
    'lane_good': (0, 230, 118, 255), 'lane_weak': (245, 248, 252, 230), 'lane_block': (255, 145, 0, 255),
    'lane_adj': (190, 205, 225, 170),
    'lead_top': (205, 214, 228, 255), 'lead_bottom': (120, 130, 146, 255), 'lead_edge': (235, 242, 252, 255),
    'adj_top': (150, 158, 172, 235), 'adj_bottom': (92, 100, 114, 235), 'adj_edge': (175, 184, 198, 235),
    'ego_top': (74, 80, 92, 255), 'ego_bottom': (26, 30, 38, 255), 'ego_edge': (140, 150, 168, 255),
    'glass': (14, 20, 30, 255), 'halo': (0, 200, 255, 255),
  },
  False: {  # day
    'path': (24, 118, 238, 120), 'path_edge': (16, 95, 210, 230),
    'lane_good': (0, 185, 80, 255), 'lane_weak': (255, 255, 255, 255), 'lane_block': (245, 120, 0, 255),
    'lane_adj': (235, 242, 255, 200),
    'lead_top': (78, 92, 116, 255), 'lead_bottom': (34, 42, 56, 255), 'lead_edge': (215, 228, 246, 255),
    'adj_top': (125, 135, 150, 240), 'adj_bottom': (78, 88, 104, 240), 'adj_edge': (60, 70, 86, 255),
    'ego_top': (62, 68, 80, 255), 'ego_bottom': (18, 22, 28, 255), 'ego_edge': (190, 200, 216, 255),
    'glass': (18, 24, 34, 255), 'halo': (0, 150, 255, 255),
  },
}


def lane_color(index: int, prob: float, is_night: bool, blocked: bool, alpha: float = 1.0) -> rl.Color:
  """Ego lanes: orange when a change is blocked, green when recognised, white when weak."""
  pal = PALETTE[is_night]
  if index not in (1, 2):
    return _c(*pal['lane_adj'], alpha=alpha)
  if blocked:
    return _c(*pal['lane_block'], alpha=alpha)
  if prob >= LANE_GOOD_PROB:
    return _c(*pal['lane_good'], alpha=alpha)
  return _c(*pal['lane_weak'], alpha=alpha)


def lane_blocked(car_state, index: int) -> bool:
  """Blind-spot vehicle or a classified solid line on that side."""
  if car_state is None or index not in (1, 2):
    return False
  left = index == 1
  if bool(getattr(car_state, 'leftBlindspot' if left else 'rightBlindspot', False)):
    return True
  code = getattr(car_state, 'leftLaneLine' if left else 'rightLaneLine', -1)
  try:
    code = int(code)
  except (TypeError, ValueError):
    return False
  # Same lane-code convention as the AR view: >= 0 classified, % 10 == 0 dashed.
  return code >= 0 and code % 10 != 0


def _rounded(x, y, w, h, roundness, color, segments=10):
  if w <= 0 or h <= 0:
    return
  rl.draw_rectangle_rounded(rl.Rectangle(x, y, w, h), roundness, segments, color)


def draw_vehicle_rear(x: float, y: float, vw: float, vh: float, is_night: bool,
                      style: str = 'lead', braking: bool = False, alpha: float = 1.0) -> None:
  """Smooth, shaded rear view of a car centred at (x, y) with size (vw, vh)."""
  if vw < 2 or alpha <= 0.01:
    return
  pal = PALETTE[is_night]
  top, bottom, edge = pal[f'{style}_top'], pal[f'{style}_bottom'], pal[f'{style}_edge']
  seg = 12 if vw > 80 else 6
  edge_w = max(1.0, vw * 0.006)

  # Soft contact shadow: stacked ellipses fade outward.
  for i, (sx, sy, a) in enumerate(((0.66, 0.16, 50), (0.58, 0.12, 80), (0.48, 0.08, 120))):
    rl.draw_ellipse(int(x), int(y + vh * 0.47), vw * sx, max(vh * sy, 1.0), _c(0, 0, 0, a, alpha))

  # Tyres peek out under the bumper.
  tw, th = vw * 0.15, vh * 0.16
  _rounded(x - vw * 0.45, y + vh * 0.34, tw, th, 0.6, _c(12, 14, 18, 255, alpha), seg)
  _rounded(x + vw * 0.45 - tw, y + vh * 0.34, tw, th, 0.6, _c(12, 14, 18, 255, alpha), seg)

  # Cabin (greenhouse) narrower than the body, very round roof.
  cab_w, cab_y, cab_h = vw * 0.74, y - vh * 0.50, vh * 0.50
  _rounded(x - cab_w / 2, cab_y, cab_w, cab_h, 0.75, _mix(top, bottom, 0.15, alpha), seg)
  # Roof highlight.
  _rounded(x - cab_w * 0.36, cab_y + vh * 0.015, cab_w * 0.72, vh * 0.05, 1.0, _c(255, 255, 255, 55, alpha), seg)

  # Rear glass with a faint reflection band.
  gw, gy, gh = cab_w * 0.80, cab_y + vh * 0.07, vh * 0.27
  _rounded(x - gw / 2, gy, gw, gh, 0.55, _c(*pal['glass'], alpha=alpha), seg)
  if vw > 40:
    rl.draw_rectangle_gradient_v(int(x - gw * 0.36), int(gy + gh * 0.12), int(gw * 0.72), int(max(gh * 0.30, 1)),
                                 _c(170, 200, 240, 70, alpha), _c(170, 200, 240, 0, alpha))

  # Body: shoulder band blends cabin into the wide hips, then the bumper.
  body_y, body_h = y - vh * 0.12, vh * 0.52
  _rounded(x - vw / 2, body_y, vw, body_h, 0.55, _mix(top, bottom, 0.45, alpha), seg)
  # Vertical shading: lighter upper body, darker lower bumper.
  _rounded(x - vw * 0.48, body_y + vh * 0.02, vw * 0.96, body_h * 0.42, 0.6, _mix(top, bottom, 0.10, alpha), seg)
  _rounded(x - vw * 0.46, body_y + body_h * 0.62, vw * 0.92, body_h * 0.36, 0.6, _mix(top, bottom, 0.85, alpha), seg)
  rl.draw_rectangle_rounded_lines_ex(rl.Rectangle(x - vw / 2, body_y, vw, body_h), 0.55, seg, edge_w, _c(*edge[:3], a=edge[3] * 0.55, alpha=alpha))
  # Shoulder crease highlight.
  _rounded(x - vw * 0.44, body_y + vh * 0.01, vw * 0.88, max(vh * 0.022, 1.0), 1.0, _c(255, 255, 255, 70, alpha), seg)

  # Plate recess and diffuser.
  pw, ph = vw * 0.26, vh * 0.10
  _rounded(x - pw / 2, y + vh * 0.12, pw, ph, 0.35, _c(18, 22, 28, 220, alpha), seg)
  _rounded(x - vw * 0.30, y + vh * 0.31, vw * 0.60, vh * 0.06, 1.0, _c(12, 14, 18, 200, alpha), seg)

  # Taillights: side pods plus a slim full-width light bar.
  lit = (255, 40, 40) if braking else ((235, 32, 32) if is_night else (210, 25, 25))
  tl_w, tl_h, tl_y = vw * 0.26, max(vh * 0.075, 1.5), y - vh * 0.05
  if braking:
    for grow, a in ((0.09, 40), (0.05, 70), (0.025, 110)):
      _rounded(x - vw * 0.48 - vw * grow, tl_y - vh * grow, vw * 0.96 + 2 * vw * grow, tl_h + 2 * vh * grow,
               1.0, _c(255, 30, 30, a, alpha), seg)
    _rounded(x - vw * 0.12, cab_y + vh * 0.025, vw * 0.24, max(vh * 0.03, 1.0), 1.0, _c(*lit, 255, alpha), seg)
  _rounded(x - vw * 0.47, tl_y, tl_w, tl_h, 1.0, _c(*lit, 255, alpha), seg)
  _rounded(x + vw * 0.47 - tl_w, tl_y, tl_w, tl_h, 1.0, _c(*lit, 255, alpha), seg)
  if vw > 30:
    _rounded(x - vw * 0.30, tl_y + tl_h * 0.30, vw * 0.60, max(tl_h * 0.40, 1.0), 1.0, _c(*lit, 230, alpha), seg)


def draw_metric_chip(x: float, top_y: float, text: str, size_ref: float, is_night: bool,
                     primary: bool, danger: bool, warning: bool, min_font: int, max_font: int) -> None:
  font = gui_app.font(FontWeight.BOLD)
  f_size = int(np.clip(size_ref * 0.16, min_font, max_font))
  ts = text_measure.measure_text_cached(font, text, f_size)
  pad_x, pad_y = max(f_size * 0.45, 3), max(f_size * 0.18, 1)
  w, h = ts.x + pad_x * 2, ts.y + pad_y * 2
  rx, ry = x - w / 2, top_y - h - max(f_size * 0.3, 2)
  if danger:
    bg, border, fg = _c(200, 25, 25, 235), _c(255, 80, 80), _c(255, 255, 255)
  elif warning:
    bg, border, fg = _c(205, 115, 15, 235), _c(255, 190, 30), _c(255, 255, 255)
  elif is_night:
    bg = _c(14, 18, 26, 225)
    border = _c(0, 215, 255, 220) if primary else _c(120, 140, 170, 170)
    fg = _c(0, 240, 255, 250) if primary else _c(190, 205, 225, 230)
  else:
    bg = _c(255, 255, 255, 240)
    border = _c(16, 95, 210, 230) if primary else _c(135, 145, 160, 190)
    fg = _c(16, 85, 190) if primary else _c(45, 55, 70, 245)
  rect = rl.Rectangle(rx, ry, w, h)
  rl.draw_rectangle_rounded(rect, 0.45, 6, bg)
  rl.draw_rectangle_rounded_lines_ex(rect, 0.45, 6, 1.2, border)
  rl.draw_text_ex(font, text, rl.Vector2(rx + pad_x, ry + pad_y), f_size, 0, fg)


def draw_ego_halo(x: float, ground_y: float, vw: float, is_night: bool,
                  left_bs: bool, right_bs: bool, alpha: float) -> None:
  """Genesis-style glow ring under the ego car, with red side waves for blind spots."""
  if alpha <= 0.01:
    return
  halo = PALETTE[is_night]['halo'][:3]
  rx, ry = vw * 0.68, vw * 0.16
  rl.draw_ellipse(int(x), int(ground_y), rx * 1.08, ry * 1.25, _c(*halo, 45, alpha))
  for i, a in enumerate((230, 170, 110)):
    rl.draw_ellipse_lines(int(x), int(ground_y), rx - i * 2, ry - i * 1.5, _c(*halo, a, alpha))
  wave_r = vw * 0.22
  for side, active in ((-1, left_bs), (1, right_bs)):
    if not active:
      continue
    cx = x + side * vw * 0.58
    start, end = (150.0, 210.0) if side < 0 else (-30.0, 30.0)
    for k in range(3):
      r0 = wave_r * (0.55 + k * 0.45)
      rl.draw_ring(rl.Vector2(cx, ground_y - vw * 0.10), r0, r0 + max(vw * 0.03, 2.0), start, end, 18,
                   _c(255, 50, 30, 220 - k * 60, alpha))


def draw_steering_wheel(x: float, center_y: float, r_outer: float, steer_deg: float, is_night: bool,
                        left_bs: bool, right_bs: bool, alpha: float) -> None:
  """Upper half of a steering wheel at the bottom edge, rotated with the wheel angle."""
  if alpha <= 0.01:
    return
  r_inner = r_outer * 0.80
  start, end = 180.0 + steer_deg, 360.0 + steer_deg
  c = rl.Vector2(x, center_y)
  if is_night:
    rim, rim_line, hub, hub_line, accent = (32, 38, 48), (85, 100, 122), (24, 30, 38), (60, 72, 90), (0, 220, 255)
  else:
    rim, rim_line, hub, hub_line, accent = (44, 52, 64), (195, 208, 224), (34, 40, 50), (165, 180, 200), (22, 115, 232)
  rl.draw_ring(rl.Vector2(x, center_y + 3), r_inner - 3, r_outer + 5, start, end, 40, _c(0, 0, 0, 110, alpha))
  rl.draw_ring(c, r_inner, r_outer, start, end, 40, _c(*rim, 250, alpha))
  # Soft top highlight gives the rim a rounded, 3D look.
  rl.draw_ring(c, r_outer - (r_outer - r_inner) * 0.35, r_outer - 1, start, end, 40, _c(255, 255, 255, 28, alpha))
  rl.draw_ring_lines(c, r_inner, r_outer, start, end, 40, _c(*rim_line, 240, alpha))
  mark = 270.0 + steer_deg
  rl.draw_ring(c, r_inner - 0.5, r_outer + 0.5, mark - 3.2, mark + 3.2, 12, _c(*accent, 255, alpha))
  hub_r = r_outer * 0.43
  rl.draw_circle_sector(c, hub_r, start, end, 32, _c(*hub, 255, alpha))
  rl.draw_circle_sector_lines(c, hub_r, start, end, 32, _c(*hub_line, 230, alpha))
  ang = math.radians(270.0 + steer_deg)
  ex, ey = x + math.cos(ang) * hub_r * 0.42, center_y + math.sin(ang) * hub_r * 0.42
  rl.draw_circle(int(ex), int(ey), max(r_outer * 0.07, 2.0), _c(*accent, 255, alpha))
  if left_bs:
    rl.draw_ring(c, r_inner - 2, r_outer + 4, start, start + 38.0, 16, _c(255, 120, 0, 230, alpha))
  if right_bs:
    rl.draw_ring(c, r_inner - 2, r_outer + 4, end - 38.0, end, 16, _c(255, 120, 0, 230, alpha))

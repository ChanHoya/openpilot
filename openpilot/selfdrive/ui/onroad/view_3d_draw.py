"""Shared 3D-view drawing for C3/C4: vehicles, lanes, ego halo and wheel."""
import math

import numpy as np
import pyray as rl

from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib import text_measure

LANE_MIN_PROB = 0.30
ADJ_LANE_MIN_PROB = 0.20
LANE_GOOD_PROB = 0.50
ROAD_EDGE_MAX_STD = 2.0


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


# Palettes: white lead/adjacent vehicles, black ego cockpit/vehicle, clear lanes & edges.
PALETTE = {
  True: {   # night
    'path': (0, 190, 255, 120), 'path_edge': (0, 225, 255, 210),
    'lane_good': (0, 230, 118, 255), 'lane_weak': (245, 248, 252, 230), 'lane_block': (255, 145, 0, 255),
    'lane_adj': (190, 205, 225, 170),
    'lead_top': (250, 252, 255, 255), 'lead_bottom': (210, 218, 230, 255), 'lead_edge': (150, 165, 185, 255),
    'lead_two_top': (190, 235, 255, 220), 'lead_two_bottom': (110, 190, 240, 200), 'lead_two_edge': (90, 215, 255, 240),
    'adj_top': (240, 244, 250, 245), 'adj_bottom': (198, 208, 220, 245), 'adj_edge': (145, 160, 180, 240),
    'ego_top': (52, 58, 68, 255), 'ego_bottom': (16, 18, 24, 255), 'ego_edge': (140, 150, 168, 255),
    'glass': (18, 24, 34, 255), 'halo': (0, 200, 255, 255),
  },
  False: {  # day
    'path': (24, 118, 238, 120), 'path_edge': (16, 95, 210, 230),
    'lane_good': (0, 185, 80, 255), 'lane_weak': (255, 255, 255, 255), 'lane_block': (245, 120, 0, 255),
    'lane_adj': (235, 242, 255, 200),
    'lead_top': (252, 254, 255, 255), 'lead_bottom': (214, 222, 234, 255), 'lead_edge': (130, 145, 165, 255),
    'lead_two_top': (180, 225, 250, 220), 'lead_two_bottom': (100, 180, 230, 200), 'lead_two_edge': (70, 190, 240, 240),
    'adj_top': (244, 248, 252, 245), 'adj_bottom': (204, 214, 226, 245), 'adj_edge': (120, 135, 155, 240),
    'ego_top': (48, 54, 64, 255), 'ego_bottom': (16, 18, 24, 255), 'ego_edge': (175, 185, 200, 255),
    'glass': (22, 28, 38, 255), 'halo': (0, 150, 255, 255),
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


def road_edge_color(std: float, is_night: bool, alpha: float = 1.0) -> rl.Color:
  """Road edge colour matching Carrot AR-view: red/magenta when confident, blue/purple when uncertain."""
  temp_f = float(np.clip(std / ROAD_EDGE_MAX_STD, 0.0, 1.0))
  conf = 1.0 - temp_f
  r = int(conf * 255.0)
  g = 20 if is_night else 0
  b = int(temp_f * 255.0)
  a = int(np.clip((0.55 + 0.45 * conf) * 240.0, 90.0, 245.0))
  return _c(r, g, b, a, alpha=alpha)


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


def _draw_procedural_vehicle_rear(x: float, y: float, vw: float, vh: float, is_night: bool,
                                  style: str, braking: bool, alpha: float) -> None:
  """Vector fallback when lead_car.png texture is unavailable."""
  pal = PALETTE[is_night]
  top, bottom, edge = pal[f'{style}_top'], pal[f'{style}_bottom'], pal[f'{style}_edge']
  seg = 12 if vw > 80 else 6
  edge_w = max(1.0, vw * 0.006)

  tw, th = vw * 0.22, vh * 0.36
  tyre_y = y + vh * 0.22
  left_tyre_x = x - vw * 0.49
  right_tyre_x = x + vw * 0.49 - tw
  _rounded(left_tyre_x, tyre_y, tw, th, 0.35, _c(12, 14, 18, 255, alpha), seg)
  _rounded(right_tyre_x, tyre_y, tw, th, 0.35, _c(12, 14, 18, 255, alpha), seg)

  cab_w, cab_h, cab_y = vw * 0.76, vh * 0.48, y - vh * 0.46
  _rounded(x - cab_w / 2, cab_y, cab_w, cab_h * 0.90, 0.60, _mix(top, bottom, 0.12, alpha), seg)
  body_y, body_h = y - vh * 0.16, vh * 0.52
  _rounded(x - vw / 2, body_y, vw, body_h, 0.45, _mix(top, bottom, 0.42, alpha), seg)
  rl.draw_rectangle_rounded_lines_ex(rl.Rectangle(x - vw / 2, body_y, vw, body_h), 0.45, seg, edge_w,
                                     _c(*edge[:3], a=edge[3] * 0.60, alpha=alpha))

  tl_w, tl_h = vw * 0.23, max(vh * 0.038, 1.8)
  lit_color = (100, 225, 255) if style == 'lead_two' else ((255, 30, 30) if braking else (200, 30, 30))
  for tly in (y - vh * 0.05, y + vh * 0.015):
    _rounded(x - vw * 0.46, tly, tl_w, tl_h, 0.8, _c(*lit_color, 255, alpha), seg)
    _rounded(x + vw * 0.46 - tl_w, tly, tl_w, tl_h, 0.8, _c(*lit_color, 255, alpha), seg)


def draw_vehicle_rear(x: float, y: float, vw: float, vh: float, is_night: bool,
                      style: str = 'lead', braking: bool = False, alpha: float = 1.0,
                      glow: bool = True) -> None:
  """Render realistic lead vehicle rear PNG (images/lead_car.png) with optional contact glow & brake highlight."""
  if vw < 2 or alpha <= 0.01:
    return

  tex = None
  try:
    tex = gui_app.texture('images/lead_car.png')
  except Exception:
    tex = None

  tex_w = getattr(tex, 'width', 0) if tex is not None else 0
  tex_h = getattr(tex, 'height', 0) if tex is not None else 0
  has_tex = isinstance(tex_w, (int, float)) and tex_w > 0 and isinstance(tex_h, (int, float)) and tex_h > 0

  draw_w = float(vw)
  draw_h = float(vw * (float(tex_h) / float(tex_w))) if has_tex else float(vh)

  # 1. Contact ground aura/shadow beneath tires
  glow_y = y + draw_h * 0.46
  if glow:
    glow_layers = 16 if draw_w >= 30 else 6
    for i in range(glow_layers):
      t = i / float(glow_layers - 1)
      rx = draw_w * (0.84 - 0.54 * t)
      ry = max(draw_h * (0.23 - 0.165 * t), 1.0)
      if t <= 0.55:
        k = t / 0.55
        r_ch, g_ch, b_ch = 0, int(140 + 85 * k), 255
      else:
        k = (t - 0.55) / 0.45
        r_ch, g_ch, b_ch = int(165 * k), int(225 + 30 * k), 255
      # Gentle outer edge fade + quadratic rise toward core (keeps stacked alpha natural)
      layer_a = (8.0 + 34.0 * (t ** 1.6)) * (16.0 / glow_layers)
      rl.draw_ellipse(int(x), int(glow_y), rx, ry, _c(r_ch, g_ch, b_ch, layer_a, alpha))
  else:
    sh_layers = 5 if draw_w >= 30 else 3
    sh_max = 26 if is_night else 18
    for i in range(sh_layers):
      t = i / float(sh_layers - 1)
      rx = draw_w * (0.58 - 0.20 * t)
      ry = max(draw_h * (0.14 - 0.07 * t), 1.0)
      rl.draw_ellipse(int(x), int(glow_y), rx, ry, _c(0, 0, 0, (4 + sh_max * t) * (5.0 / sh_layers), alpha))

  if has_tex:
    if style == 'lead_two':
      tint = _c(150, 230, 255, int(215 * alpha))
    elif style == 'adj':
      tint = _c(190, 205, 220, int(195 * alpha))
    elif is_night:
      tint = _c(230, 236, 245, int(255 * alpha))
    else:
      tint = _c(255, 255, 255, int(255 * alpha))

    src = rl.Rectangle(0.0, 0.0, float(tex_w), float(tex_h))
    dst = rl.Rectangle(x - draw_w * 0.5, y - draw_h * 0.5, draw_w, draw_h)
    rl.draw_texture_pro(tex, src, dst, rl.Vector2(0.0, 0.0), 0.0, tint)

    # Extra brake bloom aligned to CHMSL & rear lightbar when braking/stopped
    if braking and style != 'adj' and draw_w > 24:
      chmsl_y = y - draw_h * 0.402
      bar_y = y - draw_h * 0.062
      rl.draw_ellipse(int(x), int(chmsl_y), draw_w * 0.14, max(draw_h * 0.028, 1.5), _c(255, 40, 30, 130, alpha))
      for side in (-0.34, 0.34):
        rl.draw_ellipse(int(x + draw_w * side), int(bar_y), draw_w * 0.15, max(draw_h * 0.048, 2.0), _c(255, 35, 25, 115, alpha))
      rl.draw_ellipse(int(x), int(bar_y), draw_w * 0.44, max(draw_h * 0.025, 1.5), _c(255, 45, 30, 85, alpha))
  else:
    _draw_procedural_vehicle_rear(x, y, draw_w, draw_h, is_night, style, braking, alpha)


draw_sleek_sedan_rear = draw_vehicle_rear


def draw_blindspot_radar_wave(center_x: float, center_y: float, side_sign: float,
                              color: rl.Color = None, scale: float = 1.0,
                              alpha: float = 1.0, show_triangle: bool = True) -> None:
  """Draws OEM-style 3-layer wifi/radar concentric arc waves and alert triangle for blindspot warning."""
  if alpha <= 0.01:
    return
  base_color = color if color is not None else rl.Color(245, 75, 20, 240)
  cr, cg, cb = int(base_color.r), int(base_color.g), int(base_color.b)

  # Side orientation: side_sign < 0 (Left), side_sign > 0 (Right)
  # Perspective compression along Y axis to lie flat on the road surface
  y_scale = 0.58

  # Concentric arc layer radii and thicknesses
  arc_layers = (
    (42.0 * scale, max(7.0 * scale, 2.0), 220),   # Inner arc
    (72.0 * scale, max(9.0 * scale, 2.5), 240),   # Middle arc
    (106.0 * scale, max(11.0 * scale, 3.0), 255), # Outer arc
  )

  # Angular sector for radar wave (matching Hyundai/Genesis HUD radar propagation)
  # Right: 320 to 385 degrees (pointing outward/rear-right)
  # Left:  155 to 220 degrees (pointing outward/rear-left)
  if side_sign >= 0:
    deg_start, deg_end = -42.0, 26.0
  else:
    deg_start, deg_end = 154.0, 222.0

  steps = 22
  for radius, thickness, arc_alpha in arc_layers:
    cur_alpha = int(arc_alpha * alpha)
    c_fill = rl.Color(cr, cg, cb, cur_alpha)
    c_edge = rl.Color(min(255, cr + 25), min(255, cg + 30), min(255, cb + 20), int(255 * alpha))

    # Construct perspective elliptic arc polygon strip
    r_in = radius - thickness / 2.0
    r_out = radius + thickness / 2.0
    pts_outer = []
    pts_inner = []

    for s in range(steps + 1):
      t = s / float(steps)
      deg = deg_start + (deg_end - deg_start) * t
      rad = math.radians(deg)
      cos_a = math.cos(rad)
      sin_a = math.sin(rad) * y_scale  # flat on ground

      pts_outer.append(rl.Vector2(center_x + r_out * cos_a, center_y + r_out * sin_a))
      pts_inner.append(rl.Vector2(center_x + r_in * cos_a, center_y + r_in * sin_a))

    # Draw continuous quad strip between inner and outer arcs
    for i in range(steps):
      p1 = pts_inner[i]
      p2 = pts_outer[i]
      p3 = pts_outer[i + 1]
      p4 = pts_inner[i + 1]
      rl.draw_triangle(p1, p2, p3, c_fill)
      rl.draw_triangle(p1, p3, p4, c_fill)

    # Rounded end caps for soft radar wave feel
    p_start_c = rl.Vector2((pts_inner[0].x + pts_outer[0].x) / 2.0, (pts_inner[0].y + pts_outer[0].y) / 2.0)
    p_end_c = rl.Vector2((pts_inner[-1].x + pts_outer[-1].x) / 2.0, (pts_inner[-1].y + pts_outer[-1].y) / 2.0)
    rl.draw_circle(int(p_start_c.x), int(p_start_c.y), thickness * 0.48 * y_scale, c_fill)
    rl.draw_circle(int(p_end_c.x), int(p_end_c.y), thickness * 0.48 * y_scale, c_fill)

  # Triangle warning sign (▲ with exclamation !) placed right next to the outermost wave
  if show_triangle:
    tri_dist = 136.0 * scale
    mid_deg = (deg_start + deg_end) / 2.0
    mid_rad = math.radians(mid_deg)
    tri_x = center_x + tri_dist * math.cos(mid_rad)
    tri_y = center_y + tri_dist * math.sin(mid_rad) * y_scale

    tw = 40.0 * scale
    th = 36.0 * scale
    p_top = rl.Vector2(tri_x, tri_y - th * 0.58)
    p_left = rl.Vector2(tri_x - tw * 0.50, tri_y + th * 0.42)
    p_right = rl.Vector2(tri_x + tw * 0.50, tri_y + th * 0.42)

    # Semi-transparent orange-red warning triangle body
    rl.draw_triangle(p_top, p_left, p_right, _c(cr, cg, cb, 190, alpha))
    # Crisp outer alert stroke
    rl.draw_triangle_lines(p_top, p_left, p_right, _c(255, 140, 30, 255, alpha))

    # Bold exclamation mark (!) in the center of the warning triangle
    excl_color = _c(255, 245, 220, 255, alpha)
    bar_w = max(3.0 * scale, 1.5)
    bar_h = th * 0.38
    bar_y = tri_y - th * 0.18
    rl.draw_rectangle_rounded(rl.Rectangle(tri_x - bar_w / 2.0, bar_y, bar_w, bar_h), 0.5, 4, excl_color)
    rl.draw_circle(int(tri_x), int(tri_y + th * 0.28), max(2.5 * scale, 1.2), excl_color)


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


def draw_windshield_frame(rect, steer_x: float, is_night: bool, left_bs: bool, right_bs: bool,
                          alpha: float = 1.0) -> None:
  """Subtle front-windshield cowl, black ego hood edge, and left-side driver binnacle."""
  if alpha <= 0.01:
    return
  x0, y0, w, h = float(rect.x), float(rect.y), float(rect.width), float(rect.height)
  pal = PALETTE[is_night]

  # Faint black ego hood crest peeking at the lower edge of the windshield.
  rl.draw_ellipse(int(x0 + w * 0.50), int(y0 + h * 1.03), w * 0.36, h * 0.115, _c(*pal['ego_bottom'][:3], 185, alpha))
  rl.draw_ellipse_lines(int(x0 + w * 0.50), int(y0 + h * 1.03), w * 0.36, h * 0.115, _c(*pal['ego_top'][:3], 170, alpha))

  # Lower dashboard cowl shadow gradient across the bottom of the screen.
  cowl_top = y0 + h * 0.87
  rl.draw_rectangle_gradient_v(
    int(x0), int(cowl_top), int(w), int(max(1.0, (y0 + h) - cowl_top)),
    _c(12, 15, 20, 0, alpha), _c(10, 12, 16, 215, alpha),
  )

  # Subtle windshield base contour line separation between outside road and cockpit.
  cowl_line_y = y0 + h * 0.945
  line_col = _c(120, 138, 165, 95, alpha) if is_night else _c(75, 88, 108, 105, alpha)
  rl.draw_line_ex(rl.Vector2(x0, cowl_line_y), rl.Vector2(x0 + w, cowl_line_y), max(1.0, h * 0.0025), line_col)

  # Driver instrument binnacle arch behind the left-mounted steering wheel.
  hood_rx = w * 0.125
  hood_ry = h * 0.125
  binnacle_y = y0 + h * 1.005
  rl.draw_ellipse(int(steer_x), int(binnacle_y), hood_rx, hood_ry, _c(16, 20, 26, 225, alpha))
  rl.draw_ellipse_lines(int(steer_x), int(binnacle_y), hood_rx, hood_ry,
                        _c(*(95, 112, 138) if is_night else (145, 160, 180), 175, alpha))

  # Thin translucent windshield A-pillar edges on far left/right corners.
  pillar_w = max(w * 0.016, 4.0)
  pillar_col = _c(12, 14, 18, 95, alpha)
  rl.draw_rectangle(int(x0), int(y0), int(pillar_w), int(h), pillar_col)
  rl.draw_rectangle(int(x0 + w - pillar_w), int(y0), int(pillar_w), int(h), pillar_col)
  if left_bs:
    rl.draw_rectangle(int(x0), int(y0 + h * 0.25), int(pillar_w * 1.3), int(h * 0.50), _c(255, 110, 0, 130, alpha))
  if right_bs:
    rl.draw_rectangle(int(x0 + w - pillar_w * 1.3), int(y0 + h * 0.25), int(pillar_w * 1.3), int(h * 0.50),
                      _c(255, 110, 0, 130, alpha))


def draw_speed_bump_marker(x: float, y: float, dist_m: float, is_night: bool, alpha: float = 1.0) -> None:
  """3D road marker badge hovering directly over a speed bump location."""
  if alpha <= 0.02:
    return
  font = gui_app.font(FontWeight.BOLD)
  f_size = 14
  text = f"방지턱 {int(round(dist_m))}m"
  ts = text_measure.measure_text_cached(font, text, f_size)
  pad_x, pad_y = 7.0, 3.0
  icon_sz = 16.0
  total_w = icon_sz + 4.0 + ts.x + pad_x * 2
  total_h = max(icon_sz, ts.y) + pad_y * 2
  rx = x - total_w / 2
  ry = y - total_h / 2

  # Ground pin line connecting marker to road surface
  pin_col = _c(255, 185, 20, int(180 * alpha))
  rl.draw_line_ex(rl.Vector2(x, y + total_h / 2), rl.Vector2(x, y + total_h / 2 + 10), 1.8, pin_col)

  # Badge background
  bg_col = _c(20, 24, 32, int(225 * alpha))
  border_col = _c(255, 195, 30, int(230 * alpha))
  rl.draw_rectangle_rounded(rl.Rectangle(rx, ry, total_w, total_h), 0.45, 6, bg_col)
  rl.draw_rectangle_rounded_lines_ex(rl.Rectangle(rx, ry, total_w, total_h), 0.45, 6, 1.4, border_col)

  # Speed bump curved hump symbol
  ic_cx = rx + pad_x + icon_sz / 2
  ic_cy = y
  rl.draw_circle(int(ic_cx), int(ic_cy), icon_sz * 0.46, _c(255, 185, 20, int(240 * alpha)))
  rl.draw_ellipse(int(ic_cx), int(ic_cy + 1), icon_sz * 0.36, icon_sz * 0.22, _c(20, 24, 32, int(255 * alpha)))

  # Distance text
  rl.draw_text_ex(font, text, rl.Vector2(rx + pad_x + icon_sz + 4.0, ry + pad_y), f_size, 0, _c(255, 215, 60, int(250 * alpha)))


def draw_traffic_stop_marker(x: float, y: float, dist_m: float, is_night: bool, alpha: float = 1.0) -> None:
  """3D traffic light symbol hovering directly over the road stop line location."""
  if alpha <= 0.02:
    return
  font = gui_app.font(FontWeight.BOLD)
  f_size = 14
  text = f"정지선 {int(round(dist_m))}m"
  ts = text_measure.measure_text_cached(font, text, f_size)
  pad_x, pad_y = 7.0, 3.0
  icon_w = 26.0
  total_w = icon_w + 5.0 + ts.x + pad_x * 2
  total_h = 24.0
  rx = x - total_w / 2
  ry = y - total_h / 2

  # Ground pin line connecting signal badge to stop line
  pin_col = _c(255, 60, 60, int(180 * alpha))
  rl.draw_line_ex(rl.Vector2(x, y + total_h / 2), rl.Vector2(x, y + total_h / 2 + 10), 1.8, pin_col)

  # Badge background
  bg_col = _c(20, 24, 32, int(230 * alpha))
  border_col = _c(255, 70, 70, int(220 * alpha))
  rl.draw_rectangle_rounded(rl.Rectangle(rx, ry, total_w, total_h), 0.45, 6, bg_col)
  rl.draw_rectangle_rounded_lines_ex(rl.Rectangle(rx, ry, total_w, total_h), 0.45, 6, 1.4, border_col)

  # Horizontal mini traffic light body
  tl_box_x = rx + pad_x
  tl_box_y = ry + 4.0
  tl_box_h = total_h - 8.0
  rl.draw_rectangle_rounded(rl.Rectangle(tl_box_x, tl_box_y, icon_w, tl_box_h), 0.5, 4, _c(10, 12, 16, int(255 * alpha)))
  rl.draw_rectangle_rounded_lines_ex(rl.Rectangle(tl_box_x, tl_box_y, icon_w, tl_box_h), 0.5, 4, 1.0, _c(80, 90, 105, int(180 * alpha)))

  # Red light active glow
  red_x = tl_box_x + 6.0
  mid_y = ry + total_h / 2
  rl.draw_circle(int(red_x), int(mid_y), 5.5, _c(255, 30, 30, int(100 * alpha)))
  rl.draw_circle(int(red_x), int(mid_y), 3.8, _c(255, 45, 45, int(255 * alpha)))
  rl.draw_circle(int(red_x - 1), int(mid_y - 1), 1.2, _c(255, 200, 200, int(220 * alpha)))

  # Off amber & green lights
  rl.draw_circle(int(tl_box_x + 13.0), int(mid_y), 3.2, _c(60, 50, 15, int(200 * alpha)))
  rl.draw_circle(int(tl_box_x + 20.0), int(mid_y), 3.2, _c(15, 60, 25, int(200 * alpha)))

  # Text
  rl.draw_text_ex(font, text, rl.Vector2(rx + pad_x + icon_w + 5.0, ry + pad_y + 1.0), f_size, 0, _c(255, 110, 110, int(250 * alpha)))


def draw_road_speed_bump_texture(cx: float, cy: float, width_px: float, dist_m: float, is_night: bool, alpha: float = 1.0) -> None:
  """Render realistic road speed bump image (images/speed_bump_road.png) on the lower road surface."""
  if alpha <= 0.02 or width_px < 10.0:
    return

  tex = None
  try:
    tex = gui_app.texture('images/speed_bump_road.png')
  except Exception:
    tex = None

  tex_w = getattr(tex, 'width', 0) if tex is not None else 0
  tex_h = getattr(tex, 'height', 0) if tex is not None else 0
  has_tex = isinstance(tex_w, (int, float)) and tex_w > 0 and isinstance(tex_h, (int, float)) and tex_h > 0

  draw_w = float(max(52.0, width_px))
  aspect = (float(tex_h) / float(tex_w)) if has_tex else (85.0 / 584.0)
  draw_h = float(max(12.0, draw_w * aspect * 1.15))

  # Subtle contact shadow on the road
  rl.draw_ellipse(int(cx), int(cy + draw_h * 0.25), draw_w * 0.52, max(3.0, draw_h * 0.35),
                  _c(0, 0, 0, int(115 * alpha)))

  rx = cx - draw_w / 2.0
  ry = cy - draw_h / 2.0
  if has_tex:
    rl.draw_texture_pro(
      tex,
      rl.Rectangle(0.0, 0.0, float(tex_w), float(tex_h)),
      rl.Rectangle(rx, ry, draw_w, draw_h),
      rl.Vector2(0.0, 0.0),
      0.0,
      _c(255, 255, 255, int(245 * alpha)),
    )
  else:
    _rounded(rx, ry, draw_w, draw_h, 0.35, _c(28, 32, 40, int(235 * alpha)), 6)
    stripe_w = draw_w / 4.0
    for idx in (1, 3):
      _rounded(rx + idx * stripe_w, ry + draw_h * 0.12, stripe_w * 0.85, draw_h * 0.76, 0.2,
               _c(255, 195, 25, int(235 * alpha)), 4)

  # Compact distance pill below the road speed bump
  font = gui_app.font(FontWeight.BOLD)
  f_size = 18 if draw_w > 160.0 else 15
  text = f"방지턱 {int(round(dist_m))}m"
  ts = text_measure.measure_text_cached(font, text, f_size)
  pad_x, pad_y = 8.0, 3.0
  pw = ts.x + pad_x * 2.0
  ph = ts.y + pad_y * 2.0
  px = cx - pw / 2.0
  py = ry + draw_h + 6.0
  _rounded(px, py, pw, ph, 0.45, _c(16, 20, 28, int(215 * alpha)), 6)
  rl.draw_rectangle_rounded_lines_ex(rl.Rectangle(px, py, pw, ph), 0.45, 6, 1.5, _c(255, 195, 30, int(230 * alpha)))
  rl.draw_text_ex(font, text, rl.Vector2(px + pad_x, py + pad_y), f_size, 0, _c(255, 220, 75, int(255 * alpha)))


_TRAFFIC_LIGHT_4LAMP_ACTIVE = {
  'red': {'red'},
  'yellow': {'yellow'},
  'left': {'left'},
  'red_left': {'red', 'left'},
  'green': {'green'},
  'green_left': {'left', 'green'},
}

_TRAFFIC_LIGHT_4LAMP_SPEC = (
  ('red', 0.1506, (255, 45, 45)),
  ('yellow', 0.3847, (255, 205, 25)),
  ('left', 0.6188, (40, 235, 90)),
  ('green', 0.8477, (40, 235, 90)),
)


def draw_overhead_traffic_light_4lamp(cx: float, cy: float, width_px: float, lamp_state: str,
                                      dist_m: float = 0.0, remain_s: int = 0,
                                      is_night: bool = True, alpha: float = 1.0) -> None:
  """Render overhead Korean 4-lamp traffic light image (images/traffic_light_4lamp.png) with active lamp glow & distance."""
  if alpha <= 0.02 or width_px < 20.0:
    return

  tex = None
  try:
    tex = gui_app.texture('images/traffic_light_4lamp.png')
  except Exception:
    tex = None

  tex_w = getattr(tex, 'width', 0) if tex is not None else 0
  tex_h = getattr(tex, 'height', 0) if tex is not None else 0
  has_tex = isinstance(tex_w, (int, float)) and tex_w > 0 and isinstance(tex_h, (int, float)) and tex_h > 0

  tl_w = float(max(96.0, width_px))
  aspect = (float(tex_h) / float(tex_w)) if has_tex else (174.0 / 581.0)
  tl_h = float(tl_w * aspect)
  rx = cx - tl_w / 2.0
  ry = cy - tl_h / 2.0

  # Soft drop shadow behind housing
  _rounded(rx - 3.0, ry + 2.0, tl_w + 6.0, tl_h + 6.0, 0.35, _c(0, 0, 0, int(105 * alpha)), 8)

  if has_tex:
    rl.draw_texture_pro(
      tex,
      rl.Rectangle(0.0, 0.0, float(tex_w), float(tex_h)),
      rl.Rectangle(rx, ry, tl_w, tl_h),
      rl.Vector2(0.0, 0.0),
      0.0,
      _c(255, 255, 255, int(250 * alpha)),
    )
  else:
    _rounded(rx, ry, tl_w, tl_h, 0.35, _c(18, 22, 28, int(245 * alpha)), 8)
    rl.draw_rectangle_rounded_lines_ex(rl.Rectangle(rx, ry, tl_w, tl_h), 0.35, 8, 2.0, _c(90, 100, 115, int(220 * alpha)))

  active_set = _TRAFFIC_LIGHT_4LAMP_ACTIVE.get(str(lamp_state), set())
  lamp_cy = ry + tl_h * 0.5029
  lamp_r = tl_h * 0.295

  for name, u, rgb in _TRAFFIC_LIGHT_4LAMP_SPEC:
    lx = rx + tl_w * u
    if name not in active_set:
      rl.draw_circle(int(lx), int(lamp_cy), lamp_r * 1.02, _c(12, 16, 22, int(195 * alpha)))
    else:
      rl.draw_circle(int(lx), int(lamp_cy), lamp_r * 1.45, _c(*rgb, int(70 * alpha)))
      rl.draw_circle(int(lx), int(lamp_cy), lamp_r * 1.15, _c(*rgb, int(55 * alpha)))
      rl.draw_circle_lines(int(lx), int(lamp_cy), max(2.0, lamp_r * 1.02), _c(255, 255, 255, int(220 * alpha)))

  if dist_m > 0.0 or remain_s > 0:
    parts = []
    if dist_m > 0.0:
      parts.append(f"{int(round(dist_m))}m")
    if remain_s > 0:
      parts.append(f"{int(remain_s)}s")
    label = " · ".join(parts)

    if 'red' in active_set:
      border_rgb = (255, 75, 75)
      text_rgb = (255, 130, 130)
    elif 'yellow' in active_set:
      border_rgb = (255, 200, 40)
      text_rgb = (255, 225, 110)
    elif 'green' in active_set or 'left' in active_set:
      border_rgb = (55, 230, 105)
      text_rgb = (120, 250, 155)
    else:
      border_rgb = (180, 190, 205)
      text_rgb = (230, 235, 240)

    font = gui_app.font(FontWeight.BOLD)
    f_size = 18
    ts = text_measure.measure_text_cached(font, label, f_size)
    pad_x, pad_y = 9.0, 3.0
    pw = ts.x + pad_x * 2.0
    ph = ts.y + pad_y * 2.0
    px = cx - pw / 2.0
    pill_cy = ry + tl_h + 18.0
    py = pill_cy - ph / 2.0
    _rounded(px, py, pw, ph, 0.45, _c(14, 18, 26, int(225 * alpha)), 6)
    rl.draw_rectangle_rounded_lines_ex(rl.Rectangle(px, py, pw, ph), 0.45, 6, 1.5, _c(*border_rgb, int(225 * alpha)))
    rl.draw_text_ex(font, label, rl.Vector2(px + pad_x, py + pad_y), f_size, 0, _c(*text_rgb, int(255 * alpha)))


def resolve_speed_camera_state(sm, v_ego: float, now_t: float, state: dict) -> tuple[int, float, bool]:
  """Resolve active speed camera (limit_kph, calibrated_dist_m, is_camera_zone) from CarrotMan or Hyundai Navi carState."""
  valid = getattr(sm, 'valid', {})
  cm = sm['carrotMan'] if valid.get('carrotMan', False) else None
  cs = sm['carState'] if valid.get('carState', False) else None

  cand_limit = 0
  cand_dist = 0.0
  cand_src = ''
  cand_active = False

  if cm is not None:
    spd_type = int(float(getattr(cm, 'xSpdType', 0) or 0))
    cm_limit = int(float(getattr(cm, 'xSpdLimit', 0) or 0))
    cm_dist = float(getattr(cm, 'xSpdDist', 0.0) or 0.0)
    active_carrot = int(float(getattr(cm, 'activeCarrot', 0) or 0))
    if spd_type != 22 and cm_limit > 0:
      if 0.5 <= cm_dist <= 600.0:
        cand_limit, cand_dist, cand_src, cand_active = cm_limit, cm_dist, 'cm', True
      elif cm_dist <= 0.0 and active_carrot > 0:
        cand_limit, cand_dist, cand_src, cand_active = cm_limit, 0.0, 'cm_sec', True

  if not cand_active and cs is not None:
    cs_limit = int(float(getattr(cs, 'speedLimit', 0) or 0))
    cs_dist = float(getattr(cs, 'speedLimitDistance', 0.0) or 0.0)
    if cs_limit > 0 and 0.5 <= cs_dist <= 600.0:
      cand_limit, cand_dist, cand_src, cand_active = cs_limit, cs_dist, 'cs', True

  if not cand_active:
    state['raw_dist'] = 0.0
    state['est_dist'] = 0.0
    state['last_t'] = 0.0
    state['src'] = ''
    return 0, 0.0, False

  if cand_dist <= 0.0:
    state['raw_dist'] = 0.0
    state['est_dist'] = 0.0
    state['last_t'] = now_t
    state['src'] = cand_src
    return cand_limit, 0.0, True

  prev_raw = float(state.get('raw_dist', 0.0) or 0.0)
  prev_t = float(state.get('last_t', 0.0) or 0.0)
  if state.get('src') != cand_src or abs(cand_dist - prev_raw) > 0.5 or prev_t <= 0.0:
    state['raw_dist'] = cand_dist
    state['est_dist'] = cand_dist
    state['last_t'] = now_t
    state['src'] = cand_src
  else:
    dt = float(np.clip(now_t - prev_t, 0.0, 0.25))
    state['est_dist'] = max(0.5, float(state.get('est_dist', cand_dist)) - max(0.0, v_ego) * dt)
    state['last_t'] = now_t
  return cand_limit, float(state['est_dist']), True


def draw_speed_camera_sign(cx: float, cy: float, sign_r: float, limit_kph: int,
                           cam_dist: float, overspeed: bool, is_night: bool,
                           is_compact: bool = False, alpha: float = 1.0) -> None:
  """Render Korean circular speed-limit camera sign with approaching distance pill."""
  if limit_kph <= 0 or sign_r <= 2.0 or alpha <= 0.02:
    return

  rl.draw_circle(int(round(cx)), int(round(cy + 3.0)), sign_r + 3.0, _c(0, 0, 0, int((95 if is_night else 55) * alpha)))
  rl.draw_circle(int(round(cx)), int(round(cy)), sign_r, _c(250, 252, 255, int(252 * alpha)))
  ring_thick = max(3.5, sign_r * 0.17)
  rl.draw_ring(
    rl.Vector2(float(cx), float(cy)),
    max(2.0, sign_r - ring_thick),
    float(sign_r),
    0.0,
    360.0,
    36,
    _c(225, 28, 36, int(250 * alpha)),
  )

  font = gui_app.font(FontWeight.BOLD)
  num_text = str(limit_kph)
  num_size = max(14, int(round(sign_r * (0.95 if len(num_text) <= 2 else 0.78))))
  nsz = text_measure.measure_text_cached(font, num_text, num_size)
  rl.draw_text_ex(
    font,
    num_text,
    rl.Vector2(float(cx) - nsz.x * 0.5, float(cy) - nsz.y * 0.52),
    num_size,
    0,
    _c(20, 24, 30, int(255 * alpha)),
  )

  if cam_dist > 0.0:
    dist_text = f"{int(round(cam_dist))} m"
    d_size = 15 if is_compact else 20
    dsz = text_measure.measure_text_cached(font, dist_text, d_size)
    pad_x = 8.0 if is_compact else 12.0
    pad_y = 3.0 if is_compact else 5.0
    pw = dsz.x + 2.0 * pad_x
    ph = dsz.y + 2.0 * pad_y
    px = float(cx) - pw * 0.5
    py = float(cy) + float(sign_r) + (5.0 if is_compact else 7.0)
    pill_bg = _c(220, 38, 38, int(240 * alpha)) if overspeed else _c(22, 28, 38, int(210 * alpha))
    _rounded(px, py, pw, ph, 0.90, pill_bg, 8)
    rl.draw_text_ex(font, dist_text, rl.Vector2(px + pad_x, py + pad_y), d_size, 0, _c(255, 255, 255, int(255 * alpha)))

import colorsys
import time
import numpy as np
import pyray as rl
from openpilot.cereal import messaging, car
from dataclasses import dataclass, field
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.selfdrive.locationd.calibrationd import HEIGHT_INIT
from openpilot.selfdrive.ui.ui_state import ui_state, UIStatus
from openpilot.selfdrive.ui.vision_status import blindspot_source_packet, blindspot_sources
from openpilot.selfdrive.ui.road_markings import lane_dash_segments, project_blindspot_barrier, blindspot_barrier_quads
from openpilot.selfdrive.ui.mici.onroad import blend_colors
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.text_draw import draw_text_ui_style
from openpilot.system.ui.lib.shader_polygon import draw_polygon, draw_polygon_solid, Gradient
from openpilot.system.ui.lib import native_draw, native_geometry
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.lib.overlay_cache import OverlayLayer, DisplayClock
from typing import Optional, Any

CLIP_MARGIN = 500
MIN_DRAW_DISTANCE = 10.0
MAX_DRAW_DISTANCE = 100.0

THROTTLE_COLORS = [
  rl.Color(13, 248, 122, 102),   # HSLF(148/360, 0.94, 0.51, 0.4)
  rl.Color(114, 255, 92, 89),    # HSLF(112/360, 1.0, 0.68, 0.35)
  rl.Color(114, 255, 92, 0),     # HSLF(112/360, 1.0, 0.68, 0.0)
]

NO_THROTTLE_COLORS = [
  rl.Color(242, 242, 242, 102), # HSLF(148/360, 0.0, 0.95, 0.4)
  rl.Color(242, 242, 242, 89),  # HSLF(112/360, 0.0, 0.95, 0.35)
  rl.Color(242, 242, 242, 0),   # HSLF(112/360, 0.0, 0.95, 0.0)
]

LANE_LINE_COLORS = {
  UIStatus.DISENGAGED: rl.Color(200, 200, 200, 255),
  UIStatus.OVERRIDE: rl.Color(255, 255, 255, 255),
  UIStatus.ENGAGED: rl.Color(0, 255, 64, 255),
}


@dataclass
class ModelPoints:
  raw_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
  projected_points: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))


@dataclass
class LeadVehicle:
  glow: list[float] = field(default_factory=list)
  chevron: list[float] = field(default_factory=list)
  fill_alpha: int = 0
  rect: list[tuple[float, float]] = field(default_factory=list)   # 4 corners (screen space)
  color: Optional[Any] = None

@dataclass
class RadarInfoItem:
  x: float = 0.0
  y: float = 0.0
  w: float = 0.0
  h: float = 0.0
  text: str = ""
  color: Optional[Any] = None
  is_star: bool = False
  
class ModelRenderer(Widget):
  def __init__(self):
    super().__init__()
    self._overlay = OverlayLayer(0)
    self._overlay_state = None
    self._display_clock = DisplayClock()
    self._model_frame = -1
    self._radar_frame = -1
    self._params_frame = -1
    self._longitudinal_control = False
    self._experimental_mode = False
    self._blend_filter = FirstOrderFilter(1.0, 0.25, 1 / gui_app.target_fps)
    self._prev_allow_throttle = True
    self._lane_line_probs = np.zeros(4, dtype=np.float32)
    self._road_edge_stds = np.zeros(2, dtype=np.float32)
    self._lead_vehicles = [LeadVehicle(), LeadVehicle()]
    self._path_offset_z = HEIGHT_INIT[0]

    # Initialize ModelPoints objects
    self._path = ModelPoints()
    self._lane_lines = [ModelPoints() for _ in range(4)]
    self._lane_marking_codes = [-1] * 4
    self._lane_marking_segments: list[list[np.ndarray]] = [[] for _ in range(4)]
    self._road_edges = [ModelPoints() for _ in range(2)]
    self._acceleration_x = np.empty((0,), dtype=np.float32)

    self._acceleration_x_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)
    self._acceleration_x_filter2 = FirstOrderFilter(0.0, 1, 1 / gui_app.target_fps)

    self._fsd_mode: bool = False
    self._fsd_is_night: bool = True

    self._torque_filter = FirstOrderFilter(0, 0.1, 1 / gui_app.target_fps)
    self._ll_color_filter = FirstOrderFilter(0.0, 0.1, 1 / gui_app.target_fps)

    # Transform matrix (3x3 for car space to screen space)
    self._car_space_transform = np.zeros((3, 3), dtype=np.float32)
    self._transform_dirty = True
    self._clip_region = None
    
    self._lead_pt_filt = [None, None]
    self._radar_info_items: list[RadarInfoItem] = []
    self._font_display: rl.Font = gui_app.font(FontWeight.DISPLAY)
    self._exp_gradient = Gradient(
      start=(0.0, 1.0),  # Bottom of path
      end=(0.0, 0.0),  # Top of path
      colors=[],
      stops=[],
    )

    # Get longitudinal control setting from car parameters
    if car_params := Params().get("CarParams"):
      cp = messaging.log_from_bytes(car_params, car.CarParams)
      self._longitudinal_control = cp.openpilotLongitudinalControl

  def set_transform(self, transform: np.ndarray):
    self._car_space_transform = transform.astype(np.float32)
    self._transform_dirty = True

  def set_fsd_mode(self, enabled: bool, is_night: bool = True) -> None:
    self._fsd_mode = enabled
    self._fsd_is_night = is_night

  def _render(self, rect: rl.Rectangle):
    state = self._overlay_state
    self._overlay.draw(rect, self._paint, state.key if state else None,
                       enabled=state is not None and state.enabled, urgent=state is not None and state.urgent)

  def _paint(self, rect: rl.Rectangle):
    self._display_clock.tick()
    sm = ui_state.sm
    self._display_clock.update_filter(self._torque_filter, -sm['carOutput'].actuatorsOutput.torque)

    # Check if data is up-to-date
    if (sm.recv_frame["liveCalibration"] < ui_state.started_frame or
        sm.recv_frame["modelV2"] < ui_state.started_frame):
      return

    # Set up clipping region
    self._clip_region = rl.Rectangle(
      rect.x - CLIP_MARGIN, rect.y - CLIP_MARGIN, rect.width + 2 * CLIP_MARGIN, rect.height + 2 * CLIP_MARGIN
    )

    # Update state
    self._experimental_mode = sm['selfdriveState'].experimentalMode

    live_calib = sm['liveCalibration']
    self._path_offset_z = live_calib.height[0] if live_calib.height else HEIGHT_INIT[0]

    if sm.recv_frame['carParams'] != self._params_frame:
      self._params_frame = sm.recv_frame['carParams']
      self._longitudinal_control = sm['carParams'].openpilotLongitudinalControl

    model = sm['modelV2']
    radar_state = sm['radarState'] if sm.valid['radarState'] else None
    lead_one = radar_state.leadOne if radar_state else None
    render_lead_indicator = self._longitudinal_control and radar_state is not None

    # Update model data when needed
    model_updated = sm.recv_frame['modelV2'] != self._model_frame
    if model_updated or sm.recv_frame['radarState'] != self._radar_frame or self._transform_dirty:
      self._radar_frame = sm.recv_frame['radarState']
      if model_updated:
        self._model_frame = sm.recv_frame['modelV2']
        self._update_raw_points(model)

      path_x_array = self._path.raw_points[:, 0]
      if path_x_array.size == 0:
        return

      car_state = sm['carState'] if sm.valid['carState'] and sm.alive['carState'] else None
      self._update_model(lead_one, path_x_array, car_state)
      if render_lead_indicator:
        self._update_leads_carrot(radar_state, path_x_array)
        
      if ui_state.show_radar_info > 0 and radar_state is not None:
        self._update_radar_info(radar_state, path_x_array)
      else:
        self._radar_info_items = []
        
      self._transform_dirty = False

    if getattr(self, '_fsd_mode', False):
      self._draw_fsd_visualization(sm)
    else:
      # Draw elements (hide when disengaged)
      self._draw_lane_lines()
      if ui_state.status != UIStatus.DISENGAGED:
        #self._draw_lane_lines()
        self._draw_path(sm)

      self._draw_blindspots(sm)

      if render_lead_indicator and radar_state:
        self._draw_lead_indicator()
      if ui_state.show_radar_info > 0:
        self._draw_radar_info()
    from openpilot.selfdrive.ui.stop_object import draw_stop_object
    draw_stop_object(self, sm, started_frame=ui_state.started_frame)
      
  def _update_raw_points(self, model):
    """Update raw 3D points from model data"""
    self._path.raw_points = np.array([model.position.x, model.position.y, model.position.z], dtype=np.float32).T

    for i, lane_line in enumerate(model.laneLines):
      self._lane_lines[i].raw_points = np.array([lane_line.x, lane_line.y, lane_line.z], dtype=np.float32).T

    for i, road_edge in enumerate(model.roadEdges):
      self._road_edges[i].raw_points = np.array([road_edge.x, road_edge.y, road_edge.z], dtype=np.float32).T

    self._lane_line_probs = np.array(model.laneLineProbs, dtype=np.float32)
    self._road_edge_stds = np.array(model.roadEdgeStds, dtype=np.float32)
    self._acceleration_x = np.array(model.acceleration.x, dtype=np.float32)

  def _update_leads(self, radar_state, path_x_array):
    """Update positions of lead vehicles"""
    self._lead_vehicles = [LeadVehicle(), LeadVehicle()]
    leads = [radar_state.leadOne, radar_state.leadTwo]

    for i, lead_data in enumerate(leads):
      if lead_data and lead_data.status:
        d_rel, y_rel, v_rel = lead_data.dRel, lead_data.yRel, lead_data.vRel
        idx = self._get_path_length_idx(path_x_array, d_rel)

        # Get z-coordinate from path at the lead vehicle position
        z = self._path.raw_points[idx, 2] if idx < len(self._path.raw_points) else 0.0
        point = self._map_to_screen(d_rel, -y_rel, z + self._path_offset_z)
        if point:
          self._lead_vehicles[i] = self._update_lead_vehicle(d_rel, v_rel, point, self._rect)
          
  def _update_leads_carrot(self, radar_state, path_x_array):
    """Carrot: draw leadOne as outline rectangle."""
    self._lead_vehicles = [LeadVehicle(), LeadVehicle()]

    lead_one = radar_state.leadOne

    def _filter_pt(slot: int, pt: tuple[float, float], alpha: float = 0.2):
      prev = self._lead_pt_filt[slot]
      if prev is None:
        self._lead_pt_filt[slot] = (float(pt[0]), float(pt[1]))
        return float(pt[0]), float(pt[1])

      alpha = 1.0 - self._display_clock.retention(1.0 - alpha)
      x = prev[0] + (float(pt[0]) - prev[0]) * alpha
      y = prev[1] + (float(pt[1]) - prev[1]) * alpha
      self._lead_pt_filt[slot] = (x, y)
      return x, y

    if lead_one and lead_one.status:
      d_rel = float(lead_one.dRel)
      y_rel = float(lead_one.yRel)

      idx = self._get_path_length_idx(path_x_array, d_rel)
      z = float(self._path.raw_points[idx, 2]) if idx < len(self._path.raw_points) else 0.0

      pt_left = self._map_to_screen(d_rel, -y_rel - 1.2, z + self._path_offset_z)
      pt_right = self._map_to_screen(d_rel, -y_rel + 1.2, z + self._path_offset_z)

      if pt_left and pt_right:
        # 중심은 좌/우 평균으로 잡아야 함
        center_x = (pt_left[0] + pt_right[0]) * 0.5
        center_y = (pt_left[1] + pt_right[1]) * 0.5

        # 폭은 좌우 거리 기준, clamp는 유지
        path_width = float(np.clip(abs(pt_right[0] - pt_left[0]), 60.0, 400.0))

        # 중심 필터
        pt_x, pt_y = _filter_pt(0, (center_x, center_y), alpha=0.2)

        rect_x_min = self._rect.x
        rect_x_max = self._rect.x + self._rect.width
        rect_y_min = self._rect.y
        rect_y_max = self._rect.y + self._rect.height

        half_w = path_width * 0.5
        rect_h = path_width * 0.8

        left = float(np.clip(pt_x - half_w, rect_x_min, rect_x_max))
        right = float(np.clip(pt_x + half_w, rect_x_min, rect_x_max))
        bottom_y = float(np.clip(pt_y, rect_y_min, rect_y_max))
        top = float(np.clip(pt_y - rect_h, rect_y_min, rect_y_max))

        if not bool(lead_one.radar):
          c = rl.Color(0, 120, 255, 255)   # BLUE
        else:
          track_id = int(getattr(lead_one, "radarTrackId", 0))
          if track_id in (0, 1):
            c = rl.Color(201, 34, 49, 255) # RED
          else:
            c = rl.Color(255, 115, 0, 255) # ORANGE

        self._lead_vehicles[0] = LeadVehicle(
          rect=[(left, top), (right, top), (right, bottom_y), (left, bottom_y)],
          color=c,
        )
      else:
        self._lead_pt_filt[0] = None
        
  def _update_model(self, lead, path_x_array, car_state=None):
    """Update model visualization data based on model message"""
    max_distance = np.clip(path_x_array[-1], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE)
    max_idx = self._get_path_length_idx(self._lane_lines[0].raw_points[:, 0], max_distance)

    # Update lane lines using raw points
    line_width_factor = 0.12
    self._lane_marking_codes = [-1, car_state.leftLaneLine if car_state else -1, car_state.rightLaneLine if car_state else -1, -1]
    for i, lane_line in enumerate(self._lane_lines):
      if i in (1, 2):
        line_width_factor = 0.16
      lane_line.projected_points = self._map_line_to_polygon(
        lane_line.raw_points, line_width_factor * self._lane_line_probs[i], 0.0, max_idx
      )
      self._lane_marking_segments[i] = []
      code = self._lane_marking_codes[i]
      if code >= 0 and self._lane_line_probs[i] > 0.3:
        dashed = code % 10 == 0
        segments = lane_dash_segments(lane_line.raw_points, float(max_distance)) if dashed else [lane_line.raw_points]
        for segment in segments:
          points = self._map_line_to_polygon(segment, 0.05, 0.0, len(segment) - 1 if dashed else max_idx)
          if points.size:
            self._lane_marking_segments[i].append(points)
        if i == 1 and code % 10 == 4:
          points = self._map_line_to_polygon(lane_line.raw_points + np.array([0, -0.3, 0], dtype=np.float32), 0.05, 0.0, max_idx)
          if points.size:
            self._lane_marking_segments[i].append(points)

    # Update road edges using raw points
    for road_edge in self._road_edges:
      road_edge.projected_points = self._map_line_to_polygon(road_edge.raw_points, line_width_factor, 0.0, max_idx)

    # Update path using raw points
    if lead and lead.status:
      lead_d = lead.dRel * 2.0
      max_distance = np.clip(lead_d - min(lead_d * 0.35, 10.0), 0.0, max_distance)

    soon_acceleration = self._acceleration_x[len(self._acceleration_x) // 4] if len(self._acceleration_x) > 0 else 0
    self._display_clock.update_filter(self._acceleration_x_filter, soon_acceleration)
    self._display_clock.update_filter(self._acceleration_x_filter2, soon_acceleration)

    # make path width wider/thinner when initially braking/accelerating
    if self._experimental_mode and False:
      high_pass_acceleration = self._acceleration_x_filter.x - self._acceleration_x_filter2.x
      y_off = np.interp(high_pass_acceleration, [-1, 0, 1], [0.9 * 2, 0.9, 0.9 / 2])
    else:
      y_off = 0.9

    max_idx = self._get_path_length_idx(path_x_array, max_distance)
    self._path.projected_points = self._map_line_to_polygon(
      self._path.raw_points, y_off, self._path_offset_z, max_idx, allow_invert=False
    )

    self._update_experimental_gradient()

  def _update_experimental_gradient(self):
    """Pre-calculate experimental mode gradient colors"""
    if not self._experimental_mode:
      return

    max_len = min(len(self._path.projected_points) // 2, len(self._acceleration_x))

    segment_colors = []
    gradient_stops = []

    i = 0
    while i < max_len:
      # Some points (screen space) are out of frame (rect space)
      track_y = self._path.projected_points[i][1]
      if track_y < self._rect.y or track_y > (self._rect.y + self._rect.height):
        i += 1
        continue

      # Calculate color based on acceleration (0 is bottom, 1 is top)
      lin_grad_point = 1 - (track_y - self._rect.y) / self._rect.height

      # speed up: 120, slow down: 0
      path_hue = np.clip(60 + self._acceleration_x[i] * 35, 0, 120)

      saturation = min(abs(self._acceleration_x[i] * 1.5), 1)
      lightness = np.interp(saturation, [0.0, 1.0], [0.95, 0.62])
      alpha = np.interp(lin_grad_point, [0.75 / 2.0, 0.75], [0.4, 0.0])

      # Use HSL to RGB conversion
      color = self._hsla_to_color(path_hue / 360.0, saturation, lightness, alpha)

      gradient_stops.append(lin_grad_point)
      segment_colors.append(color)

      # Skip a point, unless next is last
      i += 1 + (1 if (i + 2) < max_len else 0)

    # Store the gradient in the path object
    self._exp_gradient.colors = segment_colors
    self._exp_gradient.stops = gradient_stops

  def _update_lead_vehicle(self, d_rel, v_rel, point, rect):
    speed_buff, lead_buff = 10.0, 40.0

    # Calculate fill alpha
    fill_alpha = 0
    if d_rel < lead_buff:
      fill_alpha = 255 * (1.0 - (d_rel / lead_buff))
      if v_rel < 0:
        fill_alpha += 255 * (-1 * (v_rel / speed_buff))
      fill_alpha = min(fill_alpha, 255)

    # Calculate size and position
    sz = np.clip((25 * 30) / (d_rel / 3 + 30), 15.0, 30.0) * 1
    x = np.clip(point[0], 0.0, rect.width - sz / 2)
    y = min(point[1], rect.height - sz * 0.6)

    g_xo = sz / 5
    g_yo = sz / 10

    glow = [(x + (sz * 1.35) + g_xo, y + sz + g_yo), (x, y - g_yo), (x - (sz * 1.35) - g_xo, y + sz + g_yo)]
    chevron = [(x + (sz * 1.25), y + sz), (x, y), (x - (sz * 1.25), y + sz)]

    return LeadVehicle(glow=glow, chevron=chevron, fill_alpha=int(fill_alpha))

  def _get_ll_color(self, prob: float, adjacent: bool, left: bool):
    alpha = np.clip(prob, 0.0, 0.7)
    if adjacent:
      _base_color = LANE_LINE_COLORS.get(ui_state.status, LANE_LINE_COLORS[UIStatus.DISENGAGED if not ui_state.lat_active else UIStatus.ENGAGED])
      color = rl.Color(_base_color.r, _base_color.g, _base_color.b, int(alpha * 255))

      # turn adjacent lls orange if torque is high
      torque = self._torque_filter.x
      high_torque = abs(torque) > 0.6
      if high_torque and (left == (torque > 0)):
        color = blend_colors(
          color,
          rl.Color(255, 115, 0, int(alpha * 255)),  # orange
          np.interp(abs(torque), [0.6, 0.8], [0.0, 1.0])
        )
    else:
      color = rl.Color(255, 255, 255, int(alpha * 255))

    if ui_state.status == UIStatus.DISENGAGED and not ui_state.lat_active:
      color = rl.Color(0, 0, 0, int(alpha * 255))

    return color

  def _draw_lane_lines(self):
    """Draw lane lines and road edges"""
    """Two closest lines should be green (lane line or road edges)"""
    for i, lane_line in enumerate(self._lane_lines):
      code = self._lane_marking_codes[i]
      if code >= 0 and self._lane_line_probs[i] > 0.3:
        color = rl.Color(218, 202, 37, 220) if code >= 20 else rl.Color(255, 255, 255, 220)
        for points in self._lane_marking_segments[i]:
          draw_polygon(self._rect, points, color)
        continue
      if lane_line.projected_points.size == 0:
        continue

      color = self._get_ll_color(float(self._lane_line_probs[i]), i in (1, 2), i in (0, 1))
      draw_polygon(self._rect, lane_line.projected_points, color)

    for i, road_edge in enumerate(self._road_edges):
      if road_edge.projected_points.size == 0:
        continue

      # if closest lane lines are not confident, make road edges green
      color = self._get_ll_color(float(1.0 - self._road_edge_stds[i]), float(self._lane_line_probs[i + 1]) < 0.25, i == 0)
      draw_polygon(self._rect, road_edge.projected_points, color)

  def _draw_blindspots(self, sm):
    if not (sm.valid['carState'] and sm.alive['carState'] and sm.valid['modelV2'] and sm.alive['modelV2']):
      return
    car_state = sm['carState']
    if not (car_state.leftBlindspot or car_state.rightBlindspot) or self._path.raw_points.shape[0] < 2:
      return
    max_idx = self._get_path_length_idx(self._path.raw_points[:, 0], 40.0)
    source_packet = blindspot_source_packet(sm, time.monotonic_ns())
    for side, active, shift in (
      ("left", car_state.leftBlindspot, -1.7),
      ("right", car_state.rightBlindspot, 1.7),
    ):
      if active:
        oem, vision = blindspot_sources(car_state, side, source_packet)
        color = rl.Color(255, 128, 128, 190) if oem and vision else \
                rl.Color(64, 156, 255, 165) if vision else rl.Color(255, 215, 0, 150)
        polygon = project_blindspot_barrier(self._path.raw_points[:max_idx + 1], shift, self._car_space_transform, self._clip_region)
        for quad in blindspot_barrier_quads(polygon):
          draw_polygon(self._rect, quad, color)

  def _draw_path(self, sm):
    """Draw path with dynamic coloring based on mode and throttle state."""
    if not self._path.projected_points.size:
      return

    allow_throttle = sm['longitudinalPlan'].allowThrottle or not self._longitudinal_control
    self._display_clock.update_filter(self._blend_filter, int(allow_throttle))

    if self._experimental_mode:
      # Draw with acceleration coloring
      if ui_state.status == UIStatus.DISENGAGED:
        draw_polygon(self._rect, self._path.projected_points, rl.Color(0, 0, 0, 90))
      elif len(self._exp_gradient.colors) > 1:
        draw_polygon(self._rect, self._path.projected_points, gradient=self._exp_gradient)
      else:
        draw_polygon(self._rect, self._path.projected_points, rl.Color(255, 255, 255, 30))
    else:
      # Blend throttle/no throttle colors based on transition
      blend_factor = round(self._blend_filter.x * 100) / 100
      blended_colors = self._blend_colors(NO_THROTTLE_COLORS, THROTTLE_COLORS, blend_factor)
      gradient = Gradient(
        start=(0.0, 1.0),  # Bottom of path
        end=(0.0, 0.0),  # Top of path
        colors=blended_colors,
        stops=[0.0, 0.5, 1.0],
      )

      if ui_state.status == UIStatus.DISENGAGED:
        draw_polygon(self._rect, self._path.projected_points, rl.Color(0, 0, 0, 90))
      else:
        draw_polygon(self._rect, self._path.projected_points, gradient=gradient)

  def _draw_lead_indicator_old(self):
    # Draw lead vehicles if available
    for lead in self._lead_vehicles:
      if not lead.glow or not lead.chevron:
        continue

      rl.draw_triangle_fan(lead.glow, len(lead.glow), rl.Color(218, 202, 37, 255))
      rl.draw_triangle_fan(lead.chevron, len(lead.chevron), rl.Color(201, 34, 49, lead.fill_alpha))

  def _draw_lead_indicator(self):
    # Carrot: draw outline rectangles only (no fill)
    thickness = 4.0  # 원하는 두께 (float)

    for lead in self._lead_vehicles:
      if not lead.rect or lead.color is None:
        continue

      pts = lead.rect
      c = lead.color

      if native_draw.try_outline(rl, pts, c, thickness):
        continue
      rl.draw_line_ex(pts[0], pts[1], thickness, c)
      rl.draw_line_ex(pts[1], pts[2], thickness, c)
      rl.draw_line_ex(pts[2], pts[3], thickness, c)
      rl.draw_line_ex(pts[3], pts[0], thickness, c)
    
  @staticmethod
  def _get_path_length_idx(pos_x_array: np.ndarray, path_height: float) -> int:
    """Get the index corresponding to the given path height"""
    if len(pos_x_array) == 0:
      return 0
    indices = np.where(pos_x_array <= path_height)[0]
    return indices[-1] if indices.size > 0 else 0

  def _map_to_screen(self, in_x, in_y, in_z):
    """Project a point in car space to screen space"""
    input_pt = np.array([in_x, in_y, in_z])
    pt = self._car_space_transform @ input_pt

    if abs(pt[2]) < 1e-6:
      return None

    x, y = pt[0] / pt[2], pt[1] / pt[2]

    clip = self._clip_region
    if not (clip.x <= x <= clip.x + clip.width and clip.y <= y <= clip.y + clip.height):
      return None

    return (x, y)

  def _map_line_to_polygon(self, line: np.ndarray, y_off: float, z_off: float, max_idx: int, allow_invert: bool = True) -> np.ndarray:
    return native_geometry.project_ribbon(line, y_off, z_off, max_idx, self._car_space_transform,
                                          self._clip_region, allow_invert)

  @staticmethod
  def _hsla_to_color(h, s, l, a):
    rgb = colorsys.hls_to_rgb(h, l, s)
    return rl.Color(
      int(rgb[0] * 255),
      int(rgb[1] * 255),
      int(rgb[2] * 255),
      int(a * 255)
    )

  @staticmethod
  def _blend_colors(begin_colors, end_colors, t):
    if t >= 1.0:
      return end_colors
    if t <= 0.0:
      return begin_colors

    inv_t = 1.0 - t
    return [rl.Color(
      int(inv_t * start.r + t * end.r),
      int(inv_t * start.g + t * end.g),
      int(inv_t * start.b + t * end.b),
      int(inv_t * start.a + t * end.a)
    ) for start, end in zip(begin_colors, end_colors, strict=True)]

  def _get_radar_info_color(self, lead, v_sum: float):
    radar = bool(getattr(lead, "radar", False))
    model_prob = float(getattr(lead, "modelProb", 0.0))

    if not radar:
      return rl.Color(0, 0, 255, 220)      # BLUE
    elif abs(model_prob - 0.01) < 0.001:
      return rl.Color(0, 203, 0, 220)       # GREEN
    elif v_sum > 0.0:
      return rl.Color(255, 175, 3, 220)      # ORANGE
    else:
      return rl.Color(255, 0, 0, 220)      # RED

  def _update_radar_info(self, radar_state, path_x_array):
    self._radar_info_items = []

    if self._lane_lines[2].raw_points.shape[0] == 0:
      return

    leads_groups = [
      radar_state.leadsLeft,
      radar_state.leadsRight,
      radar_state.leadsCenter,
    ]

    rect_x_min = self._rect.x
    rect_x_max = self._rect.x + self._rect.width
    rect_y_min = self._rect.y
    rect_y_max = self._rect.y + self._rect.height

    for leads in leads_groups:
      for l in leads:
        d_rel = float(getattr(l, "dRel", 0.0))
        y_rel = float(getattr(l, "yRel", 0.0))

        if d_rel <= 2.5:
          continue

        idx = self._get_path_length_idx(self._lane_lines[2].raw_points[:, 0], d_rel)
        if idx >= len(self._lane_lines[2].raw_points):
          continue

        lane_z = float(self._lane_lines[2].raw_points[idx, 2]) - 0.61
        pt = self._map_to_screen(d_rel, -y_rel, lane_z)
        if not pt:
          continue

        x, y = float(pt[0]), float(pt[1])

        v = float(getattr(l, "vLeadK", 0.0))
        v_lat = float(getattr(l, "vLat", 0.0))
        v_abs = float(np.sqrt(v * v + v_lat * v_lat))
        v_sum = v_abs if v >= 0.0 else -v_abs

        # 정지는 기존처럼 "*"
        if v_abs <= 3.0:
          self._radar_info_items.append(
            RadarInfoItem(
              x=x,
              y=y,
              w=18.0,
              h=18.0,
              text="*",
              color=rl.Color(255, 255, 255, 230),
              is_star=True,
            )
          )
          continue

        speed_val = v_sum * (3.6 if ui_state.is_metric else 2.2369363)
        text = f"{speed_val:.0f}"

        font_size = 22
        pad_x = 6
        pad_y = 2

        text_w = rl.measure_text(text, font_size)
        box_w = float(text_w + pad_x * 2)
        box_h = float(font_size + pad_y * 2)

        box_x = float(np.clip(x - box_w * 0.5, rect_x_min, rect_x_max - box_w))
        box_y = float(np.clip(y - box_h * 0.5, rect_y_min, rect_y_max - box_h))

        color = self._get_radar_info_color(l, v_sum)

        self._radar_info_items.append(
          RadarInfoItem(
            x=box_x,
            y=box_y,
            w=float(box_w),
            h=float(box_h),
            text=text,
            color=color,
            is_star=False,
          )
        )


  def _draw_radar_info(self):
    if not self._radar_info_items:
      return

    font_size = 22

    for item in self._radar_info_items:
      if item.color is None:
        continue

      if item.is_star:
        tw = rl.measure_text(item.text, font_size)
        draw_text_ui_style(item.text, item.x, item.y, font_size, item.color, font=self._font_display, border_width=1.0, shadow_offset=8.0, align="center", y_offset=0.0)
        continue

      # 박스
      rl.draw_rectangle_rounded(
        rl.Rectangle(item.x, item.y, item.w, item.h),
        0.28,
        8,
        item.color,
      )

      # 텍스트 중앙정렬
      tw = rl.measure_text(item.text, font_size)
      tx = int(item.x + (item.w - tw) / 2)
      ty = int(item.y + (item.h - font_size) / 2 - 1)

      #rl.draw_text(
      #  item.text,
      #  tx,
      #  ty,
      #  font_size,
      #  rl.WHITE,
      #)

      draw_text_ui_style(item.text, tx, ty, font_size, rl.WHITE, font=self._font_display, border_width=1.0, shadow_offset=8.0, align="left_top", y_offset=0.0)

  def _draw_polygon_outline_carrot(self, points, color, width=1.5):
    if len(points) < 2:
      return
    for i in range(len(points) - 1):
      p1 = rl.Vector2(float(points[i][0]), float(points[i][1]))
      p2 = rl.Vector2(float(points[i+1][0]), float(points[i+1][1]))
      rl.draw_line_ex(p1, p2, width, color)

  def _project_point_to_screen(self, x: float, y: float, z: float):
    pt = np.array([x, y, z], dtype=np.float32)
    proj = self._car_space_transform @ pt
    if abs(proj[2]) < 1e-5 or proj[2] <= 0:
      return None
    px = proj[0] / proj[2]
    py = proj[1] / proj[2]
    return (px, py)

  def _draw_fsd_visualization(self, sm):
    if not sm.valid['modelV2']:
      return

    lane_zero = self._lane_lines[0].raw_points
    if lane_zero.shape[0] == 0:
      return

    is_night = getattr(self, '_fsd_is_night', True)
    max_distance = float(np.clip(lane_zero[-1, 0], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE))
    max_idx = self._get_path_length_idx(lane_zero[:, 0], max_distance)

    # 1. Driving Path Ribbon (Tesla FSD Cyan at night, Deep Blue in day)
    if self._path.raw_points.shape[0] > 0:
      path_max_dist = float(np.clip(self._path.raw_points[-1, 0], MIN_DRAW_DISTANCE, MAX_DRAW_DISTANCE))
      radar_state = sm['radarState'] if sm.valid['radarState'] else None
      if radar_state and radar_state.leadOne and radar_state.leadOne.status:
        lead_d = float(radar_state.leadOne.dRel)
        path_max_dist = min(path_max_dist, max(lead_d - 1.5, 4.0))
      p_idx = self._get_path_length_idx(self._path.raw_points[:, 0], path_max_dist)
      path_poly = self._map_line_to_polygon(self._path.raw_points, 0.82, self._path_offset_z, p_idx, allow_invert=False)
      if path_poly.size > 0:
        if is_night:
          draw_polygon_solid(path_poly, rl.Color(0, 160, 255, 140))
          if len(path_poly) >= 4:
            n = len(path_poly) // 2
            self._draw_polygon_outline_carrot(path_poly[:n], rl.Color(0, 225, 255, 210), 1.5)
            self._draw_polygon_outline_carrot(path_poly[n:], rl.Color(0, 225, 255, 210), 1.5)
        else:
          draw_polygon_solid(path_poly, rl.Color(24, 118, 238, 195))
          if len(path_poly) >= 4:
            n = len(path_poly) // 2
            self._draw_polygon_outline_carrot(path_poly[:n], rl.Color(16, 95, 210, 240), 1.5)
            self._draw_polygon_outline_carrot(path_poly[n:], rl.Color(16, 95, 210, 240), 1.5)

    # 2. 4 Lane Markings (White solid/dashed with Day/Night tone)
    self._draw_fsd_lanes(is_night, max_idx)

    # 3. 3D Lead Vehicle
    radar_state = sm['radarState'] if sm.valid['radarState'] else None
    if radar_state:
      self._draw_fsd_lead_vehicle(radar_state, is_night)
      self._draw_fsd_adjacent_vehicles(radar_state, is_night)

    # 4. 3D Ego Vehicle
    self._draw_fsd_ego_vehicle(sm, is_night)

  def _draw_fsd_lanes(self, is_night: bool, max_idx: int):
    # Only draw lanes with confidence >= 0.35 to prevent phantom straight lines
    for idx, lane in enumerate(self._lane_lines):
      if idx >= len(self._lane_line_probs) or self._lane_line_probs[idx] < 0.35:
        continue
      if lane.raw_points.shape[0] == 0:
        continue
      prob = float(self._lane_line_probs[idx])
      alpha = int(np.clip(prob * 255, 140, 255))
      width_m = (0.055 if idx in (1, 2) else 0.038) if is_night else (0.065 if idx in (1, 2) else 0.045)
      poly = self._map_line_to_polygon(lane.raw_points, width_m, self._path_offset_z, max_idx, allow_invert=False)
      if poly.size > 0:
        if is_night:
          col = rl.Color(255, 255, 255, alpha) if idx in (1, 2) else rl.Color(195, 210, 230, int(alpha * 0.70))
        else:
          col = rl.Color(255, 255, 255, 255) if idx in (1, 2) else rl.Color(240, 245, 255, int(alpha * 0.85))
        draw_polygon_solid(poly, col)

  def _draw_fsd_lead_vehicles(self, radar_state, is_night: bool):
    leads = [(radar_state.leadOne, "L1"), (getattr(radar_state, 'leadTwo', None), "L2")]
    for lead, tag in leads:
      if not (lead and lead.status):
        continue

      d_rel = float(lead.dRel)
      y_rel = float(lead.yRel)
      v_rel = float(getattr(lead, 'vRel', 0.0))
      v_rel_kph = v_rel * 3.6

      pt = self._project_point_to_screen(d_rel, y_rel, self._path_offset_z)
      if not pt:
        continue

      x, y = pt
      # Scaled for mici with realistic perspective
      vw = float(np.clip(900.0 / (d_rel + 4.5), 14.0, 120.0))
      vh = float(vw * 0.70)

      # 1. Ground shadow
      rl.draw_ellipse(int(x), int(y + vh * 0.44), vw * 0.58, vh * 0.20, rl.Color(0, 0, 0, 110))

      # 2. Cabin and lower body
      cabin_w = vw * 0.82
      cabin_rect = rl.Rectangle(x - cabin_w / 2, y - vh * 0.50, cabin_w, vh * 0.48)
      lower_rect = rl.Rectangle(x - vw / 2, y - vh * 0.10, vw, vh * 0.55)

      if is_night:
        body_col = rl.Color(160, 170, 185, 250)
        body_line = rl.Color(225, 235, 248, 255)
        glass_col = rl.Color(18, 24, 32, 255)
      else:
        body_col = rl.Color(42, 52, 68, 250)
        body_line = rl.Color(215, 228, 245, 255)
        glass_col = rl.Color(22, 28, 38, 255)

      rl.draw_rectangle_rounded(cabin_rect, 0.42, 4, body_col)
      rl.draw_rectangle_rounded_lines_ex(cabin_rect, 0.42, 4, 1.0, body_line)
      rl.draw_rectangle_rounded(rl.Rectangle(x - cabin_w * 0.40, y - vh * 0.44, cabin_w * 0.80, vh * 0.32), 0.30, 3, glass_col)
      rl.draw_rectangle_rounded(lower_rect, 0.28, 4, body_col)
      rl.draw_rectangle_rounded_lines_ex(lower_rect, 0.28, 4, 1.0, body_line)

      # 3. Tail lights
      is_lead_braking = v_rel < -0.8
      tail_col = rl.Color(255, 25, 25, 255) if is_lead_braking else (rl.Color(235, 30, 30, 240) if is_night else rl.Color(215, 25, 25, 255))
      tw = max(vw * 0.24, 4.0)
      th = max(vh * 0.15, 2.0)
      tail_y = y - vh * 0.05
      rl.draw_rectangle_rounded(rl.Rectangle(x - vw * 0.46, tail_y, tw, th), 0.4, 2, tail_col)
      rl.draw_rectangle_rounded(rl.Rectangle(x + vw * 0.46 - tw, tail_y, tw, th), 0.4, 2, tail_col)

      # 4. L1/L2 Metric badge
      if tag == "L1":
        if abs(v_rel_kph) >= 1.0:
          dist_str = f"L1 {d_rel:.0f}m {v_rel_kph:+.0f}"
        else:
          dist_str = f"L1 {d_rel:.0f}m"
      else:
        dist_str = f"L2 {d_rel:.0f}m"

      ttc = -d_rel / v_rel if v_rel < -0.5 else 999.0
      badge_font = 13 if vw > 45 else 10
      ts = measure_text_cached(self._font_display, dist_str, badge_font)
      badge_pad_x = 4
      badge_pad_y = 2
      bx = x - (ts.x + badge_pad_x * 2) / 2
      by = y - vh * 0.50 - ts.y - badge_pad_y * 2 - 4
      badge_rect = rl.Rectangle(bx, by, ts.x + badge_pad_x * 2, ts.y + badge_pad_y * 2)

      if ttc < 2.5:
        b_color = rl.Color(190, 25, 25, 225)
        t_color = rl.Color(255, 255, 255, 255)
      elif ttc < 4.5:
        b_color = rl.Color(200, 115, 15, 225)
        t_color = rl.Color(255, 255, 255, 255)
      elif is_night:
        b_color = rl.Color(12, 16, 24, 210)
        t_color = rl.Color(0, 235, 255, 255) if tag == "L1" else rl.Color(190, 205, 225, 230)
      else:
        b_color = rl.Color(255, 255, 255, 230)
        t_color = rl.Color(16, 85, 190, 255) if tag == "L1" else rl.Color(45, 55, 70, 235)

      rl.draw_rectangle_rounded(badge_rect, 0.4, 3, b_color)
      rl.draw_rectangle_rounded_lines_ex(badge_rect, 0.4, 3, 1.0, t_color)
      draw_text_ui_style(dist_str, bx + badge_pad_x, by + badge_pad_y, badge_font, t_color,
                         font=self._font_display, align="left_top", y_offset=0.0)

  # Maintain backward compatibility if called
  _draw_fsd_lead_vehicle = _draw_fsd_lead_vehicles

  def _draw_fsd_adjacent_vehicles(self, radar_state, is_night: bool):
    leads = []
    if getattr(radar_state, 'leadTwo', None) and radar_state.leadTwo.status:
      leads.append(radar_state.leadTwo)

    for lead in leads:
      d_rel = float(lead.dRel)
      y_rel = float(lead.yRel)
      if d_rel < 3.0 or d_rel > 75.0:
        continue

      pt = self._project_point_to_screen(d_rel, y_rel, self._path_offset_z)
      if not pt:
        continue

      x, y = pt
      vw = float(np.clip(700.0 / (d_rel + 4.5), 12.0, 70.0))
      vh = float(vw * 0.70)

      rl.draw_ellipse(int(x), int(y + vh * 0.44), vw * 0.52, vh * 0.20, rl.Color(0, 0, 0, 90))
      adj_rect = rl.Rectangle(x - vw / 2, y - vh / 2, vw, vh)
      adj_body = rl.Color(135, 142, 155, 220) if is_night else rl.Color(110, 120, 135, 240)
      adj_line = rl.Color(170, 178, 190, 230) if is_night else rl.Color(55, 65, 80, 255)
      rl.draw_rectangle_rounded(adj_rect, 0.32, 4, adj_body)
      rl.draw_rectangle_rounded_lines_ex(adj_rect, 0.32, 4, 1.0, adj_line)

  def _draw_fsd_ego_vehicle(self, sm, is_night: bool = True):
    # Driver POV: Half steering wheel at bottom center
    rect = self.rect
    center_x = rect.x + rect.width / 2
    wheel_center_y = rect.y + rect.height + 10.0
    r_outer = 42.0
    r_inner = 31.0

    car_state = sm['carState'] if sm.valid['carState'] else None
    steer_angle = float(np.clip(-car_state.steeringAngleDeg, -75.0, 75.0)) if car_state else 0.0

    start_ang = 180.0 + steer_angle
    end_ang = 360.0 + steer_angle

    if is_night:
      rim_col = rl.Color(30, 36, 46, 250)
      rim_line = rl.Color(80, 95, 118, 240)
      hub_col = rl.Color(22, 28, 36, 255)
      hub_line = rl.Color(55, 68, 85, 230)
      accent_col = rl.Color(0, 220, 255, 255)
    else:
      rim_col = rl.Color(42, 50, 62, 250)
      rim_line = rl.Color(190, 202, 218, 255)
      hub_col = rl.Color(32, 38, 48, 255)
      hub_line = rl.Color(160, 175, 195, 240)
      accent_col = rl.Color(22, 115, 232, 255)

    rl.draw_ring(rl.Vector2(center_x, wheel_center_y), r_inner, r_outer,
                 start_ang, end_ang, 28, rim_col)
    rl.draw_ring_lines(rl.Vector2(center_x, wheel_center_y), r_inner, r_outer,
                       start_ang, end_ang, 28, rim_line)

    center_marker_ang = 270.0 + steer_angle
    rl.draw_ring(rl.Vector2(center_x, wheel_center_y), r_inner - 0.5, r_outer + 0.5,
                 center_marker_ang - 4.0, center_marker_ang + 4.0, 8, accent_col)

    hub_r = 18.0
    rl.draw_circle_sector(rl.Vector2(center_x, wheel_center_y), hub_r,
                          start_ang, end_ang, 20, hub_col)
    rl.draw_circle_sector_lines(rl.Vector2(center_x, wheel_center_y), hub_r,
                                start_ang, end_ang, 20, hub_line)

    rl.draw_circle(int(center_x), int(wheel_center_y - 8), 3.5, accent_col)

    if car_state:
      if car_state.leftBlindspot:
        rl.draw_ring(rl.Vector2(center_x, wheel_center_y), r_inner - 1, r_outer + 2,
                     start_ang, start_ang + 35.0, 10, rl.Color(255, 180, 0, 220))
      if car_state.rightBlindspot:
        rl.draw_ring(rl.Vector2(center_x, wheel_center_y), r_inner - 1, r_outer + 2,
                     end_ang - 35.0, end_ang, 10, rl.Color(255, 180, 0, 220))

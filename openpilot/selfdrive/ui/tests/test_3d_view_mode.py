from types import SimpleNamespace
from unittest.mock import MagicMock
import pyray as rl
import pytest

from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.onroad.view_3d_button import View3dButton
from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer


@pytest.fixture(autouse=True)
def mock_gui_app(monkeypatch):
  mock_font = MagicMock()
  monkeypatch.setattr(gui_app, "font", lambda weight=None: mock_font)
  monkeypatch.setattr(gui_app, "texture", lambda *args, **kwargs: MagicMock())
  monkeypatch.setattr("openpilot.system.ui.lib.text_measure.measure_text_cached", lambda font, text, size: rl.Vector2(len(str(text)) * 10, 30))
  monkeypatch.setattr("openpilot.selfdrive.ui.onroad.view_3d_button.measure_text_cached", lambda font, text, size: rl.Vector2(len(str(text)) * 10, 30))
  for fn in ("draw_text_ex", "draw_rectangle_rounded", "draw_rectangle_rounded_lines_ex",
            "draw_ellipse", "draw_triangle", "draw_circle", "draw_rectangle", "draw_rectangle_lines",
            "draw_ring", "draw_ring_lines", "draw_circle_sector", "draw_circle_sector_lines",
            "draw_rectangle_gradient_v", "draw_ellipse_lines", "draw_line_ex"):
    monkeypatch.setattr(rl, fn, lambda *args, **kwargs: None)
  monkeypatch.setattr("openpilot.selfdrive.ui.mici.onroad.model_renderer.draw_text_ui_style", lambda *args, **kwargs: None)
  return gui_app


class FakeParams:
  def __init__(self, initial=False):
    self.data = {"View3dMode": initial, "FsdViewMode": initial}

  def get_bool(self, key):
    return self.data.get(key, False)

  def put_bool(self, key, val):
    self.data[key] = bool(val)

  def put_nonblocking(self, key, val):
    pass


def test_view_3d_button_toggle(monkeypatch):
  fake_params = FakeParams(initial=False)
  monkeypatch.setattr("openpilot.selfdrive.ui.onroad.view_3d_button.Params", lambda: fake_params)

  btn = View3dButton(width=150, height=68)
  assert btn.view_3d_mode is False
  assert btn.fsd_mode is False

  btn._handle_mouse_release(None)
  assert btn.view_3d_mode is True
  assert btn.fsd_mode is True
  assert fake_params.get_bool("View3dMode") is True

  btn._handle_mouse_release(None)
  assert btn.view_3d_mode is False
  assert btn.fsd_mode is False
  assert fake_params.get_bool("View3dMode") is False


def test_model_renderer_view_3d_mode_switch():
  renderer = ModelRenderer()
  assert renderer._view_3d_mode is False
  assert renderer._fsd_mode is False

  renderer.set_view_3d_mode(True)
  assert renderer._view_3d_mode is True
  assert renderer._fsd_mode is True

  renderer.set_view_3d_mode(False)
  assert renderer._view_3d_mode is False
  assert renderer._fsd_mode is False


def test_model_renderer_view_3d_draw_call():
  renderer = ModelRenderer()
  renderer.set_view_3d_mode(True)

  calls = []
  renderer._draw_3d_view_visualization = lambda sm: calls.append("3d_view")
  renderer._draw_path_carrot = lambda sm: calls.append("path")
  renderer._render_diagnostics = SimpleNamespace(
    start=lambda: None,
    finish=lambda: None,
    call=lambda name, callback, *args: callback(*args),
    values={}
  )

  renderer._draw_carrot_overlays(object())
  assert calls == ["3d_view"]

  calls.clear()
  renderer.set_view_3d_mode(False)
  renderer._draw_lane_lines_carrot = lambda sm: calls.append("lanes")
  renderer._draw_blind_spot_carrot = lambda sm: calls.append("blindspot")
  renderer._draw_radar_info_carrot = lambda sm: calls.append("radar")

  renderer._draw_carrot_overlays(object())
  assert calls == ["path", "lanes", "blindspot", "radar"]


def test_is_night_time_schedule():
  import time
  from openpilot.selfdrive.ui.onroad.augmented_road_view import is_night_time

  # Daytime: 12:00, 14:30, 08:00
  day_noon = time.struct_time((2026, 11, 13, 12, 0, 0, 4, 317, 0))
  assert is_night_time(day_noon) is False

  day_afternoon = time.struct_time((2026, 11, 13, 14, 30, 0, 4, 317, 0))
  assert is_night_time(day_afternoon) is False

  # Nighttime: 23:34, 01:00, 05:45
  night_late = time.struct_time((2026, 11, 13, 23, 34, 0, 4, 317, 0))
  assert is_night_time(night_late) is True

  night_early = time.struct_time((2026, 11, 13, 5, 0, 0, 4, 317, 0))
  assert is_night_time(night_early) is True


def test_hud_renderer_set_view_3d_theme():
  from openpilot.selfdrive.ui.onroad.hud_renderer import HudRenderer
  hud = HudRenderer()
  assert hud._view_3d_mode is False
  assert hud._fsd_mode is False
  assert hud._is_night is True

  hud.set_view_3d_theme(True, is_night=False)
  assert hud._view_3d_mode is True
  assert hud._fsd_mode is True
  assert hud._is_night is False

  hud.set_view_3d_theme(True, is_night=True)
  assert hud._view_3d_mode is True
  assert hud._fsd_mode is True
  assert hud._is_night is True


def test_mici_model_renderer_view_3d_mode():
  from openpilot.selfdrive.ui.mici.onroad.model_renderer import ModelRenderer as MiciModelRenderer
  mici_model = MiciModelRenderer()
  assert mici_model._view_3d_mode is False
  assert mici_model._fsd_mode is False
  mici_model.set_view_3d_mode(True, is_night=False)
  assert mici_model._view_3d_mode is True
  assert mici_model._fsd_mode is True
  assert mici_model._view_3d_is_night is False
  assert mici_model._fsd_is_night is False


def test_mici_hud_renderer_view_3d_theme():
  from openpilot.selfdrive.ui.mici.onroad.hud_renderer import HudRenderer as MiciHudRenderer
  mici_hud = MiciHudRenderer()
  assert mici_hud._view_3d_mode is False
  assert mici_hud._fsd_mode is False
  mici_hud.set_view_3d_theme(True, is_night=False)
  assert mici_hud._view_3d_mode is True
  assert mici_hud._fsd_mode is True
  assert mici_hud._is_night is False


def test_view_3d_button_compact_vs_full(monkeypatch):
  fake_params = FakeParams(initial=False)
  monkeypatch.setattr("openpilot.selfdrive.ui.onroad.view_3d_button.Params", lambda: fake_params)

  btn_compact = View3dButton(width=80, height=34)
  btn_full = View3dButton(width=210, height=78)

  # Check compact mode rendering
  btn_compact._render(rl.Rectangle(0, 0, 80, 34))
  # Check full mode rendering
  btn_full._render(rl.Rectangle(0, 0, 210, 78))


def test_view_3d_button_text_labels(monkeypatch):
  fake_params = FakeParams(initial=False)
  monkeypatch.setattr("openpilot.selfdrive.ui.onroad.view_3d_button.Params", lambda: fake_params)

  btn = View3dButton(width=210, height=78)
  # When in AR view (view_3d_mode=False) -> Offers 3D view
  assert btn.view_3d_mode is False

  # When in 3D view (view_3d_mode=True) -> Offers AR view
  btn._handle_mouse_release(None)
  assert btn.view_3d_mode is True


class FakeSubMaster(dict):
  def __init__(self, **kwargs):
    super().__init__(kwargs)
    self.valid = dict.fromkeys(kwargs, True)


def _car_state(v_ego=15.0, steer=0.0, left_bs=False, right_bs=False, left_code=-1, right_code=-1):
  return SimpleNamespace(vEgo=v_ego, steeringAngleDeg=steer, leftBlindspot=left_bs, rightBlindspot=right_bs,
                         leftLaneLine=left_code, rightLaneLine=right_code)


def _lead(d, y=0.0, v=0.0):
  return SimpleNamespace(status=True, dRel=d, yRel=y, vRel=v)


def _scene_renderer(curve_y=0.0):
  import numpy as np
  xs = np.linspace(0.0, 100.0, 33)
  ys = curve_y * (xs / 30.0) ** 2
  path = np.stack((xs, ys, np.zeros_like(xs)), axis=1).astype(np.float32)
  lanes = [SimpleNamespace(raw_points=np.stack((xs, ys + off, np.full_like(xs, 1.22)), axis=1).astype(np.float32))
           for off in (-5.4, -1.8, 1.8, 5.4)]
  road_edges = [SimpleNamespace(raw_points=np.stack((xs, ys + off, np.full_like(xs, 1.22)), axis=1).astype(np.float32))
                for off in (-7.2, 7.2)]
  return SimpleNamespace(_path=SimpleNamespace(raw_points=path), _lane_lines=lanes,
                         _lane_line_probs=[0.4, 0.9, 0.42, 0.2], _road_edges=road_edges,
                         _road_edge_stds=[0.3, 0.4], _path_offset_z=1.22)


C3_RECT = rl.Rectangle(0, 0, 2160, 1080)
C4_RECT = rl.Rectangle(0, 0, 536, 240)


@pytest.mark.parametrize("rect", [C3_RECT, C4_RECT])
def test_chase_camera_cruise_places_15m_lead_mid_screen(rect):
  from openpilot.selfdrive.ui.onroad.fsd_camera import ChaseCamera
  cam = ChaseCamera()
  cam.update(rect, 1.22, 0.0, 0.0, 0.0)
  x, y, vw, vh = cam.car_box(15.0, 0.0, 1.22)
  bottom = (y + vh / 2) / rect.height
  # The old real-camera projection put a 15 m lead at about 72% height.
  assert 0.42 < bottom < 0.62
  near_x, near_y, near_w, _ = cam.car_box(5.0, 0.0, 1.22)
  assert near_y > y and near_w > vw
  assert cam.ego_box() is None  # cruise view is behind the wheel, not the car


@pytest.mark.parametrize("rect", [C3_RECT, C4_RECT])
def test_chase_camera_wide_keeps_close_lead_visible_in_front_of_ego(rect):
  from openpilot.selfdrive.ui.onroad.fsd_camera import ChaseCamera
  cam = ChaseCamera()
  cam.update(rect, 1.22, 1.0, 0.0, 0.0)
  for d in (1.5, 3.8, 5.0):
    x, y, vw, vh = cam.car_box(d, 0.0, 1.22)
    assert rect.y < y - vh / 2 and y + vh / 2 < rect.y + rect.height
  _, lead_y, _, lead_h = cam.car_box(3.8, 0.0, 1.22)
  ex, ey, ew, eh = cam.ego_box()
  assert ey + eh / 2 <= rect.y + rect.height
  assert lead_y + lead_h / 2 <= ey - eh / 2 + 0.01 * rect.height
  assert ew < 0.25 * rect.width


def test_zoom_target_rules():
  from openpilot.selfdrive.ui.onroad.fsd_camera import zoom_target
  assert zoom_target(20.0, None, 0.0) == 0.0
  assert zoom_target(20.0, 30.0, 0.5) == 0.0
  assert zoom_target(3.0, 4.0, 0.0) == 1.0
  assert zoom_target(0.0, 10.0, 0.0) >= 0.6
  assert zoom_target(15.0, None, 6.5) == 1.0
  assert 0.0 < zoom_target(15.0, 6.0, 0.0) < 1.0


def test_chase_camera_zoom_is_smoothed():
  from openpilot.selfdrive.ui.onroad.fsd_camera import ChaseCamera
  cam = ChaseCamera()
  cam.update(C3_RECT, 1.22, 1.0, 0.0, 0.05)
  assert 0.0 < cam.zoom < 0.2
  for _ in range(60):
    cam.update(C3_RECT, 1.22, 1.0, 0.0, 0.05)
  assert cam.zoom > 0.99


def test_wide_ribbon_extends_beside_ego():
  from openpilot.selfdrive.ui.onroad.fsd_camera import ChaseCamera
  r = _scene_renderer()
  cam = ChaseCamera()
  cam.update(C3_RECT, 1.22, 1.0, 0.0, 0.0)
  poly = cam.ribbon(r._lane_lines[1].raw_points, 0.075, 0.0, 32, 100.0)
  assert poly.shape[1] == 2 and poly[:, 1].max() > C3_RECT.height * 0.95


def test_lane_color_states():
  from openpilot.selfdrive.ui.onroad import fsd_draw
  green = fsd_draw.lane_color(1, 0.9, True, False)
  white = fsd_draw.lane_color(1, 0.4, True, False)
  orange = fsd_draw.lane_color(2, 0.9, True, True)
  assert green.g > green.r and green.g > green.b
  assert white.r > 230 and white.g > 230 and white.b > 230
  assert orange.r == 255 and orange.b == 0
  assert fsd_draw.lane_blocked(_car_state(left_bs=True), 1)
  assert not fsd_draw.lane_blocked(_car_state(left_bs=True), 2)
  assert fsd_draw.lane_blocked(_car_state(right_code=1), 2)       # solid line
  assert not fsd_draw.lane_blocked(_car_state(right_code=10), 2)  # dashed line
  assert not fsd_draw.lane_blocked(_car_state(), 1)               # unclassified


@pytest.mark.parametrize("sizes_name,rect", [("C3_SIZES", C3_RECT), ("C4_SIZES", C4_RECT)])
@pytest.mark.parametrize("is_night", [True, False])
def test_scene_draws_close_stop_with_ego_car(monkeypatch, sizes_name, rect, is_night):
  from openpilot.selfdrive.ui.onroad import view_3d_scene, view_3d_draw
  polys, cars, wheels = [], [], []
  monkeypatch.setattr(view_3d_scene, "draw_polygon_solid", lambda pts, col: polys.append(pts))
  real_vehicle = view_3d_draw.draw_vehicle_rear
  monkeypatch.setattr(view_3d_draw, "draw_vehicle_rear",
                      lambda x, y, vw, vh, night, style='lead', braking=False, alpha=1.0:
                      (cars.append((style, y, vh)), real_vehicle(x, y, vw, vh, night, style, braking, alpha)))
  monkeypatch.setattr(view_3d_draw, "draw_steering_wheel", lambda *a, **k: wheels.append(a))

  scene = view_3d_scene.View3dScene(getattr(view_3d_scene, sizes_name))
  radar = SimpleNamespace(leadOne=_lead(3.8, v=-1.0), leadTwo=_lead(30.0, 0.3), leadsLeft=[_lead(12.0, 3.5)], leadsRight=[])
  sm = FakeSubMaster(carState=_car_state(v_ego=0.0, left_bs=True), radarState=radar)
  for i in range(40):
    cars.clear()
    wheels.clear()
    scene.draw(_scene_renderer(), sm, rect, is_night, now=i * 0.05)
  assert scene.camera.zoom > 0.95
  assert polys
  styles = [c[0] for c in cars]
  assert styles[-1] == 'ego' and 'lead' in styles and 'adj' in styles
  lead = [c for c in cars if c[0] == 'lead'][-1]
  assert lead[1] + lead[2] / 2 < rect.height  # close lead stays on screen
  # Steering wheel transitions naturally into the vehicle cockpit interior
  assert len(wheels) == 1
  _wheel_x, wheel_y, wheel_r = wheels[0][0], wheels[0][1], wheels[0][2]
  assert wheel_y < rect.height and wheel_r < rect.height * scene.sizes.wheel_radius_frac


def test_scene_cruise_shows_wheel_and_no_ego_car(monkeypatch):
  from openpilot.selfdrive.ui.onroad import view_3d_scene, view_3d_draw
  cars, wheels = [], []
  monkeypatch.setattr(view_3d_scene, "draw_polygon_solid", lambda pts, col: None)
  monkeypatch.setattr(view_3d_draw, "draw_vehicle_rear", lambda *a, style='lead', **k: cars.append(style))
  monkeypatch.setattr(view_3d_draw, "draw_steering_wheel", lambda *a, **k: wheels.append(a))
  scene = view_3d_scene.View3dScene(view_3d_scene.C3_SIZES)
  sm = FakeSubMaster(carState=_car_state(v_ego=20.0), radarState=SimpleNamespace(leadOne=_lead(40.0), leadTwo=None))
  scene.draw(_scene_renderer(), sm, C3_RECT, True, now=0.0)
  assert cars == ['lead'] and wheels


def test_scene_sharp_curve_widens_view(monkeypatch):
  from openpilot.selfdrive.ui.onroad import view_3d_scene
  monkeypatch.setattr(view_3d_scene, "draw_polygon_solid", lambda pts, col: None)
  scene = view_3d_scene.View3dScene(view_3d_scene.C3_SIZES)
  sm = FakeSubMaster(carState=_car_state(v_ego=8.0, steer=90.0), radarState=SimpleNamespace(leadOne=None, leadTwo=None))
  for i in range(40):
    scene.draw(_scene_renderer(curve_y=8.0), sm, C3_RECT, False, now=i * 0.05)
  assert scene.camera.zoom > 0.95 and scene.camera.yaw > 0.0


def test_both_renderers_route_to_scene(monkeypatch):
  from openpilot.selfdrive.ui.onroad import view_3d_scene
  from openpilot.selfdrive.ui.mici.onroad.model_renderer import ModelRenderer as MiciModelRenderer
  calls = []
  monkeypatch.setattr(view_3d_scene.View3dScene, "draw", lambda self, r, sm, rect, night, now=None: calls.append((self.sizes, night)))
  sm = FakeSubMaster(modelV2=object())
  c3 = ModelRenderer()
  c3._fsd_is_night = False
  c3._draw_fsd_visualization(sm)
  c4 = MiciModelRenderer()
  c4._draw_fsd_visualization(sm)
  assert calls == [(view_3d_scene.C3_SIZES, False), (view_3d_scene.C4_SIZES, True)]


def test_road_edges_and_outer_lanes_drawn(monkeypatch):
  from openpilot.selfdrive.ui.onroad import view_3d_scene, view_3d_draw

  low_std_col = view_3d_draw.road_edge_color(0.2, is_night=True)
  high_std_col = view_3d_draw.road_edge_color(1.8, is_night=True)
  assert low_std_col.r > low_std_col.b
  assert high_std_col.b > high_std_col.r

  drawn = []
  monkeypatch.setattr(view_3d_scene, "draw_polygon_solid", lambda pts, col: drawn.append((pts, col)))
  monkeypatch.setattr(view_3d_draw, "draw_steering_wheel", lambda *a, **k: None)

  scene = view_3d_scene.View3dScene(view_3d_scene.C3_SIZES)
  sm = FakeSubMaster(carState=_car_state(v_ego=15.0), radarState=SimpleNamespace(leadOne=None, leadTwo=None))
  r = _scene_renderer()
  r._lane_line_probs = [0.25, 0.9, 0.9, 0.25]
  r._road_edge_stds = [0.2, 2.5]  # Left valid (<2.0), Right filtered (>=2.0)

  scene.draw(r, sm, C3_RECT, True, now=0.0)
  # Check at least 1 road edge polygon matches low_std_col.r and both outer lanes (prob=0.25) were drawn
  red_edges = [c for _, c in drawn if c.r == low_std_col.r and c.b == low_std_col.b and c.g == low_std_col.g]
  assert len(red_edges) == 1
  # Outer lanes use adj colour (190, 205, 225) in night mode
  adj_lanes = [c for _, c in drawn if (c.r, c.g, c.b) == view_3d_draw.PALETTE[True]['lane_adj'][:3]]
  assert len(adj_lanes) == 2

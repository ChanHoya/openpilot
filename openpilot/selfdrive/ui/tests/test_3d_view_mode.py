from types import SimpleNamespace
from unittest.mock import MagicMock
import numpy as np
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
            "draw_rectangle_gradient_v", "draw_ellipse_lines", "draw_line_ex", "draw_texture_pro"):
    monkeypatch.setattr(rl, fn, lambda *args, **kwargs: None)
  monkeypatch.setattr("openpilot.selfdrive.ui.mici.onroad.model_renderer.draw_text_ui_style", lambda *args, **kwargs: None)
  monkeypatch.setattr("openpilot.selfdrive.ui.onroad.model_renderer.draw_text_ui_style", lambda *args, **kwargs: None)
  return gui_app


class FakeParams:
  def __init__(self, initial=False, unknown_keys=()):
    self.data = {"View3dMode": initial, "FsdViewMode": initial}
    self.unknown_keys = set(unknown_keys)

  def get_bool(self, key):
    if key in self.unknown_keys:
      raise KeyError(f"Unknown param key: {key}")
    return self.data.get(key, False)

  def put_bool(self, key, val):
    if key in self.unknown_keys:
      raise KeyError(f"Unknown param key: {key}")
    self.data[key] = bool(val)


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
  assert fake_params.get_bool("FsdViewMode") is True

  btn._handle_mouse_release(None)
  assert btn.view_3d_mode is False
  assert btn.fsd_mode is False
  assert fake_params.get_bool("View3dMode") is False
  assert fake_params.get_bool("FsdViewMode") is False


def test_view_3d_button_persists_ar_mode_after_1s_refresh(monkeypatch):
  # Simulate device running without recompiled View3dMode in Cython params_pyx.
  fake_params = FakeParams(initial=True, unknown_keys={"View3dMode"})
  monkeypatch.setattr("openpilot.selfdrive.ui.onroad.view_3d_button.Params", lambda: fake_params)
  now = [10.0]
  monkeypatch.setattr("openpilot.selfdrive.ui.onroad.view_3d_button.time.monotonic", lambda: now[0])

  btn = View3dButton(width=150, height=68)
  assert btn.view_3d_mode is True

  btn._handle_mouse_release(None)
  assert btn.view_3d_mode is False
  assert fake_params.get_bool("FsdViewMode") is False

  # After 1.5s _render triggers periodic param sync: AR view must NOT snap back to 3D view.
  now[0] = 11.5
  btn._render(rl.Rectangle(0, 0, 150, 68))
  assert btn.view_3d_mode is False
  assert fake_params.get_bool("FsdViewMode") is False


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
  calls = []
  renderer._draw_path_carrot = lambda sm: calls.append("path")
  renderer._draw_lane_lines_carrot = lambda sm: calls.append("lanes")
  renderer._draw_blind_spot_carrot = lambda sm: calls.append("blindspot")
  renderer._draw_radar_info_carrot = lambda sm: calls.append("radar")
  renderer._render_diagnostics = SimpleNamespace(
    start=lambda: None,
    finish=lambda: None,
    call=lambda name, callback, *args: callback(*args),
    values={}
  )

  # 1) In 3D view mode: all AR base elements (path, lanes, blindspot, radar) run consistently
  renderer.set_view_3d_mode(True)
  renderer._draw_carrot_overlays(object())
  assert calls == ["path", "lanes", "blindspot", "radar"]

  # 2) In AR view mode: same base pipeline runs
  calls.clear()
  renderer.set_view_3d_mode(False)
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
  from openpilot.selfdrive.ui.onroad.view_3d_camera import ChaseCamera
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
def test_windshield_camera_keeps_close_lead_visible(rect):
  from openpilot.selfdrive.ui.onroad.view_3d_camera import ChaseCamera
  cam = ChaseCamera()
  cam.update(rect, 1.22, 0.0, 0.0, 0.0)
  for d in (1.5, 3.8, 5.0):
    x, y, vw, vh = cam.car_box(d, 0.0, 1.22)
    assert rect.y < y - vh / 2 and y + vh / 2 < rect.y + rect.height
  _, lead_y, vw, lead_h = cam.car_box(3.8, 0.0, 1.22)
  assert vw < 0.25 * rect.width
  assert cam.ego_box() is None


def test_zoom_target_rules():
  from openpilot.selfdrive.ui.onroad.view_3d_camera import zoom_target
  assert zoom_target(20.0, None, 0.0) == 0.0
  assert zoom_target(20.0, 30.0, 0.5) == 0.0
  assert zoom_target(3.0, 4.0, 0.0) == 0.0
  assert zoom_target(0.0, 10.0, 0.0) == 0.0
  assert zoom_target(15.0, None, 6.5) == 0.0


def test_chase_camera_yaw_is_smoothed():
  from openpilot.selfdrive.ui.onroad.view_3d_camera import ChaseCamera
  cam = ChaseCamera()
  cam.update(C3_RECT, 1.22, 0.0, 0.15, 0.05)
  assert 0.0 < cam.yaw < 0.05
  for _ in range(60):
    cam.update(C3_RECT, 1.22, 0.0, 0.15, 0.05)
  assert cam.yaw > 0.14


def test_wide_ribbon_extends_beside_ego():
  from openpilot.selfdrive.ui.onroad.view_3d_camera import ChaseCamera
  r = _scene_renderer()
  cam = ChaseCamera()
  cam.update(C3_RECT, 1.22, 0.0, 0.0, 0.0)
  poly = cam.ribbon(r._lane_lines[1].raw_points, 0.075, 0.0, 32, 100.0)
  assert poly.shape[1] == 2 and poly[:, 1].max() > C3_RECT.height * 0.95


def test_vehicle_palette_white_lead_adj_and_black_ego():
  from openpilot.selfdrive.ui.onroad import view_3d_draw
  for is_night in (True, False):
    pal = view_3d_draw.PALETTE[is_night]
    for channel in range(3):
      assert pal['lead_top'][channel] >= 245
      assert pal['lead_bottom'][channel] >= 205
      assert pal['adj_top'][channel] >= 235
      assert pal['adj_bottom'][channel] >= 195
      assert pal['ego_top'][channel] <= 75
      assert pal['ego_bottom'][channel] <= 30


def test_lane_color_states():
  from openpilot.selfdrive.ui.onroad import view_3d_draw
  green = view_3d_draw.lane_color(1, 0.9, True, False)
  white = view_3d_draw.lane_color(1, 0.4, True, False)
  orange = view_3d_draw.lane_color(2, 0.9, True, True)
  assert green.g > green.r and green.g > green.b
  assert white.r > 230 and white.g > 230 and white.b > 230
  assert orange.r == 255 and orange.b == 0
  assert view_3d_draw.lane_blocked(_car_state(left_bs=True), 1)
  assert not view_3d_draw.lane_blocked(_car_state(left_bs=True), 2)
  assert view_3d_draw.lane_blocked(_car_state(right_code=1), 2)       # solid line
  assert not view_3d_draw.lane_blocked(_car_state(right_code=10), 2)  # dashed line
  assert not view_3d_draw.lane_blocked(_car_state(), 1)               # unclassified


@pytest.mark.parametrize("sizes_name,rect", [("C3_SIZES", C3_RECT), ("C4_SIZES", C4_RECT)])
@pytest.mark.parametrize("is_night", [True, False])
def test_scene_draws_close_stop_in_windshield_view(monkeypatch, sizes_name, rect, is_night):
  from openpilot.selfdrive.ui.onroad import view_3d_scene, view_3d_draw
  polys, cars, wheels, frames = [], [], [], []
  monkeypatch.setattr(view_3d_scene, "draw_polygon_solid", lambda pts, col: polys.append(pts))
  real_vehicle = view_3d_draw.draw_vehicle_rear
  monkeypatch.setattr(view_3d_draw, "draw_vehicle_rear",
                      lambda x, y, vw, vh, night, style='lead', braking=False, alpha=1.0:
                      (cars.append((style, y, vh)), real_vehicle(x, y, vw, vh, night, style, braking, alpha)))
  monkeypatch.setattr(view_3d_draw, "draw_steering_wheel", lambda *a, **k: wheels.append(a))
  monkeypatch.setattr(view_3d_draw, "draw_windshield_frame", lambda *a, **k: frames.append(a))

  scene = view_3d_scene.View3dScene(getattr(view_3d_scene, sizes_name))
  radar = SimpleNamespace(leadOne=_lead(3.8, v=-1.0), leadTwo=_lead(30.0, 0.3), leadsLeft=[_lead(12.0, 3.5)], leadsRight=[])
  sm = FakeSubMaster(carState=_car_state(v_ego=0.0, left_bs=True), radarState=radar)
  scene.draw(_scene_renderer(), sm, rect, is_night, now=0.0)

  assert polys
  styles = [c[0] for c in cars]
  assert 'lead' in styles and 'adj' in styles and 'ego' not in styles
  lead = [c for c in cars if c[0] == 'lead'][-1]
  assert lead[1] + lead[2] / 2 < rect.height * 0.85  # close lead stays cleanly inside windshield
  assert len(frames) == 1
  assert len(wheels) == 1
  wheel_x, wheel_y, wheel_r = wheels[0][0], wheels[0][1], wheels[0][2]
  assert wheel_x == pytest.approx(rect.x + rect.width * 0.50)
  assert wheel_y > rect.height and wheel_r == pytest.approx(rect.height * scene.sizes.wheel_radius_frac)


def test_scene_cruise_shows_center_wheel_and_no_ego_car(monkeypatch):
  from openpilot.selfdrive.ui.onroad import view_3d_scene, view_3d_draw
  cars, wheels = [], []
  monkeypatch.setattr(view_3d_scene, "draw_polygon_solid", lambda pts, col: None)
  monkeypatch.setattr(view_3d_draw, "draw_vehicle_rear", lambda *a, style='lead', **k: cars.append(style))
  monkeypatch.setattr(view_3d_draw, "draw_steering_wheel", lambda *a, **k: wheels.append(a))
  monkeypatch.setattr(view_3d_draw, "draw_windshield_frame", lambda *a, **k: None)
  scene = view_3d_scene.View3dScene(view_3d_scene.C3_SIZES)
  sm = FakeSubMaster(carState=_car_state(v_ego=20.0), radarState=SimpleNamespace(leadOne=_lead(40.0), leadTwo=None))
  scene.draw(_scene_renderer(), sm, C3_RECT, True, now=0.0)
  assert cars == ['lead'] and wheels
  assert wheels[0][0] == pytest.approx(C3_RECT.width * 0.50)


def test_scene_sharp_curve_turns_yaw_without_zoom_jump(monkeypatch):
  from openpilot.selfdrive.ui.onroad import view_3d_scene, view_3d_draw
  monkeypatch.setattr(view_3d_scene, "draw_polygon_solid", lambda pts, col: None)
  monkeypatch.setattr(view_3d_draw, "draw_steering_wheel", lambda *a, **k: None)
  monkeypatch.setattr(view_3d_draw, "draw_windshield_frame", lambda *a, **k: None)
  scene = view_3d_scene.View3dScene(view_3d_scene.C3_SIZES)
  sm = FakeSubMaster(carState=_car_state(v_ego=8.0, steer=90.0), radarState=SimpleNamespace(leadOne=None, leadTwo=None))
  for i in range(40):
    scene.draw(_scene_renderer(curve_y=8.0), sm, C3_RECT, False, now=i * 0.05)
  assert scene.camera.zoom == 0.0 and scene.camera.yaw > 0.0


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


def test_road_four_layers_path_tires_ego_lanes_and_edges(monkeypatch):
  from openpilot.selfdrive.ui.onroad import view_3d_scene, view_3d_draw

  low_std_col = view_3d_draw.road_edge_color(0.2, is_night=True)
  high_std_col = view_3d_draw.road_edge_color(1.8, is_night=True)
  assert low_std_col.r > low_std_col.b
  assert high_std_col.b > high_std_col.r

  drawn = []
  monkeypatch.setattr(view_3d_scene, "draw_polygon_solid", lambda pts, col: drawn.append((pts, col)))
  monkeypatch.setattr(view_3d_draw, "draw_steering_wheel", lambda *a, **k: None)
  monkeypatch.setattr(view_3d_draw, "draw_windshield_frame", lambda *a, **k: None)

  scene = view_3d_scene.View3dScene(view_3d_scene.C3_SIZES)
  sm = FakeSubMaster(carState=_car_state(v_ego=15.0), radarState=SimpleNamespace(leadOne=None, leadTwo=None))
  r = _scene_renderer()
  r._lane_line_probs = [0.25, 0.9, 0.9, 0.25]
  r._road_edge_stds = [0.2, 2.5]  # Left valid (<2.0), Right filtered (>=2.0)

  scene.draw(r, sm, C3_RECT, True, now=0.0)
  # Exact 4-layer budget: 1 path + 2 tire tracks + 2 ego lanes + 1 valid road edge = 6 polygons
  assert len(drawn) == 6
  red_edges = [c for _, c in drawn if c.r == low_std_col.r and c.b == low_std_col.b and c.g == low_std_col.g]
  assert len(red_edges) == 1
  # Outer adjacent lanes [0] and [3] must NOT be drawn
  adj_lanes = [c for _, c in drawn if (c.r, c.g, c.b) == view_3d_draw.PALETTE[True]['lane_adj'][:3]]
  assert len(adj_lanes) == 0


def test_stationary_radar_and_path_noise_stabilized(monkeypatch):
  import numpy as np
  from openpilot.selfdrive.ui.onroad import view_3d_scene, view_3d_draw

  polys, cars, chips = [], [], []
  monkeypatch.setattr(view_3d_scene, "draw_polygon_solid", lambda pts, col: polys.append(pts))
  monkeypatch.setattr(view_3d_draw, "draw_vehicle_rear",
                      lambda x, y, vw, vh, night, style='lead', braking=False, alpha=1.0:
                      cars.append((x, y, vw, vh)))
  monkeypatch.setattr(view_3d_draw, "draw_metric_chip", lambda x, top_y, text, *a, **k: chips.append(text))
  monkeypatch.setattr(view_3d_draw, "draw_steering_wheel", lambda *a, **k: None)
  monkeypatch.setattr(view_3d_draw, "draw_windshield_frame", lambda *a, **k: None)

  scene = view_3d_scene.View3dScene(view_3d_scene.C3_SIZES)
  r = _scene_renderer()
  boxes = []
  path_tops = []
  for i in range(20):
    sign = 1.0 if i % 2 else -1.0
    # Stopped path fluctuates wildly in length (3.6m..5.4m) and z (+-0.08m)
    xs = np.linspace(0.0, 4.5 + 0.9 * sign, 33)
    zs = (0.08 * sign) * (xs / xs[-1])
    r._path.raw_points = np.stack((xs, np.zeros_like(xs), zs), axis=1).astype(np.float32)

    # Stopped lead jittering +-0.25m in dRel and +-0.10m in yRel
    radar = SimpleNamespace(leadOne=_lead(7.0 + 0.25 * sign, y=0.10 * sign, v=0.05 * sign), leadTwo=None)
    sm = FakeSubMaster(carState=_car_state(v_ego=0.0), radarState=radar)

    polys.clear()
    cars.clear()
    chips.clear()
    scene.draw(r, sm, C3_RECT, True, now=i * 0.05)

    assert len(cars) == 1
    assert chips == ["L1 7m"]
    boxes.append(cars[0])
    # First polygon is the driving path ribbon; its minimum screen y is the far end tip.
    path_tops.append(float(polys[0][:, 1].min()))

  xs_box = [b[0] for b in boxes]
  ys_box = [b[1] for b in boxes]
  ws_box = [b[2] for b in boxes]
  hs_box = [b[3] for b in boxes]
  assert max(xs_box) - min(xs_box) < 1.5
  assert max(ys_box) - min(ys_box) < 1.5
  assert max(ws_box) - min(ws_box) < 1.5
  assert max(hs_box) - min(hs_box) < 1.5
  assert max(path_tops) - min(path_tops) < 1.5


def test_dynamic_horizon_responsive_to_hill_slope(monkeypatch):
  from openpilot.selfdrive.ui.onroad import view_3d_scene, view_3d_draw
  monkeypatch.setattr(view_3d_scene, "draw_polygon_solid", lambda pts, col: None)
  monkeypatch.setattr(view_3d_draw, "draw_vehicle_rear", lambda *a, **k: None)
  monkeypatch.setattr(view_3d_draw, "draw_steering_wheel", lambda *a, **k: None)
  monkeypatch.setattr(view_3d_draw, "draw_windshield_frame", lambda *a, **k: None)

  scene = view_3d_scene.View3dScene(view_3d_scene.C3_SIZES)
  sm = FakeSubMaster(carState=_car_state(v_ego=15.0), radarState=SimpleNamespace(leadOne=None, leadTwo=None))

  # 1) Flat road: horizon should stabilize at middle 0.50
  r_flat = _scene_renderer()
  for i in range(25):
    scene.draw(r_flat, sm, C3_RECT, False, now=i * 0.05)
  assert 0.495 < scene.horizon_frac < 0.505
  flat_y = scene.horizon_y
  assert flat_y == pytest.approx(C3_RECT.height * 0.50, abs=5.0)

  # 2) Uphill road (z decreases): horizon moves UP (< 0.50, lower y coordinate)
  r_uphill = _scene_renderer()
  xs = r_uphill._path.raw_points[:, 0]
  # 5% uphill: z = -0.05 * x
  r_uphill._path.raw_points[:, 2] = -0.05 * xs
  for i in range(25):
    scene.draw(r_uphill, sm, C3_RECT, False, now=1.5 + i * 0.05)
  assert scene.horizon_frac < 0.49
  assert scene.horizon_y < flat_y - 15.0

  # 3) Downhill road (z increases): horizon moves DOWN (> 0.50, higher y coordinate)
  r_downhill = _scene_renderer()
  r_downhill._path.raw_points[:, 2] = 0.05 * xs
  for i in range(35):
    scene.draw(r_downhill, sm, C3_RECT, False, now=3.0 + i * 0.05)
  assert scene.horizon_frac > 0.51
  assert scene.horizon_y > flat_y + 15.0


def test_lead_vehicle_scaled_and_brake_lights(monkeypatch):
  from openpilot.selfdrive.ui.onroad import view_3d_scene, view_3d_draw

  cars = []
  monkeypatch.setattr(view_3d_scene, "draw_polygon_solid", lambda pts, col: None)
  monkeypatch.setattr(view_3d_draw, "draw_vehicle_rear",
                      lambda x, y, vw, vh, night, style='lead', braking=False, alpha=1.0:
                      cars.append((style, vw, braking)))
  monkeypatch.setattr(view_3d_draw, "draw_metric_chip", lambda *a, **k: None)
  monkeypatch.setattr(view_3d_draw, "draw_steering_wheel", lambda *a, **k: None)
  monkeypatch.setattr(view_3d_draw, "draw_windshield_frame", lambda *a, **k: None)

  scene = view_3d_scene.View3dScene(view_3d_scene.C3_SIZES)
  r = _scene_renderer()

  # 1) Lead decelerating with aLead = -1.2 m/s^2 -> brake lights MUST be True
  lead_braking = SimpleNamespace(status=True, dRel=15.0, yRel=0.0, vRel=-1.0, aLead=-1.2)
  sm_braking = FakeSubMaster(carState=_car_state(v_ego=15.0),
                             radarState=SimpleNamespace(leadOne=lead_braking, leadTwo=None))
  cars.clear()
  scene.draw(r, sm_braking, C3_RECT, True, now=0.0)
  assert len(cars) == 1
  style, vw, braking = cars[0]
  assert style == 'lead' and braking is True
  # Verify 2x scaling: at 15m, standard 1.0x width is ~123px, 2.0x is ~246px
  assert vw > 200.0

  # 2) Lead cruising with aLead = 0.0 m/s^2 while ego slightly approaches (vRel = -0.3) -> brake lights MUST be False
  lead_cruising = SimpleNamespace(status=True, dRel=15.0, yRel=0.0, vRel=-0.3, aLead=0.0)
  sm_cruising = FakeSubMaster(carState=_car_state(v_ego=15.0),
                              radarState=SimpleNamespace(leadOne=lead_cruising, leadTwo=None))
  cars.clear()
  scene.draw(r, sm_cruising, C3_RECT, True, now=0.1)
  assert len(cars) == 1
  _, _, braking_cruising = cars[0]
  assert braking_cruising is False


def test_traffic_light_badge_on_stopping(monkeypatch):
  from openpilot.selfdrive.ui.onroad import view_3d_scene, view_3d_draw

  rects_drawn = []
  monkeypatch.setattr(view_3d_scene, "draw_polygon_solid", lambda pts, col: None)
  monkeypatch.setattr(view_3d_draw, "draw_vehicle_rear", lambda *a, **k: None)
  monkeypatch.setattr(view_3d_draw, "draw_steering_wheel", lambda *a, **k: None)
  monkeypatch.setattr(view_3d_draw, "draw_windshield_frame", lambda *a, **k: None)
  import pyray as rl
  monkeypatch.setattr(rl, "draw_rectangle_rounded", lambda rec, *a, **k: rects_drawn.append(rec))

  scene = view_3d_scene.View3dScene(view_3d_scene.C3_SIZES)
  r = _scene_renderer()

  # 1) Normal cruising: no traffic signal -> badge alpha remains 0
  sm_cruise = FakeSubMaster(carState=_car_state(v_ego=15.0),
                            radarState=SimpleNamespace(leadOne=None, leadTwo=None),
                            longitudinalPlan=SimpleNamespace(trafficState=0, xState='e2eCruise'))
  rects_drawn.clear()
  scene.draw(r, sm_cruise, C3_RECT, True, now=0.0)
  assert scene._traffic_light_alpha == 0.0

  # 2) Stopping at red light (trafficState = 1, xState = 'e2eStop') -> badge fades in
  sm_stop = FakeSubMaster(carState=_car_state(v_ego=5.0),
                          radarState=SimpleNamespace(leadOne=None, leadTwo=None),
                          longitudinalPlan=SimpleNamespace(trafficState=1, xState='e2eStop'))
  for i in range(15):
    rects_drawn.clear()
    scene.draw(r, sm_stop, C3_RECT, True, now=0.1 + i * 0.05)
  assert scene._traffic_light_alpha > 0.8
  # Badge rectangle drawn in upper portion of screen
  badge = [rec for rec in rects_drawn if rec.width <= 250.0 and rec.height == 50.0]
  assert len(badge) >= 1
  assert badge[0].y < C3_RECT.height * 0.25


def test_draw_vehicle_rear_uses_lead_car_png_texture_and_ground_glow(monkeypatch):
  from openpilot.selfdrive.ui.onroad import view_3d_draw
  tex_calls = []
  ellipses_drawn = []
  fake_tex = SimpleNamespace(id=42, width=612, height=433)
  monkeypatch.setattr(gui_app, "texture", lambda path, *a, **k: fake_tex if path == "images/lead_car.png" else MagicMock())
  monkeypatch.setattr(rl, "draw_texture_pro", lambda tex, src, dst, origin, rot, tint: tex_calls.append((tex, src, dst, tint)))
  monkeypatch.setattr(rl, "draw_ellipse", lambda cx, cy, rx, ry, col: ellipses_drawn.append((cx, cy, rx, ry, col)))

  vw, vh = 200.0, 140.0
  cx, cy = 500.0, 400.0
  view_3d_draw.draw_vehicle_rear(cx, cy, vw, vh, is_night=True, style='lead', braking=True, alpha=1.0)

  assert len(tex_calls) == 1
  tex, src, dst, tint = tex_calls[0]
  assert tex is fake_tex
  assert src.width == pytest.approx(612.0) and src.height == pytest.approx(433.0)
  expected_h = vw * (433.0 / 612.0)
  assert dst.width == pytest.approx(vw)
  assert dst.height == pytest.approx(expected_h)
  assert dst.x == pytest.approx(cx - vw * 0.5)
  assert dst.y == pytest.approx(cy - expected_h * 0.5)
  # Ground glow ellipses (16 smooth gradient layers) + braking bloom ellipses (4 layers: CHMSL + L/R/center lightbar)
  assert len(ellipses_drawn) == 20

  # Verify lead_two uses translucent cyan tint without red brake bloom when braking=False
  tex_calls.clear()
  ellipses_drawn.clear()
  view_3d_draw.draw_vehicle_rear(cx, cy, vw, vh, is_night=False, style='lead_two', braking=False, alpha=0.8)
  assert len(tex_calls) == 1
  _, _, _, l2_tint = tex_calls[0]
  assert l2_tint.r == 150 and l2_tint.g == 230 and l2_tint.b == 255
  assert l2_tint.a == int(215 * 0.8)
  assert len(ellipses_drawn) == 16
  # Outer ring fades gently into cyan core without harsh steps
  assert ellipses_drawn[0][4].a < ellipses_drawn[-1][4].a


def test_curved_road_lead_vehicle_aligns_with_path_center(monkeypatch):
  import numpy as np
  from openpilot.selfdrive.ui.onroad import view_3d_scene, view_3d_draw
  cars_drawn = []
  monkeypatch.setattr(view_3d_scene, "draw_polygon_solid", lambda pts, col: None)
  monkeypatch.setattr(view_3d_draw, "draw_vehicle_rear", lambda x, y, vw, vh, *a, **k: cars_drawn.append((x, y, vw, vh)))
  monkeypatch.setattr(view_3d_draw, "draw_steering_wheel", lambda *a, **k: None)
  monkeypatch.setattr(view_3d_draw, "draw_windshield_frame", lambda *a, **k: None)

  scene = view_3d_scene.View3dScene(view_3d_scene.C3_SIZES)
  # Sharp right curve renderer: path bends rightwards as distance increases (3.5m at 30m)
  r = _scene_renderer(curve_y=3.5)

  # Lead at 30m with raw radar yRel slightly off (e.g. 0.5m), path lateral is ~3.5m
  lead = SimpleNamespace(status=True, dRel=30.0, yRel=0.5, vRel=0.0, aLead=0.0)
  sm = FakeSubMaster(carState=_car_state(v_ego=20.0), radarState=SimpleNamespace(leadOne=lead, leadTwo=None))

  scene.draw(r, sm, C3_RECT, True, now=0.0)
  assert len(cars_drawn) == 1
  lead_screen_x = cars_drawn[0][0]
  # Path lateral at 30m is 3.5m (left curve in device frame: +y is left in screen)
  path_lat_30 = view_3d_scene._path_lateral(r._path.raw_points, 30.0)
  ground_z = scene._smooth_ground_z
  z_30 = view_3d_scene._path_z(r._path.raw_points, 30.0) + ground_z
  expected_pt = scene.camera.project_point(30.0, -path_lat_30, z_30)
  assert expected_pt is not None
  # Lead car screen X must match the path centerline projection exactly
  assert lead_screen_x == pytest.approx(expected_pt[0], abs=1.0)


def test_far_lead_vehicle_distance_scaling():
  from openpilot.selfdrive.ui.onroad.view_3d_scene import View3dScene, C3_SIZES
  scene = View3dScene(C3_SIZES)
  cam = scene.camera
  cam.update(C3_RECT, 1.22, 0.0, 0.0, 0.0)

  # Check car_box with distance boost: at 20m boost is 1.0, at 50m boost is > 2.0
  box_20 = cam.car_box(20.0, 0.0, 1.22, scale_mult=2.0 * 1.0)
  # At 50m, dist_boost = 1.0 + min(1.25, 30/24 * 1.05) = 2.25
  dist_boost_50 = 1.0 + min(1.25, (50.0 - 20.0) / 24.0 * 1.05)
  assert dist_boost_50 >= 2.0
  box_50_boosted = cam.car_box(50.0, 0.0, 1.22, scale_mult=2.0 * dist_boost_50)
  box_50_normal = cam.car_box(50.0, 0.0, 1.22, scale_mult=2.0 * 1.0)

  assert box_50_boosted[2] == pytest.approx(box_50_normal[2] * dist_boost_50, rel=1e-2)
  assert box_50_boosted[2] > box_50_normal[2] * 2.0


def test_road_speed_bump_and_traffic_stop_markers(monkeypatch):
  from openpilot.selfdrive.ui.onroad import view_3d_scene, view_3d_draw
  polys_drawn = []
  bump_markers = []
  stop_markers = []

  monkeypatch.setattr(view_3d_scene, "draw_polygon_solid", lambda pts, col: polys_drawn.append((pts, col)))
  monkeypatch.setattr(view_3d_draw, "draw_vehicle_rear", lambda *a, **k: None)
  monkeypatch.setattr(view_3d_draw, "draw_steering_wheel", lambda *a, **k: None)
  monkeypatch.setattr(view_3d_draw, "draw_windshield_frame", lambda *a, **k: None)
  monkeypatch.setattr(view_3d_draw, "draw_speed_bump_marker", lambda x, y, d, night: bump_markers.append((x, y, d)))
  monkeypatch.setattr(view_3d_draw, "draw_traffic_stop_marker", lambda x, y, d, night: stop_markers.append((x, y, d)))

  scene = view_3d_scene.View3dScene(view_3d_scene.C3_SIZES)
  r = _scene_renderer()

  # 1) Speed bump active (speedBumpDistance = 35.0m)
  cs = _car_state(v_ego=15.0)
  cs.speedBumpDistance = 35.0
  sm = FakeSubMaster(carState=cs, radarState=SimpleNamespace(leadOne=None, leadTwo=None))
  scene.draw(r, sm, C3_RECT, True, now=0.0)

  assert len(bump_markers) == 1
  assert bump_markers[0][2] == 35.0
  assert len(polys_drawn) >= 1  # 3D speed bump hump polygon drawn

  # 2) Traffic stopping active (trafficState = 1)
  polys_drawn.clear()
  bump_markers.clear()
  cs.speedBumpDistance = 0.0
  sm_stop = FakeSubMaster(carState=cs, radarState=SimpleNamespace(leadOne=None, leadTwo=None),
                          longitudinalPlan=SimpleNamespace(trafficState=1, xState='e2eStop', trafficStopModelLeadOffset=12.0))
  scene.draw(r, sm_stop, C3_RECT, True, now=0.1)

  assert len(stop_markers) == 1
  assert stop_markers[0][2] == 12.0
  assert len(polys_drawn) >= 1  # 3D stop line ribbon polygon drawn


def test_path_end_lead_box_replaced_by_sleek_sedan_in_view_3d_mode(monkeypatch):
  from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer
  from openpilot.selfdrive.ui.onroad import view_3d_draw

  cars_drawn = []
  boxes_drawn = []
  monkeypatch.setattr(view_3d_draw, "draw_sleek_sedan_rear", lambda x, y, w, h, *a, **k: cars_drawn.append((x, y, w, h)))

  renderer = ModelRenderer()
  monkeypatch.setattr(renderer, "_draw_rect_fill_outline_carrot", lambda x, y, w, h, *a, **k: boxes_drawn.append((x, y, w, h)))
  renderer._carrot_lead_status = True
  renderer._carrot_radar_track_id = 1
  renderer._carrot_path_x = 1000
  renderer._carrot_path_y = 600
  renderer._carrot_path_width_px = 300
  renderer._carrot_lead_two_status = 0
  renderer._carrot_tf_distance = 0.0

  # 1) In 3D view mode: sleek sedan is drawn with distance-proportional scale & distance box below tires
  renderer._rect = rl.Rectangle(0.0, 0.0, 2160.0, 1080.0)
  renderer._carrot_radar_dist = 10.0
  text_boxes = []
  monkeypatch.setattr(renderer, "_draw_text_box_carrot", lambda x, y, *a, **k: text_boxes.append((x, y)))
  renderer.set_view_3d_mode(True)
  renderer._draw_path_end_overlay_carrot()

  assert len(cars_drawn) == 1
  assert len(boxes_drawn) == 0
  cx, cy, cw, ch = cars_drawn[0]
  expected_scale = renderer._lead_car_distance_scale(10.0)
  assert cw == pytest.approx(300 * expected_scale)
  assert ch == pytest.approx(cw * (433.0 / 612.0))
  assert cx == pytest.approx(1000)
  # Distance box must sit below the tires (cy + ch * 0.50), spread wider than 80px
  assert len(text_boxes) == 1
  assert text_boxes[0][1] > cy + ch * 0.50 - 1.0
  assert abs(1000 - text_boxes[0][0]) >= 135

  # 1b) Standstill close lead lowers car monotonically so bumper half sits near bottom edge
  cars_drawn.clear()
  renderer._carrot_path_y = 1040
  renderer._carrot_path_width_px = 900
  renderer._carrot_radar_dist = 2.0
  renderer._draw_path_end_overlay_carrot()
  _, cy_close, _, ch_close = cars_drawn[0]
  assert cy_close + ch_close * 0.50 > 1080.0 + ch_close * 0.15

  # 1c) Approaching from 8m -> 5m -> 2m strictly moves car downward (monotonic cy_bottom)
  bottoms = []
  for d_m, py_m in ((8.0, 820), (5.0, 910), (2.0, 1020)):
    cars_drawn.clear()
    renderer._carrot_path_y = py_m
    renderer._carrot_path_width_px = 500
    renderer._carrot_radar_dist = d_m
    renderer._draw_path_end_overlay_carrot()
    _, cy_d, _, ch_d = cars_drawn[0]
    bottoms.append(cy_d + ch_d * 0.50)
  assert bottoms[0] < bottoms[1] < bottoms[2]

  renderer._carrot_path_y = 600
  renderer._carrot_path_width_px = 300

  # 2) In AR view mode: traditional bounding box is drawn
  cars_drawn.clear()
  boxes_drawn.clear()
  renderer.set_view_3d_mode(False)
  renderer._draw_path_end_overlay_carrot()

  assert len(cars_drawn) == 0
  assert len(boxes_drawn) == 1
  bx, by, bw, bh = boxes_drawn[0]
  assert bw == pytest.approx(300 + 20)

  # 3) _draw_carrot_overlays draws path_labels (lead car) AFTER lanes so lanes stay behind car
  order = []
  monkeypatch.setattr(renderer, "_draw_path_carrot", lambda sm: order.append("path"))
  monkeypatch.setattr(renderer, "_draw_lane_lines_carrot", lambda sm: order.append("lanes"))
  monkeypatch.setattr(renderer, "_draw_road_markers_carrot", lambda sm: order.append("markers"))
  monkeypatch.setattr(renderer, "_draw_path_end_overlay_carrot", lambda: order.append("path_labels"))
  monkeypatch.setattr(renderer, "_draw_blind_spot_carrot", lambda sm: order.append("blindspot"))
  monkeypatch.setattr(renderer, "_draw_radar_info_carrot", lambda sm: order.append("radar"))
  renderer._draw_carrot_overlays({})
  assert order.index("lanes") < order.index("path_labels")


def test_close_distance_under_1m_lead_car_handling():
  from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer
  renderer = ModelRenderer()
  renderer._rect = rl.Rectangle(0.0, 0.0, 1920.0, 1080.0)
  renderer._clip_region = renderer._rect
  renderer._car_space_transform = np.array([
    [960.0, 1000.0, 0.0],
    [540.0, 0.0, 800.0],
    [1.0, 0.0, 0.0],
  ], dtype=np.float32)

  # Path array
  xs = np.linspace(0.0, 50.0, 100, dtype=np.float32)
  ys = np.zeros(100, dtype=np.float32)
  zs = np.zeros(100, dtype=np.float32)
  renderer._path.raw_points = np.stack([xs, ys, zs], axis=1)

  lead_one = SimpleNamespace(status=True, dRel=0.8, yRel=0.0, vRel=0.0, radar=True, radarTrackId=1, aLead=0.0)
  radar_state = SimpleNamespace(leadOne=lead_one, leadTwo=None)
  sm = {
    'modelV2': SimpleNamespace(leadsV3=[]),
    'radarState': radar_state,
    'longitudinalPlan': SimpleNamespace(xState=0, trafficState=0, tFollow=1.5, desiredDistance=10.0, longitudinalPlanSource=SimpleNamespace(raw=0)),
    'selfdriveState': SimpleNamespace(enabled=True),
    'carState': SimpleNamespace(vEgo=0.0, brakeHoldActive=False, softHoldActive=False, carrotCruise=False),
  }
  sm_valid = {'radarState': True, 'modelV2': True, 'carState': True}
  class MockSM:
    def __init__(self, d, v):
      self.d = d
      self.valid = v
    def __getitem__(self, k):
      return self.d[k]

  sm_obj = MockSM(sm, sm_valid)

  # Run multiple frames to allow EMA filter to settle
  for _ in range(10):
    renderer._update_path_end_carrot(sm_obj)
  assert renderer._carrot_lead_status is True
  assert renderer._carrot_path_width_px > 400  # Wide close car
  assert renderer._carrot_path_y > 400.0       # Positioned low on screen, does not disappear


def test_curve_aligns_lead_vehicle_inside_path_trajectory():
  from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer
  renderer = ModelRenderer()
  renderer._rect = rl.Rectangle(0.0, 0.0, 1920.0, 1080.0)
  renderer._clip_region = renderer._rect
  # Real openpilot calib_transform (view_frame_from_device_frame): +Y_calib input (right) shifts point to screen right (> 960)
  renderer._car_space_transform = np.array([
    [960.0, 1000.0, 0.0],
    [540.0, 0.0, 800.0],
    [1.0, 0.0, 0.0],
  ], dtype=np.float32)

  # Sharp right curve: in calib frame (+Y = right), at x=20m path y=+3.0m; in radarState (+Y = left), yRel=-3.0m
  xs = np.linspace(0.0, 50.0, 100, dtype=np.float32)
  ys = 3.0 * (xs / 20.0) ** 2  # curving right (+Y in calib frame)
  zs = np.zeros(100, dtype=np.float32)
  renderer._path.raw_points = np.stack([xs, ys, zs], axis=1)

  lead_one = SimpleNamespace(status=True, dRel=20.0, yRel=-3.0, vRel=0.0, radar=True, radarTrackId=1, aLead=0.0)
  radar_state = SimpleNamespace(leadOne=lead_one, leadTwo=None)
  sm = {
    'modelV2': SimpleNamespace(leadsV3=[]),
    'radarState': radar_state,
    'longitudinalPlan': SimpleNamespace(xState=0, trafficState=0, tFollow=1.5, desiredDistance=10.0, longitudinalPlanSource=SimpleNamespace(raw=0)),
    'selfdriveState': SimpleNamespace(enabled=True),
    'carState': SimpleNamespace(vEgo=10.0, brakeHoldActive=False, softHoldActive=False, carrotCruise=False),
  }
  sm_valid = {'radarState': True, 'modelV2': True, 'carState': True}
  class MockSM:
    def __init__(self, d, v):
      self.d = d
      self.valid = v
    def __getitem__(self, k):
      return self.d[k]

  sm_obj = MockSM(sm, sm_valid)

  renderer._carrot_path_fx = 960.0  # start at center
  renderer._update_path_end_carrot(sm_obj)
  # In a right turn, screen x must shift to the right (> 960.0) following the curved path
  assert renderer._carrot_path_x > 960.0


def test_standstill_lead_vehicle_uses_sensor_y_for_exact_lane_alignment():
  from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer
  renderer = ModelRenderer()
  renderer._rect = rl.Rectangle(0.0, 0.0, 1920.0, 1080.0)
  renderer._clip_region = renderer._rect
  renderer._car_space_transform = np.array([
    [960.0, 1000.0, 0.0],
    [540.0, 0.0, 800.0],
    [1.0, 0.0, 0.0],
  ], dtype=np.float32)

  # At standstill, path is straight ahead (y=0) because vehicle is stopped
  xs = np.linspace(0.0, 50.0, 100, dtype=np.float32)
  ys = np.zeros(100, dtype=np.float32)
  zs = np.zeros(100, dtype=np.float32)
  renderer._path.raw_points = np.stack([xs, ys, zs], axis=1)

  # But lead vehicle is stopped slightly to the left (yRel=+0.8m, left-positive in radarState)
  lead_one = SimpleNamespace(status=True, dRel=5.0, yRel=0.8, vRel=0.0, vLead=0.0, radar=True, radarTrackId=1, aLead=0.0)
  radar_state = SimpleNamespace(leadOne=lead_one, leadTwo=None)
  sm = {
    'modelV2': SimpleNamespace(leadsV3=[]),
    'radarState': radar_state,
    'longitudinalPlan': SimpleNamespace(xState=0, trafficState=0, tFollow=1.5, desiredDistance=10.0, longitudinalPlanSource=SimpleNamespace(raw=0)),
    'selfdriveState': SimpleNamespace(enabled=True),
    'carState': SimpleNamespace(vEgo=0.0, brakeHoldActive=False, softHoldActive=False, carrotCruise=False),
  }
  sm_valid = {'radarState': True, 'modelV2': True, 'carState': True}
  class MockSM:
    def __init__(self, d, v):
      self.d = d
      self.valid = v
    def __getitem__(self, k):
      return self.d[k]

  sm_obj = MockSM(sm, sm_valid)

  renderer._carrot_path_fx = 960.0
  renderer._update_path_end_carrot(sm_obj)
  # At standstill, lead vehicle must shift left (< 960.0) matching sensor_y (-0.8m input), not staying stuck at center 960.0
  assert renderer._carrot_path_x < 960.0


def test_strict_lead_brake_light_logic(monkeypatch):
  from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer
  from openpilot.selfdrive.ui.onroad import view_3d_draw

  braking_states = []
  monkeypatch.setattr(view_3d_draw, "draw_sleek_sedan_rear", lambda *a, **k: braking_states.append(k.get('braking', False)))

  renderer = ModelRenderer()
  renderer.set_view_3d_mode(True)
  renderer._carrot_lead_status = True
  renderer._carrot_path_x = 960
  renderer._carrot_path_y = 540
  renderer._carrot_path_width_px = 200

  # Case 1: Standstill (lead vehicle stopped at red light) -> Brake light MUST be ON
  renderer._carrot_lead_v_lead = 0.0
  renderer._carrot_v_ego = 0.0
  renderer._carrot_lead_v_rel = 0.0
  renderer._carrot_lead_a_rel = 0.0
  renderer._draw_path_end_overlay_carrot()
  assert braking_states[-1] is True

  # Case 2: Ego car accelerates (e.g. driver accelerates, v_rel becomes negative, but a_lead is 0) -> Brake light MUST be OFF
  braking_states.clear()
  renderer._carrot_lead_v_lead = 15.0
  renderer._carrot_v_ego = 18.0
  renderer._carrot_lead_v_rel = -3.0
  renderer._carrot_lead_a_rel = 0.0  # Lead car is maintaining speed, not braking
  renderer._draw_path_end_overlay_carrot()
  assert braking_states[-1] is False

  # Case 3: Lead car actually brakes (a_lead = -1.2 m/s^2) -> Brake light MUST be ON
  braking_states.clear()
  renderer._carrot_lead_v_lead = 12.0
  renderer._carrot_v_ego = 15.0
  renderer._carrot_lead_v_rel = -1.0
  renderer._carrot_lead_a_rel = -1.2  # Lead car actively decelerating
  renderer._draw_path_end_overlay_carrot()
  assert braking_states[-1] is True


def test_dual_lead_vehicles_rendering(monkeypatch):
  from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer
  from openpilot.selfdrive.ui.onroad import view_3d_draw

  cars_drawn = []
  monkeypatch.setattr(view_3d_draw, "draw_sleek_sedan_rear", lambda x, y, w, h, *a, **k: cars_drawn.append((x, y, w, h, k.get('alpha', 1.0))))

  renderer = ModelRenderer()
  renderer.set_view_3d_mode(True)
  renderer._carrot_lead_status = True
  renderer._carrot_path_x = 960
  renderer._carrot_path_y = 650
  renderer._carrot_path_width_px = 300
  renderer._carrot_radar_dist = 10.0

  # leadTwo ahead at distance 25m
  renderer._carrot_lead_two_status = 1
  renderer._carrot_lead_two_dist = 25.0
  renderer._carrot_lead_two_xl = 900
  renderer._carrot_lead_two_xr = 1020
  renderer._carrot_lead_two_y = 520

  renderer._draw_path_end_overlay_carrot()
  # Both lead cars should be rendered: leadTwo (alpha 0.80) and leadOne (alpha 1.0)
  assert len(cars_drawn) == 2
  car2, car1 = cars_drawn[0], cars_drawn[1]
  assert car2[4] == pytest.approx(0.80)  # leadTwo
  assert car1[4] == pytest.approx(1.0)   # leadOne
  # leadTwo is further away (smaller width, higher Y)
  assert car2[2] < car1[2]
  assert car2[1] < car1[1]


def test_lane_dash_animation_offset_smooth_continuous():
  from openpilot.selfdrive.ui.road_markings import lane_dash_segments, LANE_DASH_LENGTH_M, LANE_DASH_GAP_M
  line = np.array([[float(i), 0.0, 0.0] for i in range(50)], dtype=np.float32)

  # offset = 0.0
  segs_0 = lane_dash_segments(line, 40.0, offset=0.0)
  assert len(segs_0) > 0
  start_0 = segs_0[0][0, 0]

  # offset = -2.0 (car moved 2m forward, dashes move 2m backward)
  segs_2 = lane_dash_segments(line, 40.0, offset=-2.0)
  assert len(segs_2) > 0
  # First dash starts at 0.0, but ends earlier than when offset=0
  assert segs_2[0][-1, 0] == pytest.approx(LANE_DASH_LENGTH_M - 2.0)


def test_draw_blindspot_radar_wave_wifi_arc_and_triangle(monkeypatch):
  from openpilot.selfdrive.ui.onroad import view_3d_draw
  triangles_drawn = []
  circles_drawn = []
  rects_drawn = []
  monkeypatch.setattr(rl, "draw_triangle", lambda *a, **k: triangles_drawn.append(a))
  monkeypatch.setattr(rl, "draw_triangle_lines", lambda *a, **k: triangles_drawn.append(a))
  monkeypatch.setattr(rl, "draw_circle", lambda *a, **k: circles_drawn.append(a))
  monkeypatch.setattr(rl, "draw_rectangle_rounded", lambda rec, *a, **k: rects_drawn.append(rec))

  # Test right-side blindspot radar wave (side_sign = 1.0)
  cx, cy = 1200.0, 750.0
  view_3d_draw.draw_blindspot_radar_wave(cx, cy, side_sign=1.0, scale=1.0, alpha=1.0, show_triangle=True)

  # 1) Concentric arc polygon strips (3 layers * 22 steps * 2 triangles = 132 triangles + 1 warning triangle)
  assert len(triangles_drawn) >= 130
  # 2) Soft rounded end caps for radar waves
  assert len(circles_drawn) >= 6
  # 3) Warning triangle exclamation mark
  assert len(rects_drawn) >= 1


def test_model_renderer_blindspot_replaces_barrier_with_radar_wave(monkeypatch):
  from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer
  from openpilot.selfdrive.ui.onroad import view_3d_draw
  waves_called = []
  monkeypatch.setattr(view_3d_draw, "draw_blindspot_radar_wave", lambda cx, cy, side, *a, **k: waves_called.append((cx, cy, side)))

  renderer = ModelRenderer()
  renderer._rect = rl.Rectangle(0.0, 0.0, 1920.0, 1080.0)

  # Mock points for left blindspot (x < 960)
  left_points = np.array([[400.0, 700.0], [380.0, 600.0], [350.0, 500.0]], dtype=np.float32)
  renderer._draw_blind_spot_segments_carrot(left_points, rl.Color(255, 100, 20, 255))
  assert len(waves_called) == 1
  assert waves_called[0][2] == -1.0  # side_sign < 0 for left

  # Mock points for right blindspot (x > 960)
  right_points = np.array([[1500.0, 700.0], [1520.0, 600.0], [1550.0, 500.0]], dtype=np.float32)
  renderer._draw_blind_spot_segments_carrot(right_points, rl.Color(255, 100, 20, 255))
  assert len(waves_called) == 2
  assert waves_called[1][2] == 1.0   # side_sign > 0 for right


def test_model_renderer_draws_speed_bump_and_traffic_markers(monkeypatch):
  from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer
  from openpilot.selfdrive.ui.onroad import view_3d_draw

  bumps = []
  lamps = []
  stop_lines = []
  badges = []
  monkeypatch.setattr(view_3d_draw, "draw_road_speed_bump_texture", lambda cx, cy, w, d, *a, **k: bumps.append((cx, cy, w, d)))
  monkeypatch.setattr(view_3d_draw, "draw_overhead_traffic_light_4lamp",
                      lambda cx, cy, w, lamp, d, rem, *a, **k: lamps.append((cx, cy, w, lamp, d, rem)))
  monkeypatch.setattr(rl, "draw_line_ex", lambda *a, **k: stop_lines.append(a))

  renderer = ModelRenderer()
  renderer.set_view_3d_mode(True)
  renderer._rect = rl.Rectangle(0.0, 0.0, 1920.0, 1080.0)
  renderer._clip_region = renderer._rect
  renderer._car_space_transform = np.array([
    [960.0, 1000.0, 0.0],
    [540.0, 0.0, 800.0],
    [1.0, 0.0, 0.0],
  ], dtype=np.float32)

  xs = np.linspace(0.0, 80.0, 100, dtype=np.float32)
  renderer._path.raw_points = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)

  class _FakeSM(dict):
    def __init__(self, data, valid):
      super().__init__(data)
      self.valid = valid

  sm = _FakeSM(
    {
      'carState': SimpleNamespace(speedBumpDistance=32.0, vEgo=0.0),
      'carrotMan': SimpleNamespace(xSpdType=22, xSpdDist=32.0, trafficState=1),
      'longitudinalPlan': SimpleNamespace(trafficState=1, xState='e2eStop', xStop=25.0),
    },
    {'carState': True, 'carrotMan': True, 'longitudinalPlan': True},
  )

  from openpilot.selfdrive.ui.onroad.view_3d_scene import View3dScene, C3_SIZES
  scene = View3dScene(C3_SIZES)
  monkeypatch.setattr(scene, "draw_traffic_light_badge", lambda *a, **k: badges.append(True))
  renderer._view_3d_scene = scene

  # Prevent any local /dev/shm TrafficLight file from overriding the unit test inputs
  from openpilot.selfdrive.ui.ui_state import ui_state
  monkeypatch.setattr(ui_state, "params_memory", SimpleNamespace(get=lambda *a, **k: None), raising=False)

  renderer._draw_road_markers_carrot(sm)

  assert len(bumps) == 1
  assert bumps[0][1] > renderer._rect.height * 0.45
  assert bumps[0][3] == pytest.approx(22.0)  # 32.0 - 10.0m display offset
  assert len(lamps) == 1
  assert renderer._rect.height * 0.24 <= lamps[0][1] <= renderer._rect.height * 0.43
  assert lamps[0][3] == 'red'
  assert lamps[0][4] == pytest.approx(25.0)
  assert len(stop_lines) == 1
  # Badge must not double-draw when overhead 4-lamp was already drawn
  assert len(badges) == 0

  # Close approach: raw 11.0m -> compensated 1.0m stays visible on lower road
  bumps.clear()
  sm['carState'].speedBumpDistance = 11.0
  renderer._draw_road_markers_carrot(sm)
  assert len(bumps) == 1
  assert bumps[0][3] == pytest.approx(1.0)
  assert bumps[0][1] > renderer._rect.height * 0.45

  # Raw 10.0m (already passed physical bump) hides speed bump graphic on time
  bumps.clear()
  sm['carState'].speedBumpDistance = 10.0
  sm['carrotMan'].xSpdDist = 10.0
  renderer._draw_road_markers_carrot(sm)
  assert len(bumps) == 0

  # Final standstill close xStop=4.0m keeps overhead traffic light lowered away from top header
  lamps.clear()
  sm['longitudinalPlan'].xStop = 4.0
  renderer._draw_road_markers_carrot(sm)
  assert len(lamps) == 1
  assert renderer._rect.height * 0.24 <= lamps[0][1] <= renderer._rect.height * 0.43

  # When no signal distance (tl_dist == 0) in 3D mode, fallback traffic badge is called
  lamps.clear()
  badges.clear()
  sm['longitudinalPlan'].xStop = 0.0
  sm['longitudinalPlan'].trafficState = 1
  renderer._path.raw_points = np.zeros((32, 3), dtype=np.float32)  # model stop x == 0 -> no 4-lamp distance
  renderer._draw_road_markers_carrot(sm)
  assert len(lamps) == 0
  assert len(badges) == 1

  # C-ITS / CarrotNavi TrafficLight JSON applies latency compensation (raw_dist - v_ego * age)
  import json
  import time
  lamps.clear()
  renderer._path.raw_points = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
  now_t = time.monotonic()
  payload = json.dumps({"lamp": "green", "distance": 60.0, "remain": 12, "ts": now_t - 1.0})
  monkeypatch.setattr(ui_state, "params_memory", SimpleNamespace(get=lambda *a, **k: payload), raising=False)
  sm['carState'].vEgo = 10.0  # 10 m/s * 1.0 s -> 50.0 m compensated distance
  renderer._carrot_tl_raw_ts = 0.0
  renderer._draw_road_markers_carrot(sm)
  assert len(lamps) == 1
  assert lamps[0][3] == 'green'
  assert lamps[0][4] == pytest.approx(50.0, abs=0.8)
  assert lamps[0][5] in (10, 11)


def test_road_bump_and_4lamp_traffic_light_png_assets_exist():
  from pathlib import Path
  from PIL import Image
  img_dir = Path(__file__).resolve().parents[2] / "assets" / "images"
  for name in ("speed_bump_road.png", "traffic_light_4lamp.png"):
    p = img_dir / name
    assert p.exists(), f"missing asset {p}"
    im = Image.open(p)
    assert im.mode == "RGBA"
    assert im.size[0] > im.size[1]
    assert im.getpixel((0, 0))[3] == 0


def test_lead_distance_calibration_and_tf_bar_display(monkeypatch):
  from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer
  from openpilot.selfdrive.ui.onroad import model_renderer as mr_mod

  assert ModelRenderer._estimate_cam_to_bumper_m(0.0) == pytest.approx(1.52)
  assert ModelRenderer._estimate_cam_to_bumper_m(3.00) == pytest.approx(2.10)

  renderer = ModelRenderer()
  renderer._rect = rl.Rectangle(0.0, 0.0, 2160.0, 1080.0)
  renderer._carrot_cam_to_bumper_m = ModelRenderer._estimate_cam_to_bumper_m(3.00)
  renderer._carrot_lead_status = True
  renderer._carrot_radar_dist = 0.0
  renderer._carrot_vision_dist = 4.7
  renderer._carrot_tf_distance = 7.0
  renderer._carrot_tf_left = (900.0, 800.0)
  renderer._carrot_tf_right = (1260.0, 800.0)

  drawn_texts = []
  monkeypatch.setattr(renderer, "_draw_line_segment_carrot", lambda *a, **k: None)
  monkeypatch.setattr(mr_mod, "draw_text_ui_style", lambda text, *a, **k: drawn_texts.append(text))

  # When tf_distance >= lead_dist - 2.0 at close/standstill range, show actual lead_dist (4.7m), not 7m
  renderer._draw_tf_marker_carrot(is_3d=True, lead_active=True, lead_dist=4.7, center_x=1080.0, disp_y=960.0)
  assert drawn_texts == ["4.7m"]

  # Vision-only over-report correction on long-wheelbase vehicle reduces by (2.10 - 1.52) = 0.58m
  class _FakeSM(dict):
    def __init__(self, data, valid):
      super().__init__(data)
      self.valid = valid

  xs = np.linspace(0.0, 80.0, 100, dtype=np.float32)
  renderer._path.raw_points = np.stack([xs, np.zeros_like(xs), np.zeros_like(xs)], axis=1)
  renderer._clip_region = renderer._rect
  renderer._car_space_transform = np.array([
    [1080.0, -1000.0, 0.0],
    [540.0, 0.0, -800.0],
    [1.0, 0.0, 0.0],
  ], dtype=np.float32)
  sm = _FakeSM(
    {
      'modelV2': SimpleNamespace(leadsV3=[SimpleNamespace(prob=0.9, x=[12.10])]),
      'radarState': SimpleNamespace(
        leadOne=SimpleNamespace(status=True, radar=False, dRel=10.58, yRel=0.0, radarTrackId=-1, dPath=0.0, vLead=5.0, vRel=0.0, aLead=0.0),
        leadTwo=None,
      ),
      'longitudinalPlan': SimpleNamespace(xState=0, trafficState=0, tFollow=1.2, desiredDistance=7.0, longitudinalPlanSource=SimpleNamespace(raw=0)),
      'selfdriveState': SimpleNamespace(enabled=True),
      'carState': SimpleNamespace(vEgo=5.0, brakeHoldActive=False, softHoldActive=False, carrotCruise=False),
    },
    {'radarState': True},
  )
  renderer._update_path_end_carrot(sm)
  assert renderer._carrot_vision_dist == pytest.approx(10.0)
  assert renderer._carrot_lead_dist_m == pytest.approx(10.0)


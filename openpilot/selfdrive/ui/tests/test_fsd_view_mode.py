from types import SimpleNamespace
from unittest.mock import MagicMock
import pyray as rl
import pytest

from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.onroad.fsd_button import FsdButton
from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer


@pytest.fixture(autouse=True)
def mock_gui_app(monkeypatch):
  mock_font = MagicMock()
  monkeypatch.setattr(gui_app, "font", lambda weight=None: mock_font)
  monkeypatch.setattr(gui_app, "texture", lambda *args, **kwargs: MagicMock())
  monkeypatch.setattr("openpilot.system.ui.lib.text_measure.measure_text_cached", lambda font, text, size: rl.Vector2(len(str(text)) * 10, 30))
  monkeypatch.setattr("openpilot.selfdrive.ui.onroad.fsd_button.measure_text_cached", lambda font, text, size: rl.Vector2(len(str(text)) * 10, 30))
  monkeypatch.setattr("openpilot.selfdrive.ui.onroad.model_renderer.measure_text_cached", lambda font, text, size: rl.Vector2(len(str(text)) * 10, 30))
  for fn in ("draw_text_ex", "draw_rectangle_rounded", "draw_rectangle_rounded_lines_ex",
            "draw_ellipse", "draw_triangle", "draw_circle", "draw_rectangle", "draw_rectangle_lines",
            "draw_ring", "draw_ring_lines", "draw_circle_sector", "draw_circle_sector_lines"):
    monkeypatch.setattr(rl, fn, lambda *args, **kwargs: None)
  monkeypatch.setattr("openpilot.selfdrive.ui.mici.onroad.model_renderer.draw_text_ui_style", lambda *args, **kwargs: None)
  monkeypatch.setattr("openpilot.selfdrive.ui.mici.onroad.model_renderer.measure_text_cached", lambda *args, **kwargs: rl.Vector2(40, 15))
  return gui_app


class FakeParams:
  def __init__(self, initial=False):
    self.data = {"FsdViewMode": initial}

  def get_bool(self, key):
    return self.data.get(key, False)

  def put_bool(self, key, val):
    self.data[key] = bool(val)

  def put_nonblocking(self, key, val):
    pass


def test_fsd_button_toggle(monkeypatch):
  fake_params = FakeParams(initial=False)
  monkeypatch.setattr("openpilot.selfdrive.ui.onroad.fsd_button.Params", lambda: fake_params)

  btn = FsdButton(width=150, height=68)
  assert btn.fsd_mode is False

  btn._handle_mouse_release(None)
  assert btn.fsd_mode is True
  assert fake_params.get_bool("FsdViewMode") is True

  btn._handle_mouse_release(None)
  assert btn.fsd_mode is False
  assert fake_params.get_bool("FsdViewMode") is False


def test_model_renderer_fsd_mode_switch():
  renderer = ModelRenderer()
  assert renderer._fsd_mode is False

  renderer.set_fsd_mode(True)
  assert renderer._fsd_mode is True

  renderer.set_fsd_mode(False)
  assert renderer._fsd_mode is False


def test_model_renderer_fsd_draw_call():
  renderer = ModelRenderer()
  renderer.set_fsd_mode(True)

  calls = []
  renderer._draw_fsd_visualization = lambda sm: calls.append("fsd")
  renderer._draw_path_carrot = lambda sm: calls.append("path")
  renderer._render_diagnostics = SimpleNamespace(
    start=lambda: None,
    finish=lambda: None,
    call=lambda name, callback, *args: callback(*args),
    values={}
  )

  renderer._draw_carrot_overlays(object())
  assert calls == ["fsd"]

  calls.clear()
  renderer.set_fsd_mode(False)
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


def test_hud_renderer_set_fsd_theme():
  from openpilot.selfdrive.ui.onroad.hud_renderer import HudRenderer
  hud = HudRenderer()
  assert hud._fsd_mode is False
  assert hud._is_night is True

  hud.set_fsd_theme(True, is_night=False)
  assert hud._fsd_mode is True
  assert hud._is_night is False

  hud.set_fsd_theme(True, is_night=True)
  assert hud._fsd_mode is True
  assert hud._is_night is True


def test_mici_model_renderer_fsd_mode():
  from openpilot.selfdrive.ui.mici.onroad.model_renderer import ModelRenderer as MiciModelRenderer
  mici_model = MiciModelRenderer()
  assert mici_model._fsd_mode is False
  mici_model.set_fsd_mode(True, is_night=False)
  assert mici_model._fsd_mode is True
  assert mici_model._fsd_is_night is False


def test_mici_hud_renderer_fsd_theme():
  from openpilot.selfdrive.ui.mici.onroad.hud_renderer import HudRenderer as MiciHudRenderer
  mici_hud = MiciHudRenderer()
  assert mici_hud._fsd_mode is False
  mici_hud.set_fsd_theme(True, is_night=False)
  assert mici_hud._fsd_mode is True
  assert mici_hud._is_night is False


def test_fsd_button_compact_vs_full(monkeypatch):
  fake_params = FakeParams(initial=False)
  monkeypatch.setattr("openpilot.selfdrive.ui.onroad.fsd_button.Params", lambda: fake_params)

  btn_compact = FsdButton(width=80, height=34)
  btn_full = FsdButton(width=210, height=78)

  # Check compact mode rendering
  btn_compact._render(rl.Rectangle(0, 0, 80, 34))
  # Check full mode rendering
  btn_full._render(rl.Rectangle(0, 0, 210, 78))


def test_fsd_button_text_labels(monkeypatch):
  fake_params = FakeParams(initial=False)
  monkeypatch.setattr("openpilot.selfdrive.ui.onroad.fsd_button.Params", lambda: fake_params)

  btn = FsdButton(width=210, height=78)
  # When in AR view (fsd_mode=False) -> Offers 3D view
  assert btn.fsd_mode is False

  # When in 3D view (fsd_mode=True) -> Offers AR view
  btn._handle_mouse_release(None)
  assert btn.fsd_mode is True


def test_model_renderer_lead_vehicles_l1_l2():
  import numpy as np
  from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer

  renderer = ModelRenderer()
  lead1 = SimpleNamespace(status=True, dRel=28.0, yRel=0.0, vRel=-1.5)
  lead2 = SimpleNamespace(status=True, dRel=65.0, yRel=0.5, vRel=0.0)
  radar_state = SimpleNamespace(leadOne=lead1, leadTwo=lead2)

  class FakeSubMaster(dict):
    def __init__(self, **kwargs):
      super().__init__(kwargs)
      self.valid = dict.fromkeys(kwargs, True)

  sm = FakeSubMaster(
    radarState=radar_state,
    carState=SimpleNamespace(brakePressed=False, aEgo=0.0, leftBlinker=False, rightBlinker=False, leftBlindspot=False, rightBlindspot=False),
  )

  renderer._path = SimpleNamespace(raw_points=np.zeros((10, 3)))
  renderer._path_offset_z = 0.0
  renderer._map_to_screen = lambda d, y, z: (1080.0, 540.0)

  # Verify drawing runs cleanly without exception for both night and day modes
  renderer._draw_fsd_lead_vehicles(sm, is_night=True)
  renderer._draw_fsd_lead_vehicles(sm, is_night=False)


def test_mici_model_renderer_lead_vehicles():
  from openpilot.selfdrive.ui.mici.onroad.model_renderer import ModelRenderer as MiciModelRenderer

  mici_model = MiciModelRenderer()
  lead1 = SimpleNamespace(status=True, dRel=32.0, yRel=0.0, vRel=-2.0)
  lead2 = SimpleNamespace(status=True, dRel=70.0, yRel=0.2, vRel=0.0)
  radar_state = SimpleNamespace(leadOne=lead1, leadTwo=lead2)

  mici_model._project_point_to_screen = lambda d, y, z: (268.0, 120.0)
  mici_model._path_offset_z = 0.0

  mici_model._draw_fsd_lead_vehicles(radar_state, is_night=True)
  mici_model._draw_fsd_lead_vehicles(radar_state, is_night=False)


def test_fsd_steering_wheel_ego_rendering():
  from openpilot.selfdrive.ui.onroad.model_renderer import ModelRenderer
  from openpilot.selfdrive.ui.mici.onroad.model_renderer import ModelRenderer as MiciModelRenderer

  class FakeSubMaster(dict):
    def __init__(self, **kwargs):
      super().__init__(kwargs)
      self.valid = dict.fromkeys(kwargs, True)

  sm = FakeSubMaster(
    carState=SimpleNamespace(steeringAngleDeg=15.0, leftBlindspot=True, rightBlindspot=False),
  )

  renderer = ModelRenderer()
  renderer._rect = rl.Rectangle(0, 0, 2160, 1080)
  renderer._draw_fsd_ego_vehicle(sm, is_night=True)
  renderer._draw_fsd_ego_vehicle(sm, is_night=False)

  mici_model = MiciModelRenderer()
  mici_model._rect = rl.Rectangle(0, 0, 536, 240)
  mici_model._draw_fsd_ego_vehicle(sm, is_night=True)
  mici_model._draw_fsd_ego_vehicle(sm, is_night=False)




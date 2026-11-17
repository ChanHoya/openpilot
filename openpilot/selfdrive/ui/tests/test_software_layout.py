import pytest
from unittest.mock import MagicMock
from openpilot.selfdrive.ui.layouts.settings.software import SoftwareLayout
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.manager.process_config import enable_updated


@pytest.fixture
def mock_gui_app(monkeypatch):
  import pyray as rl
  from openpilot.system.ui.lib.application import gui_app
  mock_font = MagicMock()
  monkeypatch.setattr(gui_app, "font", lambda weight=None: mock_font)
  monkeypatch.setattr(gui_app, "texture", lambda *args, **kwargs: MagicMock())
  monkeypatch.setattr("openpilot.system.ui.lib.text_measure.measure_text_cached", lambda font, text, size: rl.Vector2(len(str(text)) * 10, 30))
  monkeypatch.setattr("openpilot.system.ui.widgets.list_view.measure_text_cached", lambda font, text, size: rl.Vector2(len(str(text)) * 10, 30))
  monkeypatch.setattr("openpilot.system.ui.widgets.label.measure_text_cached", lambda font, text, size: rl.Vector2(len(str(text)) * 10, 30))
  return gui_app


def test_can_update_offroad(mock_gui_app, monkeypatch):
  layout = SoftwareLayout()
  monkeypatch.setattr(ui_state, "is_offroad", lambda: True)
  assert layout._can_update() is True


def test_can_update_onroad_engaged(mock_gui_app, monkeypatch):
  layout = SoftwareLayout()
  monkeypatch.setattr(ui_state, "is_offroad", lambda: False)
  monkeypatch.setattr(type(ui_state), "engaged", property(lambda self: True))
  assert layout._can_update() is False


def test_can_update_onroad_moving(mock_gui_app, monkeypatch):
  layout = SoftwareLayout()
  monkeypatch.setattr(ui_state, "is_offroad", lambda: False)
  monkeypatch.setattr(type(ui_state), "engaged", property(lambda self: False))

  mock_sm = MagicMock()
  mock_sm.recv_frame = {"carState": 100}
  mock_cs = MagicMock()
  mock_cs.standstill = False
  mock_cs.vEgo = 15.0
  mock_cs.gearShifter = 2  # Drive
  mock_sm.__getitem__.side_effect = lambda key: mock_cs if key == "carState" else None

  monkeypatch.setattr(ui_state, "sm", mock_sm)
  assert layout._can_update() is False


def test_can_update_onroad_standstill(mock_gui_app, monkeypatch):
  layout = SoftwareLayout()
  monkeypatch.setattr(ui_state, "is_offroad", lambda: False)
  monkeypatch.setattr(type(ui_state), "engaged", property(lambda self: False))

  mock_sm = MagicMock()
  mock_sm.recv_frame = {"carState": 100}
  mock_cs = MagicMock()
  mock_cs.standstill = True
  mock_cs.vEgo = 0.0
  mock_cs.gearShifter = 2
  mock_sm.__getitem__.side_effect = lambda key: mock_cs if key == "carState" else None

  monkeypatch.setattr(ui_state, "sm", mock_sm)
  assert layout._can_update() is True


def test_can_update_onroad_park(mock_gui_app, monkeypatch):
  layout = SoftwareLayout()
  monkeypatch.setattr(ui_state, "is_offroad", lambda: False)
  monkeypatch.setattr(type(ui_state), "engaged", property(lambda self: False))

  mock_sm = MagicMock()
  mock_sm.recv_frame = {"carState": 100}
  mock_cs = MagicMock()
  mock_cs.standstill = False
  mock_cs.vEgo = 0.0
  mock_cs.gearShifter = 1  # Park
  mock_sm.__getitem__.side_effect = lambda key: mock_cs if key == "carState" else None

  monkeypatch.setattr(ui_state, "sm", mock_sm)
  assert layout._can_update() is True


def test_enable_updated_process_onroad():
  params = MagicMock()
  params.get_bool.side_effect = lambda key: key == "SoftwareMenu"
  # Should be enabled even when started=True
  assert enable_updated(started=True, params=params, CP=MagicMock()) is True
  assert enable_updated(started=False, params=params, CP=MagicMock()) is True

import pytest
from unittest.mock import MagicMock, patch
from openpilot.common.params import Params
from openpilot.system.ui.widgets import DialogResult
from openpilot.selfdrive.ui.onroad.camera_scc_button import (
  CameraSCCButton,
  SCC_MODE_NAMES,
  OPTIONS_LIST,
)


class FakeRect:
  def __init__(self, x=0, y=0, width=160, height=88):
    self.x = x
    self.y = y
    self.width = width
    self.height = height


import pyray as rl
from openpilot.system.ui.lib.application import gui_app


@pytest.fixture
def mock_gui_app(monkeypatch):
  mock_font = MagicMock()
  monkeypatch.setattr(gui_app, "font", lambda weight=None: mock_font)
  monkeypatch.setattr(gui_app, "texture", lambda *args, **kwargs: MagicMock())
  monkeypatch.setattr("openpilot.system.ui.lib.text_measure.measure_text_cached", lambda font, text, size: rl.Vector2(len(str(text)) * 10, 30))
  monkeypatch.setattr("openpilot.system.ui.widgets.label.measure_text_cached", lambda font, text, size: rl.Vector2(len(str(text)) * 10, 30))
  monkeypatch.setattr("openpilot.selfdrive.ui.onroad.camera_scc_button.measure_text_cached", lambda font, text, size: rl.Vector2(len(str(text)) * 10, 30))
  return gui_app


def test_camera_scc_mode_names():
  assert SCC_MODE_NAMES[0] == "순정(0)"
  assert SCC_MODE_NAMES[1] == "오파(1)"
  assert SCC_MODE_NAMES[2] == "동기(2)"
  assert SCC_MODE_NAMES[3] == "연결(3)"
  assert len(OPTIONS_LIST) == 4


def test_camera_scc_button_rect(mock_gui_app):
  btn = CameraSCCButton(width=160, height=88)
  assert btn.rect.width == 160
  assert btn.rect.height == 88

  btn.set_rect(FakeRect(100, 200, 180, 90))
  assert btn.rect.x == 100
  assert btn.rect.y == 200
  assert btn.rect.width == 180
  assert btn.rect.height == 90


def test_camera_scc_button_standstill_dialog_confirm(mock_gui_app, monkeypatch):
  params = Params()
  params.put_int("HyundaiCameraSCC", 0)

  btn = CameraSCCButton()
  assert btn._get_current_scc_val() == 0

  # Mock standstill to True
  monkeypatch.setattr(btn, "_is_standstill", lambda: True)

  pushed_widgets = []
  monkeypatch.setattr(gui_app, "push_widget", lambda w: pushed_widgets.append(w))

  btn._show_scc_selection_dialog()
  assert len(pushed_widgets) == 1
  multi_dialog = pushed_widgets[0]
  assert multi_dialog.title == "HYUNDAI CAMERA SCC 모드 선택"

  # User selects option 1 (오파 롱컨)
  multi_dialog.selection = OPTIONS_LIST[1]

  # Trigger callback of multi_dialog with CONFIRM
  multi_dialog._callback(DialogResult.CONFIRM)

  # Check that ConfirmDialog was pushed
  assert len(pushed_widgets) == 2
  confirm_dialog = pushed_widgets[1]
  assert "정지 상태가 확인" in confirm_dialog._label._text

  # User confirms soft reset
  with patch.object(btn, "_apply_mode_and_restart") as mock_apply:
    confirm_dialog._callback(DialogResult.CONFIRM)
    mock_apply.assert_called_once_with(1)


def test_camera_scc_button_driving_pending_and_standstill_restart(mock_gui_app, monkeypatch):
  params = Params()
  params.put_int("HyundaiCameraSCC", 0)

  btn = CameraSCCButton()
  # Mock driving (not standstill)
  standstill_state = {"stopped": False}
  monkeypatch.setattr(btn, "_is_standstill", lambda: standstill_state["stopped"])

  pushed_widgets = []
  monkeypatch.setattr(gui_app, "push_widget", lambda w: pushed_widgets.append(w))

  btn._show_scc_selection_dialog()
  multi_dialog = pushed_widgets[0]

  # User selects option 1 while driving
  multi_dialog.selection = OPTIONS_LIST[1]
  multi_dialog._callback(DialogResult.CONFIRM)

  # Check that pending notice dialog was pushed
  assert len(pushed_widgets) == 2
  notice_dialog = pushed_widgets[1]
  assert "예약" in notice_dialog._label._text
  assert btn._pending_scc_mode == 1

  # Confirm notice
  notice_dialog._callback(DialogResult.CONFIRM)
  assert btn._pending_scc_mode == 1

  # While driving, update_state does not trigger restart
  with patch.object(btn, "_apply_mode_and_restart") as mock_apply:
    btn.update_state()
    mock_apply.assert_not_called()

  # Now vehicle comes to a complete standstill
  standstill_state["stopped"] = True
  with patch.object(btn, "_apply_mode_and_restart") as mock_apply:
    btn.update_state()
    mock_apply.assert_called_once_with(1)
    assert btn._pending_scc_mode is None
    assert btn._is_restarting is True


def test_camera_scc_button_dialog_cancel(mock_gui_app, monkeypatch):
  params = Params()
  params.put_int("HyundaiCameraSCC", 1)

  btn = CameraSCCButton()
  monkeypatch.setattr(btn, "_is_standstill", lambda: True)

  pushed_widgets = []
  monkeypatch.setattr(gui_app, "push_widget", lambda w: pushed_widgets.append(w))

  btn._show_scc_selection_dialog()
  multi_dialog = pushed_widgets[0]

  # User cancels in multi_dialog
  multi_dialog._callback(DialogResult.CANCEL)
  assert len(pushed_widgets) == 1  # No confirm dialog pushed

  # User opens again and cancels on ConfirmDialog
  btn._show_scc_selection_dialog()
  multi_dialog_2 = pushed_widgets[1]
  multi_dialog_2.selection = OPTIONS_LIST[0]
  multi_dialog_2._callback(DialogResult.CONFIRM)

  assert len(pushed_widgets) == 3
  confirm_dialog = pushed_widgets[2]

  with patch.object(btn, "_apply_mode_and_restart") as mock_apply:
    confirm_dialog._callback(DialogResult.CANCEL)
    mock_apply.assert_not_called()

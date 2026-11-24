from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pyray as rl
import pytest

from openpilot.selfdrive.carrot.drive_report import DriveReportStore, DriveReportTracker
from openpilot.selfdrive.ui.layouts.home import HomeLayout
from openpilot.selfdrive.ui.onroad import augmented_road_view
from openpilot.selfdrive.ui.onroad.drive_report_dialog import (
  DriveReportDialog,
  DriveReportMiniHud,
  ICON_AUTO,
  ICON_BRAKE,
  ICON_GAS,
  ICON_STEER,
)
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import ASSETS_DIR, MousePos, gui_app
from openpilot.system.ui.widgets import Widget


def _sample_report() -> dict:
  return {
    "report_id": "20261123_185012",
    "time_range_label": "2026-11-23 18:12 ~ 18:50",
    "vehicle_name": "GENESIS G80 FL",
    "distance_m": 24600.0,
    "distance_km": 24.6,
    "duration_s": 2294.0,
    "duration_text": "38분 14초",
    "moving_s": 1950.0,
    "moving_text": "32분 30초",
    "idle_s": 344.0,
    "idle_text": "5분 44초",
    "score": 91,
    "grade": "A",
    "deductions": [{"reason": "수동 급정거 1회", "points": -8}],
    "ratios": {"auto_pct": 82, "steer_pct": 5, "pedal_pct": 13},
    "controls": {
      "auto": {"label": "자동주행", "seconds": 1881.0, "count": 6, "text": "31분 21초 (6회)"},
      "steer": {"label": "수동조향", "seconds": 112.0, "count": 4, "text": "1분 52초 (4회)"},
      "gas": {"label": "수동페달 (가속)", "seconds": 184.0, "count": 9, "text": "3분 04초 (9회)"},
      "brake": {"label": "수동페달 (브레이크)", "seconds": 117.0, "count": 7, "text": "1분 57초 (7회)"},
    },
    "speed": {
      "avg_kph": 38.5,
      "max_kph": 84.2,
      "long_jerk_rms": 0.42,
      "long_score": 92,
      "lat_accel_peak": 1.65,
      "lat_score": 91,
    },
    "hard_brakes": {
      "threshold": -2.8,
      "total_count": 1,
      "auto_count": 0,
      "manual_count": 1,
      "auto_peak_decel": 0.0,
      "manual_peak_decel": -3.12,
      "events": [
        {
          "time_str": "18:34",
          "source": "manual",
          "source_label": "수동",
          "start_speed_kph": 54,
          "end_speed_kph": 12,
          "peak_decel": -3.12,
        }
      ],
    },
    "file_path": "/data/drive_reports/20261123_185012.json",
  }


@pytest.fixture
def mock_raylib(monkeypatch):
  drawn_labels = []
  drawn_label_sizes = []
  drawn_icons = []
  drawn_textures = []
  drawn_texts = []
  monkeypatch.setattr(gui_app, "font", lambda weight=None: MagicMock())
  monkeypatch.setattr(gui_app, "texture", lambda asset_path, w=None, h=None, **kwargs: (
    drawn_icons.append(asset_path) or NS(id=1, width=w or 128, height=h or 128, asset_path=asset_path)
  ))
  monkeypatch.setattr("openpilot.system.ui.lib.text_measure.measure_text_cached", lambda font, text, size: rl.Vector2(len(str(text)) * 12, size))
  monkeypatch.setattr("openpilot.selfdrive.ui.layouts.home.measure_text_cached", lambda font, text, size: rl.Vector2(len(str(text)) * 12, size))

  def _cap_label(rect, text, *args, **kwargs):
    drawn_labels.append((rect, str(text)))
    fs = args[0] if len(args) >= 1 else kwargs.get("font_size", 0)
    drawn_label_sizes.append((str(text), int(fs)))

  monkeypatch.setattr("openpilot.selfdrive.ui.onroad.drive_report_dialog.gui_label", _cap_label)
  monkeypatch.setattr("openpilot.selfdrive.ui.layouts.home.gui_label", _cap_label)
  monkeypatch.setattr(rl, "draw_rectangle_rec", lambda *args, **kwargs: None)
  monkeypatch.setattr(rl, "draw_rectangle_rounded", lambda *args, **kwargs: None)
  monkeypatch.setattr(rl, "draw_rectangle_rounded_lines_ex", lambda *args, **kwargs: None)
  monkeypatch.setattr(rl, "draw_line_ex", lambda *args, **kwargs: None)
  monkeypatch.setattr(rl, "draw_circle", lambda *args, **kwargs: None)
  monkeypatch.setattr(rl, "draw_ring", lambda *args, **kwargs: None)
  monkeypatch.setattr(rl, "draw_text_ex", lambda font, text, *args, **kwargs: drawn_texts.append(str(text)))

  def _cap_tex(tex, src, dst, origin, rot, tint):
    drawn_textures.append((getattr(tex, "asset_path", ""), tint))

  monkeypatch.setattr(rl, "draw_texture_pro", _cap_tex)
  monkeypatch.setattr(rl, "is_key_pressed", lambda *args, **kwargs: False)
  monkeypatch.setattr(rl, "check_collision_point_rec", lambda pos, rec: (
    rec.x <= pos.x <= rec.x + rec.width and rec.y <= pos.y <= rec.y + rec.height
  ))
  return NS(labels=drawn_labels, label_sizes=drawn_label_sizes, icons=drawn_icons, textures=drawn_textures, texts=drawn_texts)


def test_drive_report_icon_assets_exist():
  for asset_rel in (ICON_AUTO, ICON_STEER, ICON_GAS, ICON_BRAKE):
    path = Path(ASSETS_DIR) / asset_rel
    assert path.is_file(), f"missing asset: {path}"


def test_dialog_renders_all_three_pages_and_handles_touch(mock_raylib):
  closed_calls = []
  report = _sample_report()
  dialog = DriveReportDialog(
    report,
    on_close=lambda: closed_calls.append(True),
  )

  # Render page 0 (2160x1080 widescreen C3/3X)
  mock_raylib.labels.clear()
  mock_raylib.icons.clear()
  dialog._render(rl.Rectangle(0, 0, 2160, 1080))
  texts_p0 = [t for _, t in mock_raylib.labels]
  assert set(mock_raylib.icons) == {ICON_AUTO, ICON_STEER, ICON_GAS, ICON_BRAKE}
  assert any("자동주행 82%" in t for t in texts_p0)
  assert any("수동조향 5%" in t for t in texts_p0)
  assert any("수동페달 13%" in t for t in texts_p0)
  assert any("정차 5분 44초" in t for t in texts_p0)
  assert any("31분 21초 (6회)" in t for t in texts_p0)

  # Touch next button -> Page 1
  next_btn = dialog._btn_next
  dialog._handle_mouse_release(MousePos(next_btn.x + 10, next_btn.y + 10))
  assert dialog.page == 1
  mock_raylib.labels.clear()
  dialog._render(rl.Rectangle(0, 0, 2160, 1080))
  texts_p1 = [t for _, t in mock_raylib.labels]
  assert any("38.5 km/h" in t for t in texts_p1)
  assert any("84.2 km/h" in t for t in texts_p1)

  # Touch 3rd tab directly -> Page 2
  tab3 = dialog._tab_rects[2]
  dialog._handle_mouse_release(MousePos(tab3.x + 10, tab3.y + 10))
  assert dialog.page == 2
  mock_raylib.labels.clear()
  dialog._render(rl.Rectangle(0, 0, 2160, 1080))
  texts_p2 = [t for _, t in mock_raylib.labels]
  assert any("자동주행 급정거" in t for t in texts_p2)
  assert any("수동 급정거" in t for t in texts_p2)
  assert any("단말에 저장되었습니다" in t for t in texts_p2)
  assert not any("카톡" in t for t in texts_p2)

  # Close button dismisses dialog
  dialog._handle_mouse_release(MousePos(dialog._btn_done.x + 10, dialog._btn_done.y + 10))
  assert dialog.is_closed is True
  assert closed_calls == [True]


def test_dialog_auto_closes_when_shifting_out_of_park(mock_raylib, monkeypatch):
  pushed = []
  popped = []
  nav_stack = [object()]

  def fake_push(w):
    nav_stack.append(w)
    pushed.append(w)

  def fake_pop(idx=None):
    target_idx = len(nav_stack) - 1 if idx is None else idx
    popped.append(nav_stack.pop(target_idx))

  monkeypatch.setattr(gui_app, "_nav_stack", nav_stack, raising=False)
  monkeypatch.setattr(gui_app, "push_widget", fake_push)
  monkeypatch.setattr(gui_app, "pop_widget", fake_pop)
  monkeypatch.setattr(gui_app, "widget_in_stack", lambda w: w in nav_stack)

  car_state = NS(gearShifter="park", vEgo=0.0, aEgo=0.0, steeringPressed=False, gasPressed=False, brakePressed=False)

  class FakeSM(dict):
    def __init__(self):
      super().__init__({
        "carState": car_state,
        "carControl": NS(enabled=False, latActive=False, longActive=False),
        "selfdriveState": NS(enabled=False),
      })
      self.valid = {"carState": True, "carControl": True, "selfdriveState": True, "controlsState": False}
      self.updated = {"carState": True}
      self.recv_frame = {"carState": 1}

  sm = FakeSM()
  monkeypatch.setattr(ui_state, "started", True)
  monkeypatch.setattr(ui_state, "sm", sm)

  dialog = DriveReportDialog(_sample_report())
  gui_app.push_widget(dialog)
  assert len(nav_stack) == 2

  dialog._update_state()
  assert not dialog.is_closed

  sm["carState"].gearShifter = "drive"
  dialog._update_state()
  assert dialog.is_closed is True
  assert popped == [dialog]


def test_augmented_road_view_hook_triggers_popup_on_park_and_skips_short_trip(tmp_path, mock_raylib, monkeypatch):
  pushed = []
  monkeypatch.setattr(gui_app, "push_widget", lambda w: pushed.append(w))

  view = augmented_road_view.AugmentedRoadView.__new__(augmented_road_view.AugmentedRoadView)
  view.close = lambda: None
  view._drive_report_tracker = DriveReportTracker(min_distance_m=300.0, min_duration_s=60.0)
  view._drive_report_store = DriveReportStore(base_dir=tmp_path)
  view._drive_report_dialog = None
  view._last_drive_report_car_frame = -1

  class FakeSM(dict):
    def __init__(self):
      super().__init__({
        "carState": NS(gearShifter="drive", vEgo=20.0, aEgo=0.0, steeringPressed=False, gasPressed=False, brakePressed=False),
        "carControl": NS(enabled=True, latActive=True, longActive=True),
        "selfdriveState": NS(enabled=True),
      })
      self.valid = {"carState": True, "carControl": True, "selfdriveState": True, "controlsState": False}
      self.updated = {"carState": True}
      self.recv_frame = {"carState": 0}

  sm = FakeSM()
  monkeypatch.setattr(ui_state, "started", True)
  monkeypatch.setattr(ui_state, "sm", sm)
  monkeypatch.setattr(ui_state, "CP", NS(carFingerprint="GENESIS_G80_2ND_GEN_FL"))

  t_box = [100.0]
  monkeypatch.setattr(augmented_road_view.time, "monotonic", lambda: t_box[0])
  monkeypatch.setattr(augmented_road_view.time, "time", lambda: 1795430000.0 + t_box[0])

  # 1) Short trip: 20m -> P does not trigger popup
  for i in range(1, 5):
    t_box[0] = 100.0 + i * 0.5
    sm.recv_frame["carState"] = i
    sm["carState"].vEgo = 5.0
    view._update_drive_report()
  sm.recv_frame["carState"] = 5
  sm["carState"].gearShifter = "park"
  sm["carState"].vEgo = 0.0
  view._update_drive_report()
  assert pushed == []

  # 2) Valid trip: > 300m -> P saves & pushes DriveReportDialog
  sm["carState"].gearShifter = "drive"
  sm["carState"].vEgo = 20.0
  for i in range(6, 30):
    t_box[0] = 200.0 + (i - 6) * 1.0
    sm.recv_frame["carState"] = i
    view._update_drive_report()
  t_box[0] = 225.0
  sm.recv_frame["carState"] = 30
  sm["carState"].gearShifter = "park"
  sm["carState"].vEgo = 0.0
  view._update_drive_report()

  assert len(pushed) == 1
  assert isinstance(pushed[0], DriveReportDialog)
  assert len(view._drive_report_store.list_reports()) == 1


def test_dialog_enlarged_fonts_and_saved_report_history_browser(tmp_path, mock_raylib):
  store = DriveReportStore(base_dir=tmp_path)
  for idx in range(7):
    rep = _sample_report()
    rep["report_id"] = f"20261124_08000{idx}"
    rep["ended_at"] = 1795500000.0 + idx * 600.0
    rep["time_range_label"] = f"2026-11-24 08:0{idx} ~ 08:1{idx}"
    rep["score"] = 70 + idx * 4
    rep["distance_km"] = 10.0 + idx
    store.save_report(rep)

  latest = store.get_report("20261124_080006")
  dialog = DriveReportDialog(latest, store=store)

  # 1) Check enlarged font sizes on Page 0 / Page 1 / Page 2
  mock_raylib.label_sizes.clear()
  dialog._render(rl.Rectangle(0, 0, 2160, 1080))
  fs_p0 = dict(mock_raylib.label_sizes)
  assert fs_p0["31분 21초 (6회)"] >= 44
  assert fs_p0["자동주행"] >= 34
  assert fs_p0["오파 활성 누적"] >= 26

  dialog.set_page(1)
  mock_raylib.label_sizes.clear()
  dialog._render(rl.Rectangle(0, 0, 2160, 1080))
  fs_p1 = dict(mock_raylib.label_sizes)
  assert fs_p1["38.5 km/h"] >= 60
  assert fs_p1["가감속 부드러움 (종방향)"] >= 34

  dialog.set_page(2)
  mock_raylib.label_sizes.clear()
  dialog._render(rl.Rectangle(0, 0, 2160, 1080))
  fs_p2 = dict(mock_raylib.label_sizes)
  assert fs_p2["18:34  [수동]  54 -> 12 km/h"] >= 30

  # 2) Open history list via Page 3 button ("저장 목록 >") and verify pagination + load
  btn_hist = dialog._btn_page3_history
  dialog._handle_mouse_release(MousePos(btn_hist.x + 10, btn_hist.y + 10))
  assert dialog.history_mode is True
  assert dialog.history_page == 0

  mock_raylib.labels.clear()
  dialog._render(rl.Rectangle(0, 0, 2160, 1080))
  assert len(dialog._history_row_rects) == 5
  assert any("저장된 주행 리포트 목록" in txt for _, txt in mock_raylib.labels)

  # Go to page 2 of history (2 oldest reports) and click one to load it
  dialog._handle_mouse_release(MousePos(dialog._btn_next.x + 10, dialog._btn_next.y + 10))
  assert dialog.history_page == 1
  dialog._render(rl.Rectangle(0, 0, 2160, 1080))
  assert len(dialog._history_row_rects) == 2

  row_rect, chosen_id = dialog._history_row_rects[1]
  dialog._handle_mouse_release(MousePos(row_rect.x + 20, row_rect.y + 20))
  assert dialog.history_mode is False
  assert dialog.page == 0
  assert dialog._report["report_id"] == chosen_id


def test_mini_hud_blinks_active_channel_and_dims_inactive(mock_raylib):
  hud = DriveReportMiniHud()
  live_stats = {
    "trip_started": True,
    "gear": "drive",
    "items": [
      {"key": "auto", "label": "자동주행", "seconds": 125.0, "time_text": "02분05초", "pct": 62.5, "active": True},
      {"key": "steer", "label": "수동조향", "seconds": 30.0, "time_text": "00분30초", "pct": 15.0, "active": False},
      {"key": "gas", "label": "수동가속", "seconds": 25.0, "time_text": "00분25초", "pct": 12.5, "active": False},
      {"key": "brake", "label": "수동감속", "seconds": 20.0, "time_text": "00분20초", "pct": 10.0, "active": False},
    ],
  }

  mock_raylib.textures.clear()
  mock_raylib.label_sizes.clear()
  rect = hud.render(rl.Rectangle(0, 0, 2160, 1080), live_stats, now_t=0.0)
  assert rect.x > 2160 * 0.60 and rect.y > 700
  assert rect.width <= 710.0
  mini_sizes = dict(mock_raylib.label_sizes)
  assert mini_sizes["자동주행 02분05초(62.5%)"] >= 36
  assert mini_sizes["수동조향 00분30초(15.0%)"] >= 33
  texts = [s[1] for s in hud._last_draw_state]
  assert texts == [
    "자동주행 02분05초(62.5%)",
    "수동조향 00분30초(15.0%)",
    "수동가속 00분25초(12.5%)",
    "수동감속 00분20초(10.0%)",
  ]
  # Active logo keeps full RGB color; inactive logos are gray-tinted & muted
  by_asset = {asset: tint for asset, tint in mock_raylib.textures}
  assert (by_asset[ICON_AUTO].r, by_asset[ICON_AUTO].g, by_asset[ICON_AUTO].b) == (255, 255, 255)
  assert by_asset[ICON_AUTO].a >= 170
  assert by_asset[ICON_STEER].a <= 120 and by_asset[ICON_STEER].r < 160
  assert by_asset[ICON_GAS].a <= 120 and by_asset[ICON_BRAKE].a <= 120

  # Active channel alpha varies with time (blinking glow)
  alphas = []
  for t in (0.0, 0.17, 0.35, 0.52):
    hud.render(rl.Rectangle(0, 0, 2160, 1080), live_stats, now_t=t)
    alphas.append(hud._last_draw_state[0][3])
  assert len(set(alphas)) >= 2


def test_mini_hud_tap_in_park_and_home_button_open_history(tmp_path, mock_raylib, monkeypatch):
  monkeypatch.setenv("DRIVE_REPORTS_DIR", str(tmp_path))
  store = DriveReportStore(base_dir=tmp_path)
  store.save_report(_sample_report())

  pushed = []
  monkeypatch.setattr(gui_app, "push_widget", lambda w: pushed.append(w))

  view = augmented_road_view.AugmentedRoadView.__new__(augmented_road_view.AugmentedRoadView)
  view.close = lambda: None
  view._drive_report_tracker = DriveReportTracker()
  view._drive_report_store = store
  view._drive_report_mini_hud = DriveReportMiniHud()
  view._drive_report_dialog = None
  view._view_3d_button = NS(is_pressed=False, rect=rl.Rectangle(0, 0, 0, 0))
  view._hud_renderer = NS(user_interacting=lambda: False)
  clicks = []
  view._click_callback = lambda: clicks.append(True)

  class FakeSM(dict):
    def __init__(self):
      super().__init__({"carState": NS(gearShifter="drive", vEgo=15.0, aEgo=0.0)})
      self.valid = {"carState": True}

  sm = FakeSM()
  monkeypatch.setattr(ui_state, "sm", sm)

  hud_rect = view._drive_report_mini_hud.render(rl.Rectangle(0, 0, 2160, 1080), view._drive_report_tracker.get_live_stats(), now_t=1.0)
  tap_pos = MousePos(hud_rect.x + 20, hud_rect.y + 20)

  # Tapping mini HUD during D must not toggle sidebar or open modal
  view._handle_mouse_press(tap_pos)
  view._handle_mouse_release(tap_pos)
  assert clicks == []
  assert pushed == []

  # Tapping mini HUD in Park opens DriveReportDialog directly in history mode
  sm["carState"].gearShifter = "park"
  view._handle_mouse_release(tap_pos)
  assert len(pushed) == 1
  assert isinstance(pushed[0], DriveReportDialog)
  assert pushed[0].history_mode is True

  # HomeLayout header button also opens DriveReportDialog in history mode
  pushed.clear()
  home = HomeLayout.__new__(HomeLayout)
  Widget.__init__(home)
  home.update_available = False
  home.alert_count = 0
  home.current_state = 0
  home._version_text = "openpilot"
  home._drive_report_dialog = None
  home.header_rect = rl.Rectangle(40, 40, 1500, 80)
  home.update_notif_rect = rl.Rectangle(0, 0, 200, 70)
  home.alert_notif_rect = rl.Rectangle(0, 0, 220, 70)
  home.drive_report_btn_rect = rl.Rectangle(40, 50, 240, 70)
  mock_raylib.texts.clear()
  home._render_header()
  assert any("주행 리포트" in t for t in mock_raylib.texts)
  home._handle_mouse_release(MousePos(60, 70))
  assert len(pushed) == 1
  assert isinstance(pushed[0], DriveReportDialog)
  assert pushed[0].history_mode is True


def test_mini_hud_renders_across_ar_3d_and_vr_modes(mock_raylib, monkeypatch):
  rendered_modes = []

  view = augmented_road_view.AugmentedRoadView.__new__(augmented_road_view.AugmentedRoadView)
  view.close = lambda: None
  view._refresh_plot_mode = lambda _t: None
  view._update_drive_report = lambda: None
  view._render_diagnostics = NS(start=lambda: None, call=lambda name, fn, *a, **kw: fn(*a, **kw), finish=lambda: None)
  view._suppress_camera_for_cluster = False
  view._switch_stream_if_needed = lambda _sm: None
  view._update_calibration = lambda: None
  view.is_view_3d_night_mode = lambda: False
  view._render_3d_view_background = lambda *a, **kw: None
  view._overlay_state = NS(update=lambda *a, **kw: None)
  view.client = None
  monkeypatch.setattr(augmented_road_view.CameraView, "stream_type", 0)
  view.model_renderer = NS(set_view_mode=lambda *a, **kw: None, render=lambda *a, **kw: None)
  view._hud_renderer = NS(set_view_3d_theme=lambda *a, **kw: None, render=lambda *a, **kw: None)
  view.alert_renderer = NS(render=lambda *a, **kw: None, get_alert=lambda _sm: None)
  view.driver_state_renderer = NS(draw_onroad=lambda *a, **kw: None)
  view._draw_border_carrot = lambda *a, **kw: None
  view._pm = NS(send=lambda *a, **kw: None)
  view._plot_mode = 0
  view._drive_report_tracker = DriveReportTracker()
  view._drive_report_dialog = None
  view._drive_report_mini_hud = DriveReportMiniHud()

  orig_render = view._drive_report_mini_hud.render

  def _spy_render(rect, stats, now_t=None):
    rendered_modes.append(view._view_3d_button.view_mode)
    return orig_render(rect, stats, now_t=0.0)

  view._drive_report_mini_hud.render = _spy_render

  monkeypatch.setattr(ui_state, "started", True)
  monkeypatch.setattr(ui_state, "sm", {})
  monkeypatch.setattr(augmented_road_view, "draw_vr_background", lambda *a, **kw: None)
  monkeypatch.setattr(augmented_road_view.CameraView, "_render", lambda *a, **kw: None)
  monkeypatch.setattr(augmented_road_view.messaging, "new_message", lambda *a, **kw: NS(uiDebug=NS()))
  monkeypatch.setattr(rl, "begin_scissor_mode", lambda *a: None)
  monkeypatch.setattr(rl, "end_scissor_mode", lambda: None)
  monkeypatch.setattr(gui_app, "is_recording", lambda: False)

  for vm in (0, 1, 2):  # 0=AR, 1=3D, 2=VR
    view._view_3d_button = NS(view_mode=vm, view_3d_mode=(vm == 1), render=lambda *a, **kw: None)
    view._render(rl.Rectangle(0, 0, 2160, 1080))

  assert rendered_modes == [0, 1, 2]



"""Comma 3 / 3X (2160x1080, 2:1) Driving Quality Report modal & live mini HUD."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import pyray as rl

from openpilot.selfdrive.carrot.drive_report import (
  DriveReportStore,
  format_duration_compact_ko,
  format_duration_count_ko,
  format_duration_ko,
  normalize_gear_shifter,
)
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import (
  FontWeight,
  MousePos,
  TextAlignment,
  gui_app,
)
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.label import gui_label

BASE_W = 2160.0
BASE_H = 1080.0

ICON_AUTO = "icons/drive_report_auto.png"
ICON_STEER = "icons/drive_report_steer.png"
ICON_GAS = "icons/drive_report_gas.png"
ICON_BRAKE = "icons/drive_report_brake.png"

# Approved 3-color semantic system matching the Parking report simulation
COLOR_AUTO = rl.Color(34, 197, 94, 255)        # 자동주행 (Green #22C55E)
COLOR_STEER = rl.Color(245, 158, 11, 255)      # 수동조향 (Amber #F59E0B)
COLOR_PEDAL = rl.Color(59, 130, 246, 255)      # 수동페달 (Blue #3B82F6)
COLOR_PEDAL_LIGHT = rl.Color(96, 165, 250, 255)
COLOR_BRAKE_ACTIVE = rl.Color(239, 68, 68, 255)
COLOR_DANGER = rl.Color(239, 68, 68, 255)

BG_BACKDROP = rl.Color(7, 11, 20, 224)
BG_PANEL = rl.Color(17, 24, 39, 252)
BG_RAIL = rl.Color(13, 19, 32, 255)
BG_CARD = rl.Color(26, 36, 59, 255)
BG_CARD_ALT = rl.Color(22, 31, 51, 255)
BORDER_SUBTLE = rl.Color(47, 63, 94, 255)
BORDER_ACTIVE = rl.Color(59, 130, 246, 255)

TEXT_PRIMARY = rl.Color(241, 245, 249, 255)
TEXT_MUTED = rl.Color(148, 163, 184, 255)
TEXT_DIM = rl.Color(100, 116, 139, 255)

BTN_DEFAULT = rl.Color(38, 52, 82, 255)
BTN_PRIMARY = rl.Color(37, 99, 235, 255)
BTN_DISABLED = rl.Color(42, 49, 66, 255)

HISTORY_PER_PAGE = 5

PAGE_TITLES = (
  "1. 자동주행 vs 수동 제어 비율",
  "2. 주행 속도 및 승차감 품질",
  "3. 급정거 분석 · 로컬 저장",
)

NAV_TABS = (
  ("1. 제어 비율", "비율·시간"),
  ("2. 속도·승차감", "품질점수"),
  ("3. 급정거·저장", "저장기록"),
)


def _score_color(score: int) -> rl.Color:
  if score >= 90:
    return COLOR_AUTO
  if score >= 75:
    return COLOR_STEER
  return COLOR_DANGER


class _DriveIconRenderer:
  """Shared texture cache and helper for drive-report icons."""

  def __init__(self):
    self._icon_cache: dict[str, Any] = {}

  def get_texture(self, asset_path: str):
    if asset_path in self._icon_cache:
      return self._icon_cache[asset_path]
    tex = None
    try:
      tex = gui_app.texture(asset_path, 128, 128)
    except Exception:
      tex = None
    self._icon_cache[asset_path] = tex
    return tex

  def draw(
    self,
    asset_path: str,
    x: float,
    y: float,
    size: float,
    fallback_color: rl.Color,
    tint: rl.Color = rl.WHITE,
  ) -> None:
    tex = self.get_texture(asset_path)
    if tex is not None and getattr(tex, "id", 0):
      src = rl.Rectangle(0.0, 0.0, float(tex.width), float(tex.height))
      dst = rl.Rectangle(float(x), float(y), float(size), float(size))
      rl.draw_texture_pro(tex, src, dst, rl.Vector2(0.0, 0.0), 0.0, tint)
    else:
      r = size * 0.5
      fb = rl.Color(fallback_color.r, fallback_color.g, fallback_color.b, tint.a)
      rl.draw_circle(int(x + r), int(y + r), r * 0.72, fb)


class DriveReportMiniHud:
  """Bottom-right real-time driving control breakdown overlay shown onroad."""

  _ROW_SPECS = (
    ("auto", "자동주행", ICON_AUTO, COLOR_AUTO),
    ("steer", "수동조향", ICON_STEER, COLOR_STEER),
    ("gas", "수동가속", ICON_GAS, COLOR_PEDAL),
    ("brake", "수동감속", ICON_BRAKE, COLOR_BRAKE_ACTIVE),
  )

  def __init__(self):
    self._icons = _DriveIconRenderer()
    self.rect = rl.Rectangle(0, 0, 0, 0)
    self._last_draw_state: list[tuple[str, str, bool, int]] = []

  def render(
    self,
    viewport: rl.Rectangle,
    live_stats: dict[str, Any] | None,
    now_t: float | None = None,
  ) -> rl.Rectangle:
    if viewport.width < 500 or viewport.height < 360:
      self.rect = rl.Rectangle(0, 0, 0, 0)
      return self.rect

    s = max(0.35, min(viewport.width / BASE_W, viewport.height / BASE_H))
    box_w = min(viewport.width * 0.38, 700.0 * s)
    box_h = min(viewport.height * 0.36, 288.0 * s)
    margin_r = 24.0 * s
    margin_b = 20.0 * s

    x = viewport.x + viewport.width - box_w - margin_r
    y = viewport.y + viewport.height - box_h - margin_b
    self.rect = rl.Rectangle(x, y, box_w, box_h)

    rl.draw_rectangle_rounded(self.rect, 0.12, 10, rl.Color(11, 17, 30, 210))
    rl.draw_rectangle_rounded_lines_ex(
      self.rect, 0.12, 10, max(1.0, 1.5 * s), rl.Color(62, 82, 120, 210)
    )

    stats = live_stats or {}
    by_key = {item.get("key"): item for item in (stats.get("items") or ()) if isinstance(item, dict)}

    now = time.monotonic() if now_t is None else float(now_t)
    pulse = 0.35 + 0.65 * (0.5 + 0.5 * math.sin(now * 9.0))

    pad_x = 18.0 * s
    pad_y = 14.0 * s
    inner_h = box_h - 2.0 * pad_y
    row_gap = 6.0 * s
    row_h = (inner_h - 3.0 * row_gap) / 4.0
    icon_sz = min(row_h - 6.0 * s, 48.0 * s)

    self._last_draw_state = []
    for idx, (key, default_label, icon_path, active_col) in enumerate(self._ROW_SPECS):
      item = by_key.get(key) or {}
      label = str(item.get("label") or default_label)
      sec = float(item.get("seconds") or 0.0)
      time_text = str(item.get("time_text") or format_duration_compact_ko(sec))
      pct = float(item.get("pct") or 0.0)
      active = bool(item.get("active", False))

      row_y = y + pad_y + idx * (row_h + row_gap)
      row_rect = rl.Rectangle(x + pad_x * 0.6, row_y, box_w - pad_x * 1.2, row_h)

      if active:
        row_bg = rl.Color(active_col.r, active_col.g, active_col.b, int(34 + 34 * pulse))
        rl.draw_rectangle_rounded(row_rect, 0.32, 6, row_bg)

      icon_x = x + pad_x
      icon_y = row_y + (row_h - icon_sz) * 0.5

      if active:
        halo_col = rl.Color(active_col.r, active_col.g, active_col.b, int(115 + 140 * pulse))
        rl.draw_circle(
          int(icon_x + icon_sz * 0.5),
          int(icon_y + icon_sz * 0.5),
          icon_sz * 0.60,
          halo_col,
        )
        tint_alpha = int(round(170 + 85 * pulse))
        tint = rl.Color(255, 255, 255, tint_alpha)
        fb_col = active_col
        text_col = TEXT_PRIMARY
        font_sz = max(18, int(round(38 * s)))
        weight = FontWeight.BOLD
      else:
        rl.draw_circle(
          int(icon_x + icon_sz * 0.5),
          int(icon_y + icon_sz * 0.5),
          icon_sz * 0.50,
          rl.Color(75, 85, 99, 65),
        )
        tint_alpha = 105
        tint = rl.Color(125, 132, 145, tint_alpha)
        fb_col = rl.Color(110, 118, 130, 120)
        text_col = rl.Color(145, 155, 172, 165)
        font_sz = max(17, int(round(35 * s)))
        weight = FontWeight.MEDIUM

      self._icons.draw(icon_path, icon_x, icon_y, icon_sz, fb_col, tint=tint)

      row_text = f"{label} {time_text}({pct:.1f}%)"
      text_x = icon_x + icon_sz + 16.0 * s
      text_w = max(40.0, (x + box_w - pad_x) - text_x)
      gui_label(
        rl.Rectangle(text_x, row_y, text_w, row_h),
        row_text,
        font_sz,
        text_col,
        weight,
        TextAlignment.LEFT,
      )
      self._last_draw_state.append((key, row_text, active, tint_alpha))

    return self.rect


class DriveReportDialog(Widget):
  """No-scroll 3-page widescreen (2160x1080) driving quality report modal + saved history browser."""

  def __init__(
    self,
    report: dict[str, Any] | None = None,
    on_close: Callable[[], None] | None = None,
    store: DriveReportStore | None = None,
    initial_history: bool = False,
  ):
    super().__init__()
    self._store = store if store is not None else DriveReportStore()
    self._report: dict[str, Any] = dict(report or {})
    self._on_close = on_close
    self._page = 0
    self._closed = False
    self._history_mode = bool(initial_history)
    self._history_page = 0
    self._history_entries: list[dict[str, Any]] = []
    self._history_row_rects: list[tuple[rl.Rectangle, str]] = []
    self._refresh_history()

    if not self._report and self._history_entries:
      first_id = str(self._history_entries[0].get("report_id") or "")
      loaded = self._store.get_report(first_id) if first_id else None
      if loaded is not None:
        self._report = loaded

    self._tab_rects: list[rl.Rectangle] = [rl.Rectangle(0, 0, 0, 0) for _ in range(3)]
    self._btn_history = rl.Rectangle(0, 0, 0, 0)
    self._btn_page3_history = rl.Rectangle(0, 0, 0, 0)
    self._btn_close_x = rl.Rectangle(0, 0, 0, 0)
    self._btn_prev = rl.Rectangle(0, 0, 0, 0)
    self._btn_next = rl.Rectangle(0, 0, 0, 0)
    self._btn_done = rl.Rectangle(0, 0, 0, 0)
    self._icons = _DriveIconRenderer()

  def _refresh_history(self) -> None:
    try:
      self._history_entries = self._store.list_reports(limit=100, include_full=False)
    except Exception:
      self._history_entries = []

  def _draw_icon(self, asset_path: str, x: float, y: float, size: float, fallback_color: rl.Color) -> None:
    self._icons.draw(asset_path, x, y, size, fallback_color)

  @property
  def page(self) -> int:
    return self._page

  @property
  def is_closed(self) -> bool:
    return self._closed

  @property
  def history_mode(self) -> bool:
    return self._history_mode

  @property
  def history_page(self) -> int:
    return self._history_page

  def open_history(self) -> None:
    self._refresh_history()
    self._history_page = 0
    self._history_mode = True

  def close_history(self) -> None:
    self._history_mode = False

  def select_report(self, report_id: str) -> bool:
    loaded = self._store.get_report(report_id)
    if loaded is None:
      return False
    self._report = loaded
    self._history_mode = False
    self._page = 0
    return True

  def set_page(self, idx: int) -> None:
    self._history_mode = False
    self._page = max(0, min(2, int(idx)))

  def next_page(self) -> None:
    if self._history_mode:
      total_pages = max(1, (len(self._history_entries) + HISTORY_PER_PAGE - 1) // HISTORY_PER_PAGE)
      self._history_page = min(total_pages - 1, self._history_page + 1)
    else:
      self.set_page(self._page + 1)

  def prev_page(self) -> None:
    if self._history_mode:
      self._history_page = max(0, self._history_page - 1)
    else:
      self.set_page(self._page - 1)

  def close_dialog(self) -> None:
    if self._closed:
      return
    self._closed = True
    try:
      if hasattr(gui_app, "widget_in_stack") and gui_app.widget_in_stack(self):
        nav_stack = getattr(gui_app, "_nav_stack", None)
        if isinstance(nav_stack, list) and self in nav_stack:
          idx = nav_stack.index(self)
          if idx > 0:
            gui_app.pop_widget(idx)
        else:
          gui_app.pop_widget()
      elif not hasattr(gui_app, "widget_in_stack"):
        gui_app.pop_widget()
    except Exception:
      pass
    if self._on_close is not None:
      cb = self._on_close
      self._on_close = None
      try:
        cb()
      except Exception:
        pass

  def _should_auto_close_for_gear(self) -> bool:
    try:
      if not ui_state.started:
        return False
      sm = ui_state.sm
      if sm is None or not sm.valid.get("carState", False):
        return False
      gear = normalize_gear_shifter(getattr(sm["carState"], "gearShifter", None))
      return gear not in ("park", "unknown")
    except Exception:
      return False

  def _update_state(self) -> None:
    if not self._closed and self._should_auto_close_for_gear():
      self.close_dialog()

  def _handle_keyboard(self) -> None:
    if rl.is_key_pressed(rl.KeyboardKey.KEY_RIGHT):
      self.next_page()
    elif rl.is_key_pressed(rl.KeyboardKey.KEY_LEFT):
      self.prev_page()
    elif rl.is_key_pressed(rl.KeyboardKey.KEY_ESCAPE) or rl.is_key_pressed(rl.KeyboardKey.KEY_ENTER):
      self.close_dialog()

  def _handle_mouse_release(self, mouse_pos: MousePos) -> None:
    if self._closed:
      return
    for idx, tab_rect in enumerate(self._tab_rects):
      if tab_rect.width > 0 and rl.check_collision_point_rec(mouse_pos, tab_rect):
        self.set_page(idx)
        return
    if self._btn_history.width > 0 and rl.check_collision_point_rec(mouse_pos, self._btn_history):
      if self._history_mode:
        self.close_history()
      else:
        self.open_history()
      return
    if not self._history_mode and self._page == 2 and self._btn_page3_history.width > 0 and rl.check_collision_point_rec(mouse_pos, self._btn_page3_history):
      self.open_history()
      return
    if self._history_mode:
      for row_rect, rep_id in self._history_row_rects:
        if row_rect.width > 0 and rl.check_collision_point_rec(mouse_pos, row_rect):
          self.select_report(rep_id)
          return
    if self._btn_close_x.width > 0 and rl.check_collision_point_rec(mouse_pos, self._btn_close_x):
      self.close_dialog()
      return
    if self._btn_done.width > 0 and rl.check_collision_point_rec(mouse_pos, self._btn_done):
      self.close_dialog()
      return
    if self._btn_prev.width > 0 and rl.check_collision_point_rec(mouse_pos, self._btn_prev):
      self.prev_page()
      return
    if self._btn_next.width > 0 and rl.check_collision_point_rec(mouse_pos, self._btn_next):
      self.next_page()
      return

  def _render(self, rect: rl.Rectangle) -> None:
    self._handle_keyboard()

    rl.draw_rectangle_rec(rect, BG_BACKDROP)

    scale = min(rect.width / BASE_W, rect.height / BASE_H)
    scale = max(scale, 0.2)

    panel_w = 2000.0 * scale
    panel_h = 940.0 * scale
    px = rect.x + (rect.width - panel_w) / 2.0
    py = rect.y + (rect.height - panel_h) / 2.0
    panel = rl.Rectangle(px, py, panel_w, panel_h)

    rl.draw_rectangle_rounded(panel, 0.03, 12, BG_PANEL)
    rl.draw_rectangle_rounded_lines_ex(panel, 0.03, 12, max(1.0, 2.0 * scale), BORDER_SUBTLE)

    # Split: Left Rail (500) + Right Stage (1460) inside panel
    rail_w = 500.0 * scale
    rail = rl.Rectangle(px, py, rail_w, panel_h)
    rl.draw_rectangle_rounded(rail, 0.03, 12, BG_RAIL)
    rl.draw_line_ex(
      rl.Vector2(px + rail_w, py + 16 * scale),
      rl.Vector2(px + rail_w, py + panel_h - 16 * scale),
      max(1.0, 2.0 * scale),
      BORDER_SUBTLE,
    )

    stage_x = px + rail_w + 36.0 * scale
    stage_w = panel_w - rail_w - 72.0 * scale
    stage_y = py + 28.0 * scale
    stage_h = panel_h - 56.0 * scale

    self._render_left_rail(rail, scale)
    self._render_right_stage(rl.Rectangle(stage_x, stage_y, stage_w, stage_h), scale)

  def _render_left_rail(self, rail: rl.Rectangle, s: float) -> None:
    pad = 26.0 * s
    rx = rail.x + pad
    rw = rail.width - 2.0 * pad

    # P badge + Header
    p_badge = rl.Rectangle(rx, rail.y + 26.0 * s, 74.0 * s, 74.0 * s)
    rl.draw_rectangle_rounded(p_badge, 0.24, 8, BTN_PRIMARY)
    gui_label(p_badge, "P", int(40 * s), rl.WHITE, FontWeight.BOLD, TextAlignment.CENTER)

    distance_km = float(self._report.get("distance_km") or 0.0)
    duration_text = str(self._report.get("duration_text") or format_duration_ko(float(self._report.get("duration_s") or 0.0)))
    gui_label(
      rl.Rectangle(rx + 88.0 * s, rail.y + 22.0 * s, rw - 88.0 * s, 42.0 * s),
      "주행 품질 리포트",
      int(34 * s),
      TEXT_PRIMARY,
      FontWeight.BOLD,
      TextAlignment.LEFT,
    )
    gui_label(
      rl.Rectangle(rx + 88.0 * s, rail.y + 66.0 * s, rw - 88.0 * s, 36.0 * s),
      f"{distance_km:.1f}km · {duration_text}" if self._report else "저장 기록을 선택하세요",
      int(26 * s),
      TEXT_MUTED,
      FontWeight.MEDIUM,
      TextAlignment.LEFT,
    )

    # Score Card + Ring Gauge
    score_box = rl.Rectangle(rx, rail.y + 120.0 * s, rw, 224.0 * s)
    rl.draw_rectangle_rounded(score_box, 0.12, 10, BG_CARD)
    rl.draw_rectangle_rounded_lines_ex(score_box, 0.12, 10, max(1.0, 1.5 * s), BORDER_SUBTLE)

    score = int(self._report.get("score") or 0)
    grade = str(self._report.get("grade") or "-")
    hb_total = int((self._report.get("hard_brakes") or {}).get("total_count") or 0)
    ring_col = _score_color(score) if self._report else BORDER_SUBTLE

    cx = score_box.x + 88.0 * s
    cy = score_box.y + score_box.height / 2.0
    center = rl.Vector2(cx, cy)
    r_in = 52.0 * s
    r_out = 68.0 * s
    rl.draw_ring(center, r_in, r_out, 0.0, 360.0, 48, rl.Color(39, 52, 82, 255))
    sweep = max(0.0, min(360.0, (score / 100.0) * 360.0)) if self._report else 0.0
    if sweep > 0.5:
      rl.draw_ring(center, r_in, r_out, -90.0, -90.0 + sweep, 48, ring_col)

    gui_label(
      rl.Rectangle(cx - 52.0 * s, cy - 32.0 * s, 104.0 * s, 64.0 * s),
      str(score) if self._report else "--",
      int(46 * s),
      TEXT_PRIMARY,
      FontWeight.BOLD,
      TextAlignment.CENTER,
    )

    info_x = score_box.x + 174.0 * s
    info_w = score_box.width - 188.0 * s
    gui_label(
      rl.Rectangle(info_x, score_box.y + 34.0 * s, info_w, 34.0 * s),
      "종합 주행 점수",
      int(26 * s),
      TEXT_MUTED,
      FontWeight.MEDIUM,
      TextAlignment.LEFT,
    )
    gui_label(
      rl.Rectangle(info_x, score_box.y + 74.0 * s, info_w, 54.0 * s),
      f"Grade {grade}",
      int(42 * s),
      ring_col,
      FontWeight.BOLD,
      TextAlignment.LEFT,
    )
    gui_label(
      rl.Rectangle(info_x, score_box.y + 138.0 * s, info_w, 38.0 * s),
      f"급정거 {hb_total}회",
      int(28 * s),
      TEXT_PRIMARY if hb_total == 0 else COLOR_STEER,
      FontWeight.MEDIUM,
      TextAlignment.LEFT,
    )

    # 3 Vertical Page Tabs
    tab_y = rail.y + 366.0 * s
    tab_h = 108.0 * s
    tab_gap = 14.0 * s
    for idx, (title, sub) in enumerate(NAV_TABS):
      y = tab_y + idx * (tab_h + tab_gap)
      rect = rl.Rectangle(rx, y, rw, tab_h)
      self._tab_rects[idx] = rect
      active = (not self._history_mode) and (idx == self._page)
      fill = BTN_PRIMARY if active else BG_CARD
      border = COLOR_PEDAL_LIGHT if active else BORDER_SUBTLE
      rl.draw_rectangle_rounded(rect, 0.16, 8, fill)
      rl.draw_rectangle_rounded_lines_ex(rect, 0.16, 8, max(1.0, 2.0 * s if active else 1.5 * s), border)

      gui_label(
        rl.Rectangle(rect.x + 20.0 * s, rect.y + 14.0 * s, rect.width - 156.0 * s, 44.0 * s),
        title,
        int(31 * s),
        rl.WHITE if active else TEXT_PRIMARY,
        FontWeight.BOLD,
        TextAlignment.LEFT,
      )
      gui_label(
        rl.Rectangle(rect.x + 20.0 * s, rect.y + 60.0 * s, rect.width - 40.0 * s, 34.0 * s),
        f"페이지 {idx + 1} / 3",
        int(24 * s),
        rl.Color(219, 234, 254, 255) if active else TEXT_MUTED,
        FontWeight.NORMAL,
        TextAlignment.LEFT,
      )
      gui_label(
        rl.Rectangle(rect.x + rect.width - 148.0 * s, rect.y + 16.0 * s, 128.0 * s, 40.0 * s),
        sub,
        int(24 * s),
        rl.Color(219, 234, 254, 255) if active else TEXT_DIM,
        FontWeight.SEMI_BOLD,
        TextAlignment.RIGHT,
      )

    # Bottom Saved Report History Toggle Button
    self._btn_history = rl.Rectangle(rx, rail.y + rail.height - 96.0 * s, rw, 68.0 * s)
    if self._history_mode:
      fill_col = BTN_PRIMARY
      border_col = COLOR_PEDAL_LIGHT
      txt_col = rl.WHITE
      btn_txt = "현재 리포트 보기"
    else:
      fill_col = rl.Color(20, 62, 42, 255)
      border_col = COLOR_AUTO
      txt_col = COLOR_AUTO
      btn_txt = f"저장 목록 ({len(self._history_entries)}건)"
    rl.draw_rectangle_rounded(self._btn_history, 0.2, 8, fill_col)
    rl.draw_rectangle_rounded_lines_ex(self._btn_history, 0.2, 8, max(1.0, 2.0 * s), border_col)
    gui_label(
      rl.Rectangle(self._btn_history.x + 14.0 * s, self._btn_history.y, self._btn_history.width - 28.0 * s, self._btn_history.height),
      btn_txt,
      int(27 * s),
      txt_col,
      FontWeight.BOLD,
      TextAlignment.CENTER,
    )

  def _render_right_stage(self, stage: rl.Rectangle, s: float) -> None:
    # Header row
    header_h = 74.0 * s
    if self._history_mode:
      title_text = "저장된 주행 리포트 목록"
      right_meta = f"총 {len(self._history_entries)}건"
    else:
      title_text = PAGE_TITLES[self._page]
      right_meta = str(self._report.get("time_range_label") or "")

    gui_label(
      rl.Rectangle(stage.x, stage.y, 760.0 * s, header_h),
      title_text,
      int(38 * s),
      TEXT_PRIMARY,
      FontWeight.BOLD,
      TextAlignment.LEFT,
    )
    gui_label(
      rl.Rectangle(stage.x + 690.0 * s, stage.y, 466.0 * s, header_h),
      right_meta,
      int(26 * s),
      TEXT_MUTED,
      FontWeight.MEDIUM,
      TextAlignment.RIGHT,
    )

    dr_pill = rl.Rectangle(stage.x + stage.width - 256.0 * s, stage.y + 12.0 * s, 172.0 * s, 50.0 * s)
    rl.draw_rectangle_rounded(dr_pill, 0.25, 8, BG_CARD)
    rl.draw_rectangle_rounded_lines_ex(dr_pill, 0.25, 8, max(1.0, 1.5 * s), BORDER_SUBTLE)
    gui_label(dr_pill, "D/R 자동닫힘", int(22 * s), TEXT_MUTED, FontWeight.MEDIUM, TextAlignment.CENTER)

    self._btn_close_x = rl.Rectangle(stage.x + stage.width - 68.0 * s, stage.y + 8.0 * s, 68.0 * s, 58.0 * s)
    rl.draw_rectangle_rounded(self._btn_close_x, 0.22, 8, BTN_DEFAULT)
    gui_label(self._btn_close_x, "X", int(32 * s), TEXT_PRIMARY, FontWeight.BOLD, TextAlignment.CENTER)

    body_y = stage.y + header_h + 14.0 * s
    footer_h = 84.0 * s
    body_h = stage.height - header_h - footer_h - 32.0 * s
    body = rl.Rectangle(stage.x, body_y, stage.width, body_h)

    if self._history_mode:
      self._render_history_list(body, s)
      total_pages = max(1, (len(self._history_entries) + HISTORY_PER_PAGE - 1) // HISTORY_PER_PAGE)
      cur_page = self._history_page
    else:
      self._history_row_rects = []
      if self._page == 0:
        self._render_page_controls(body, s)
      elif self._page == 1:
        self._render_page_speed(body, s)
      else:
        self._render_page_hard_brake(body, s)
      total_pages = 3
      cur_page = self._page

    # Footer navigation bar
    footer_y = stage.y + stage.height - footer_h
    gui_label(
      rl.Rectangle(stage.x, footer_y, 170.0 * s, footer_h),
      f"{cur_page + 1} / {total_pages}",
      int(30 * s),
      TEXT_MUTED,
      FontWeight.BOLD,
      TextAlignment.LEFT,
    )

    dot_y = footer_y + footer_h / 2.0
    max_dots = min(6, total_pages)
    for idx in range(max_dots):
      dot_x = stage.x + 165.0 * s + idx * 34.0 * s
      active_dot = idx == cur_page if total_pages <= 6 else idx == min(5, cur_page)
      rl.draw_circle(
        int(dot_x),
        int(dot_y),
        9.0 * s if active_dot else 6.0 * s,
        COLOR_PEDAL_LIGHT if active_dot else BORDER_SUBTLE,
      )

    btn_h = 74.0 * s
    btn_y = footer_y + (footer_h - btn_h) / 2.0
    self._btn_done = rl.Rectangle(stage.x + stage.width - 260.0 * s, btn_y, 260.0 * s, btn_h)
    self._btn_next = rl.Rectangle(self._btn_done.x - 210.0 * s, btn_y, 190.0 * s, btn_h)
    self._btn_prev = rl.Rectangle(self._btn_next.x - 210.0 * s, btn_y, 190.0 * s, btn_h)

    prev_enabled = cur_page > 0
    next_enabled = cur_page < (total_pages - 1)
    rl.draw_rectangle_rounded(self._btn_prev, 0.2, 8, BTN_DEFAULT if prev_enabled else BTN_DISABLED)
    gui_label(self._btn_prev, "< 이전", int(28 * s),
              TEXT_PRIMARY if prev_enabled else TEXT_DIM, FontWeight.BOLD, TextAlignment.CENTER)

    rl.draw_rectangle_rounded(self._btn_next, 0.2, 8, BTN_DEFAULT if next_enabled else BTN_DISABLED)
    gui_label(self._btn_next, "다음 >", int(28 * s),
              TEXT_PRIMARY if next_enabled else TEXT_DIM, FontWeight.BOLD, TextAlignment.CENTER)

    rl.draw_rectangle_rounded(self._btn_done, 0.2, 8, BTN_PRIMARY)
    gui_label(self._btn_done, "확인 · 닫기", int(30 * s), rl.WHITE, FontWeight.BOLD, TextAlignment.CENTER)

  def _render_history_list(self, body: rl.Rectangle, s: float) -> None:
    self._history_row_rects = []
    self._btn_page3_history = rl.Rectangle(0, 0, 0, 0)

    if not self._history_entries:
      empty_card = rl.Rectangle(body.x, body.y, body.width, body.height)
      rl.draw_rectangle_rounded(empty_card, 0.08, 10, BG_CARD)
      rl.draw_rectangle_rounded_lines_ex(empty_card, 0.08, 10, max(1.0, 1.5 * s), BORDER_SUBTLE)
      gui_label(
        rl.Rectangle(body.x + 40.0 * s, body.y + body.height * 0.4, body.width - 80.0 * s, 60.0 * s),
        "단말에 저장된 주행 리포트가 없습니다.",
        int(34 * s),
        TEXT_MUTED,
        FontWeight.BOLD,
        TextAlignment.CENTER,
      )
      return

    total_pages = max(1, (len(self._history_entries) + HISTORY_PER_PAGE - 1) // HISTORY_PER_PAGE)
    self._history_page = max(0, min(total_pages - 1, self._history_page))
    start = self._history_page * HISTORY_PER_PAGE
    visible = self._history_entries[start:start + HISTORY_PER_PAGE]

    gap = 14.0 * s
    row_h = (body.height - (HISTORY_PER_PAGE - 1) * gap) / HISTORY_PER_PAGE
    current_id = str(self._report.get("report_id") or "")

    for idx, item in enumerate(visible):
      rep_id = str(item.get("report_id") or "")
      row = rl.Rectangle(body.x, body.y + idx * (row_h + gap), body.width, row_h)
      self._history_row_rects.append((row, rep_id))

      is_cur = bool(current_id and rep_id == current_id)
      rl.draw_rectangle_rounded(row, 0.14, 10, rl.Color(30, 48, 82, 255) if is_cur else BG_CARD)
      rl.draw_rectangle_rounded_lines_ex(
        row, 0.14, 10, max(1.0, 2.0 * s if is_cur else 1.5 * s),
        COLOR_PEDAL_LIGHT if is_cur else BORDER_SUBTLE,
      )

      sc = int(item.get("score") or 0)
      gr = str(item.get("grade") or "-")
      sc_col = _score_color(sc)
      pill = rl.Rectangle(row.x + 22.0 * s, row.y + (row.height - 76.0 * s) * 0.5, 168.0 * s, 76.0 * s)
      rl.draw_rectangle_rounded(pill, 0.24, 8, BG_RAIL)
      gui_label(pill, f"{sc}점 · {gr}", int(32 * s), sc_col, FontWeight.BOLD, TextAlignment.CENTER)

      time_lbl = str(item.get("time_range_label") or rep_id)
      dist_km = float(item.get("distance_km") or 0.0)
      dur_txt = str(item.get("duration_text") or format_duration_ko(float(item.get("duration_s") or 0.0)))
      veh = str(item.get("vehicle_name") or "")

      tx = pill.x + pill.width + 24.0 * s
      tw = row.width - (tx - row.x) - 240.0 * s
      gui_label(
        rl.Rectangle(tx, row.y + 18.0 * s, tw, 44.0 * s),
        time_lbl,
        int(33 * s),
        TEXT_PRIMARY,
        FontWeight.BOLD,
        TextAlignment.LEFT,
      )
      sub_parts = [f"{dist_km:.1f}km", dur_txt]
      if veh:
        sub_parts.append(veh)
      gui_label(
        rl.Rectangle(tx, row.y + 68.0 * s, tw, 38.0 * s),
        " · ".join(sub_parts),
        int(27 * s),
        TEXT_MUTED,
        FontWeight.MEDIUM,
        TextAlignment.LEFT,
      )

      action_rect = rl.Rectangle(row.x + row.width - 216.0 * s, row.y + (row.height - 68.0 * s) * 0.5, 192.0 * s, 68.0 * s)
      rl.draw_rectangle_rounded(action_rect, 0.22, 8, BTN_PRIMARY if not is_cur else rl.Color(20, 62, 42, 255))
      gui_label(
        action_rect,
        "현재 표시중" if is_cur else "불러오기 >",
        int(27 * s),
        rl.WHITE if not is_cur else COLOR_AUTO,
        FontWeight.BOLD,
        TextAlignment.CENTER,
      )

  def _render_page_controls(self, body: rl.Rectangle, s: float) -> None:
    ratios = self._report.get("ratios") or {}
    auto_pct = int(ratios.get("auto_pct") or 0)
    steer_pct = int(ratios.get("steer_pct") or 0)
    pedal_pct = int(ratios.get("pedal_pct") or 0)

    # Top Ratio Bar Card
    bar_card = rl.Rectangle(body.x, body.y, body.width, 224.0 * s)
    rl.draw_rectangle_rounded(bar_card, 0.08, 10, BG_CARD)
    rl.draw_rectangle_rounded_lines_ex(bar_card, 0.08, 10, max(1.0, 1.5 * s), BORDER_SUBTLE)

    cx = bar_card.x + 26.0 * s
    cw = bar_card.width - 52.0 * s
    gui_label(
      rl.Rectangle(cx, bar_card.y + 16.0 * s, 640.0 * s, 44.0 * s),
      "자동주행 vs 수동 제어 비율",
      int(34 * s),
      TEXT_PRIMARY,
      FontWeight.BOLD,
      TextAlignment.LEFT,
    )
    duration_text = str(self._report.get("duration_text") or format_duration_ko(float(self._report.get("duration_s") or 0.0)))
    moving_text = str(self._report.get("moving_text") or format_duration_ko(float(self._report.get("moving_s") or 0.0)))
    idle_text = str(self._report.get("idle_text") or format_duration_ko(float(self._report.get("idle_s") or 0.0)))
    gui_label(
      rl.Rectangle(cx, bar_card.y + 62.0 * s, cw, 34.0 * s),
      f"정차 제외 · 전체 {duration_text} (실주행 {moving_text} / 정차 {idle_text})",
      int(25 * s),
      TEXT_MUTED,
      FontWeight.MEDIUM,
      TextAlignment.LEFT,
    )
    gui_label(
      rl.Rectangle(bar_card.x + bar_card.width - 390.0 * s, bar_card.y + 14.0 * s, 364.0 * s, 48.0 * s),
      f"자동주행 {auto_pct}%",
      int(42 * s),
      COLOR_AUTO,
      FontWeight.BOLD,
      TextAlignment.RIGHT,
    )

    # Stacked progress bar
    seg_y = bar_card.y + 106.0 * s
    seg_h = 34.0 * s
    rl.draw_rectangle_rounded(rl.Rectangle(cx, seg_y, cw, seg_h), 0.45, 8, BG_RAIL)
    cur_x = cx
    total_pct = max(1, auto_pct + steer_pct + pedal_pct)
    for pct, color in ((auto_pct, COLOR_AUTO), (steer_pct, COLOR_STEER), (pedal_pct, COLOR_PEDAL)):
      if pct <= 0:
        continue
      seg_w = cw * (pct / total_pct)
      rl.draw_rectangle_rec(rl.Rectangle(cur_x, seg_y, seg_w, seg_h), color)
      cur_x += seg_w

    # Color-matched legend row with circular icons
    leg_y = bar_card.y + 154.0 * s
    leg_icon_sz = 38.0 * s
    legends = (
      (COLOR_AUTO, f"자동주행 {auto_pct}%", (ICON_AUTO,)),
      (COLOR_STEER, f"수동조향 {steer_pct}%", (ICON_STEER,)),
      (COLOR_PEDAL, f"수동페달 {pedal_pct}%", (ICON_GAS, ICON_BRAKE)),
    )
    leg_w = cw / 3.0
    for idx, (col, label, icons) in enumerate(legends):
      lx = cx + idx * leg_w
      rl.draw_circle(int(lx + 8.0 * s), int(leg_y + 24.0 * s), 8.0 * s, col)
      cur_icon_x = lx + 24.0 * s
      for icon_path in icons:
        self._draw_icon(icon_path, cur_icon_x, leg_y + 5.0 * s, leg_icon_sz, col)
        cur_icon_x += leg_icon_sz + 6.0 * s
      text_x = cur_icon_x + 6.0 * s
      gui_label(
        rl.Rectangle(text_x, leg_y + 2.0 * s, max(40.0 * s, lx + leg_w - text_x - 8.0 * s), 44.0 * s),
        label,
        int(32 * s),
        TEXT_PRIMARY,
        FontWeight.BOLD,
        TextAlignment.LEFT,
      )

    # 2x2 Breakdown Cards with enlarged two-row layout so no title/value clips
    controls = self._report.get("controls") or {}
    cards = (
      ("자동주행", "오파 활성 누적", controls.get("auto") or {}, COLOR_AUTO, ICON_AUTO),
      ("수동조향", "핸들 개입 누적", controls.get("steer") or {}, COLOR_STEER, ICON_STEER),
      ("수동가속", "가속 페달 개입", controls.get("gas") or {}, COLOR_PEDAL, ICON_GAS),
      ("수동감속", "브레이크 페달 개입", controls.get("brake") or {}, COLOR_BRAKE_ACTIVE, ICON_BRAKE),
    )

    grid_y = body.y + 244.0 * s
    grid_h = body.height - 244.0 * s
    gap_x = 24.0 * s
    gap_y = 20.0 * s
    card_w = (body.width - gap_x) / 2.0
    card_h = (grid_h - gap_y) / 2.0

    for idx, (title, sub, data, stripe_col, icon_path) in enumerate(cards):
      col_i = idx % 2
      row_i = idx // 2
      card = rl.Rectangle(
        body.x + col_i * (card_w + gap_x),
        grid_y + row_i * (card_h + gap_y),
        card_w,
        card_h,
      )
      rl.draw_rectangle_rounded(card, 0.1, 10, BG_CARD)
      rl.draw_rectangle_rounded_lines_ex(card, 0.1, 10, max(1.0, 1.5 * s), BORDER_SUBTLE)

      rl.draw_rectangle_rounded(
        rl.Rectangle(card.x + 18.0 * s, card.y + 22.0 * s, 10.0 * s, card.height - 44.0 * s),
        0.5,
        6,
        stripe_col,
      )
      icon_sz = 76.0 * s
      self._draw_icon(
        icon_path,
        card.x + 42.0 * s,
        card.y + (card.height - icon_sz) / 2.0,
        icon_sz,
        stripe_col,
      )
      tx = card.x + 136.0 * s
      tw = card.width - 164.0 * s
      gui_label(
        rl.Rectangle(tx, card.y + 20.0 * s, tw, 44.0 * s),
        title,
        int(36 * s),
        TEXT_PRIMARY,
        FontWeight.BOLD,
        TextAlignment.LEFT,
      )
      gui_label(
        rl.Rectangle(tx, card.y + 70.0 * s, tw, 36.0 * s),
        sub,
        int(27 * s),
        TEXT_MUTED,
        FontWeight.MEDIUM,
        TextAlignment.LEFT,
      )

      val_text = str(data.get("text") or format_duration_count_ko(float(data.get("seconds") or 0.0), int(data.get("count") or 0)))
      gui_label(
        rl.Rectangle(tx, card.y + card.height - 86.0 * s, tw, 64.0 * s),
        val_text,
        int(46 * s),
        TEXT_PRIMARY,
        FontWeight.BOLD,
        TextAlignment.RIGHT,
      )

  def _render_page_speed(self, body: rl.Rectangle, s: float) -> None:
    speed = self._report.get("speed") or {}
    avg_kph = float(speed.get("avg_kph") or 0.0)
    max_kph = float(speed.get("max_kph") or 0.0)
    long_score = int(speed.get("long_score") or 0)
    lat_score = int(speed.get("lat_score") or 0)
    long_jerk = float(speed.get("long_jerk_rms") or 0.0)
    lat_peak = float(speed.get("lat_accel_peak") or 0.0)

    moving_text = str(self._report.get("moving_text") or format_duration_ko(float(self._report.get("moving_s") or 0.0)))
    idle_text = str(self._report.get("idle_text") or format_duration_ko(float(self._report.get("idle_s") or 0.0)))

    gap = 24.0 * s
    top_h = 282.0 * s
    top_w = (body.width - 2.0 * gap) / 3.0

    metrics = (
      ("평균 주행 속도", f"{avg_kph:.1f}", "km/h", f"이동 {moving_text}", COLOR_PEDAL_LIGHT),
      ("최고 주행 속도", f"{max_kph:.1f}", "km/h", f"정차 {idle_text}", TEXT_PRIMARY),
    )
    for idx, (label, val, unit, sub, col) in enumerate(metrics):
      card = rl.Rectangle(body.x + idx * (top_w + gap), body.y, top_w, top_h)
      rl.draw_rectangle_rounded(card, 0.1, 10, BG_CARD)
      rl.draw_rectangle_rounded_lines_ex(card, 0.1, 10, max(1.0, 1.5 * s), BORDER_SUBTLE)
      gui_label(
        rl.Rectangle(card.x + 26.0 * s, card.y + 22.0 * s, card.width - 52.0 * s, 42.0 * s),
        label,
        int(32 * s),
        TEXT_MUTED,
        FontWeight.SEMI_BOLD,
        TextAlignment.LEFT,
      )
      gui_label(
        rl.Rectangle(card.x + 26.0 * s, card.y + 82.0 * s, card.width - 52.0 * s, 96.0 * s),
        f"{val} {unit}",
        int(64 * s),
        col,
        FontWeight.BOLD,
        TextAlignment.LEFT,
      )
      gui_label(
        rl.Rectangle(card.x + 26.0 * s, card.y + card.height - 62.0 * s, card.width - 52.0 * s, 42.0 * s),
        sub,
        int(32 * s),
        TEXT_MUTED,
        FontWeight.MEDIUM,
        TextAlignment.LEFT,
      )

    # Third top card: deductions summary
    ded_card = rl.Rectangle(body.x + 2.0 * (top_w + gap), body.y, top_w, top_h)
    rl.draw_rectangle_rounded(ded_card, 0.1, 10, BG_CARD)
    rl.draw_rectangle_rounded_lines_ex(ded_card, 0.1, 10, max(1.0, 1.5 * s), BORDER_SUBTLE)
    gui_label(
      rl.Rectangle(ded_card.x + 26.0 * s, ded_card.y + 22.0 * s, ded_card.width - 52.0 * s, 42.0 * s),
      "감점 요약",
      int(32 * s),
      TEXT_MUTED,
      FontWeight.SEMI_BOLD,
      TextAlignment.LEFT,
    )
    deductions = list(self._report.get("deductions") or [])
    if not deductions:
      gui_label(
        rl.Rectangle(ded_card.x + 26.0 * s, ded_card.y + 104.0 * s, ded_card.width - 52.0 * s, 80.0 * s),
        "감점 없음 (매우 안정적)",
        int(34 * s),
        COLOR_AUTO,
        FontWeight.BOLD,
        TextAlignment.LEFT,
      )
    else:
      for row_i, ded in enumerate(deductions[:3]):
        ry = ded_card.y + (80.0 + row_i * 62.0) * s
        reason = str(ded.get("reason") or "")
        pts = int(ded.get("points") or 0)
        gui_label(
          rl.Rectangle(ded_card.x + 26.0 * s, ry, ded_card.width - 134.0 * s, 48.0 * s),
          reason,
          int(28 * s),
          TEXT_PRIMARY,
          FontWeight.MEDIUM,
          TextAlignment.LEFT,
        )
        gui_label(
          rl.Rectangle(ded_card.x + ded_card.width - 122.0 * s, ry, 96.0 * s, 48.0 * s),
          f"{pts}점",
          int(32 * s),
          COLOR_STEER,
          FontWeight.BOLD,
          TextAlignment.RIGHT,
        )

    # Bottom row: 2 smoothness bars
    bot_y = body.y + top_h + gap
    bot_h = body.height - top_h - gap
    bot_w = (body.width - gap) / 2.0

    smooth_items = (
      ("가감속 부드러움 (종방향)", long_score, f"종방향 저크 RMS {long_jerk:.2f} m/s^3", "낮을수록 부드러운 가감속"),
      ("조향 쏠림 안정성 (횡방향)", lat_score, f"최대 횡가속도 {lat_peak:.2f} m/s^2", "코너링 안정성 지표"),
    )
    for idx, (label, sc, metric_txt, helper_txt) in enumerate(smooth_items):
      card = rl.Rectangle(body.x + idx * (bot_w + gap), bot_y, bot_w, bot_h)
      rl.draw_rectangle_rounded(card, 0.1, 10, BG_CARD)
      rl.draw_rectangle_rounded_lines_ex(card, 0.1, 10, max(1.0, 1.5 * s), BORDER_SUBTLE)

      bar_col = _score_color(sc)
      gui_label(
        rl.Rectangle(card.x + 28.0 * s, card.y + 28.0 * s, card.width - 196.0 * s, 50.0 * s),
        label,
        int(36 * s),
        TEXT_PRIMARY,
        FontWeight.BOLD,
        TextAlignment.LEFT,
      )
      gui_label(
        rl.Rectangle(card.x + card.width - 178.0 * s, card.y + 20.0 * s, 150.0 * s, 66.0 * s),
        f"{sc}점",
        int(54 * s),
        bar_col,
        FontWeight.BOLD,
        TextAlignment.RIGHT,
      )

      track = rl.Rectangle(card.x + 28.0 * s, card.y + 114.0 * s, card.width - 56.0 * s, 34.0 * s)
      rl.draw_rectangle_rounded(track, 0.5, 8, BG_RAIL)
      fill_w = max(12.0 * s, track.width * (max(0, min(100, sc)) / 100.0))
      rl.draw_rectangle_rounded(rl.Rectangle(track.x, track.y, fill_w, track.height), 0.5, 8, bar_col)

      gui_label(
        rl.Rectangle(card.x + 28.0 * s, card.y + 186.0 * s, card.width - 56.0 * s, 48.0 * s),
        metric_txt,
        int(34 * s),
        TEXT_PRIMARY,
        FontWeight.SEMI_BOLD,
        TextAlignment.LEFT,
      )
      gui_label(
        rl.Rectangle(card.x + 28.0 * s, card.y + 244.0 * s, card.width - 56.0 * s, 40.0 * s),
        helper_txt,
        int(27 * s),
        TEXT_MUTED,
        FontWeight.NORMAL,
        TextAlignment.LEFT,
      )

  def _render_page_hard_brake(self, body: rl.Rectangle, s: float) -> None:
    hb = self._report.get("hard_brakes") or {}
    auto_cnt = int(hb.get("auto_count") or 0)
    manual_cnt = int(hb.get("manual_count") or 0)
    total_cnt = int(hb.get("total_count") or (auto_cnt + manual_cnt))
    auto_peak = float(hb.get("auto_peak_decel") or 0.0)
    manual_peak = float(hb.get("manual_peak_decel") or 0.0)
    events = list(hb.get("events") or [])

    gap = 24.0 * s
    top_h = 412.0 * s
    left_w = 530.0 * s
    right_w = body.width - left_w - gap

    # Left card: split Auto vs Manual hard braking
    left_card = rl.Rectangle(body.x, body.y, left_w, top_h)
    rl.draw_rectangle_rounded(left_card, 0.08, 10, BG_CARD)
    rl.draw_rectangle_rounded_lines_ex(left_card, 0.08, 10, max(1.0, 1.5 * s), BORDER_SUBTLE)

    half_h = top_h / 2.0
    rl.draw_line_ex(
      rl.Vector2(left_card.x + 24.0 * s, left_card.y + half_h),
      rl.Vector2(left_card.x + left_card.width - 24.0 * s, left_card.y + half_h),
      max(1.0, 1.5 * s),
      BORDER_SUBTLE,
    )

    split_rows = (
      ("자동주행 급정거", auto_cnt, auto_peak, COLOR_AUTO if auto_cnt == 0 else COLOR_STEER, left_card.y),
      ("수동 급정거", manual_cnt, manual_peak, TEXT_PRIMARY if manual_cnt == 0 else COLOR_DANGER, left_card.y + half_h),
    )
    for title, cnt, peak, col, ry in split_rows:
      gui_label(
        rl.Rectangle(left_card.x + 28.0 * s, ry + 34.0 * s, 320.0 * s, 48.0 * s),
        title,
        int(36 * s),
        TEXT_PRIMARY,
        FontWeight.BOLD,
        TextAlignment.LEFT,
      )
      gui_label(
        rl.Rectangle(left_card.x + 28.0 * s, ry + 102.0 * s, 320.0 * s, 42.0 * s),
        f"최대 감속 {peak:.2f} m/s^2" if peak < -0.05 else "최대 감속 0.00 m/s^2",
        int(30 * s),
        TEXT_MUTED,
        FontWeight.MEDIUM,
        TextAlignment.LEFT,
      )
      gui_label(
        rl.Rectangle(left_card.x + left_card.width - 178.0 * s, ry + 36.0 * s, 150.0 * s, 114.0 * s),
        f"{cnt}회",
        int(64 * s),
        col,
        FontWeight.BOLD,
        TextAlignment.RIGHT,
      )

    # Right card: Event list
    right_card = rl.Rectangle(body.x + left_w + gap, body.y, right_w, top_h)
    rl.draw_rectangle_rounded(right_card, 0.08, 10, BG_CARD)
    rl.draw_rectangle_rounded_lines_ex(right_card, 0.08, 10, max(1.0, 1.5 * s), BORDER_SUBTLE)

    gui_label(
      rl.Rectangle(right_card.x + 28.0 * s, right_card.y + 22.0 * s, 600.0 * s, 46.0 * s),
      "급감속 이벤트 (<= -2.8 m/s^2)",
      int(34 * s),
      TEXT_PRIMARY,
      FontWeight.BOLD,
      TextAlignment.LEFT,
    )
    tot_pill = rl.Rectangle(right_card.x + right_card.width - 160.0 * s, right_card.y + 20.0 * s, 132.0 * s, 48.0 * s)
    rl.draw_rectangle_rounded(tot_pill, 0.3, 8, BG_CARD_ALT)
    gui_label(tot_pill, f"총 {total_cnt}회", int(28 * s), COLOR_STEER if total_cnt > 0 else COLOR_AUTO, FontWeight.BOLD, TextAlignment.CENTER)

    if not events:
      gui_label(
        rl.Rectangle(right_card.x + 28.0 * s, right_card.y + 170.0 * s, right_card.width - 56.0 * s, 64.0 * s),
        "감지된 급감속 이벤트가 없습니다.",
        int(34 * s),
        TEXT_MUTED,
        FontWeight.MEDIUM,
        TextAlignment.CENTER,
      )
    else:
      row_y0 = right_card.y + 86.0 * s
      row_h = 94.0 * s
      row_gap = 12.0 * s
      for idx, ev in enumerate(events[:3]):
        row = rl.Rectangle(right_card.x + 24.0 * s, row_y0 + idx * (row_h + row_gap), right_card.width - 48.0 * s, row_h)
        rl.draw_rectangle_rounded(row, 0.14, 8, BG_RAIL)
        src_label = str(ev.get("source_label") or ("수동" if ev.get("source") == "manual" else "자동주행"))
        t_str = str(ev.get("time_str") or "")
        v_start = int(ev.get("start_speed_kph") or 0)
        v_end = int(ev.get("end_speed_kph") or 0)
        decel = float(ev.get("peak_decel") or 0.0)

        gui_label(
          rl.Rectangle(row.x + 22.0 * s, row.y, row.width - 252.0 * s, row.height),
          f"{t_str}  [{src_label}]  {v_start} -> {v_end} km/h",
          int(32 * s),
          TEXT_PRIMARY,
          FontWeight.SEMI_BOLD,
          TextAlignment.LEFT,
        )
        gui_label(
          rl.Rectangle(row.x + row.width - 230.0 * s, row.y, 208.0 * s, row.height),
          f"{decel:.2f} m/s^2",
          int(36 * s),
          COLOR_DANGER if decel <= -2.8 else COLOR_STEER,
          FontWeight.BOLD,
          TextAlignment.RIGHT,
        )

    # Bottom card: Local File Path + History Browser Button
    bot_y = body.y + top_h + gap
    bot_h = body.height - top_h - gap
    bot_card = rl.Rectangle(body.x, bot_y, body.width, bot_h)
    rl.draw_rectangle_rounded(bot_card, 0.08, 10, BG_CARD)
    rl.draw_rectangle_rounded_lines_ex(bot_card, 0.08, 10, max(1.0, 1.5 * s), BORDER_SUBTLE)

    file_path = str(self._report.get("file_path") or "/data/drive_reports/...")
    btn_w = 248.0 * s
    info_w = bot_card.width - btn_w - 76.0 * s
    gui_label(
      rl.Rectangle(bot_card.x + 28.0 * s, bot_card.y + 34.0 * s, info_w, 46.0 * s),
      f"저장 위치: {file_path}",
      int(28 * s),
      TEXT_MUTED,
      FontWeight.MEDIUM,
      TextAlignment.LEFT,
    )
    gui_label(
      rl.Rectangle(bot_card.x + 28.0 * s, bot_card.y + 100.0 * s, info_w, 52.0 * s),
      "주행 리포트가 단말에 저장되었습니다 (저장 목록에서 과거 기록 조회 가능)",
      int(30 * s),
      COLOR_AUTO,
      FontWeight.BOLD,
      TextAlignment.LEFT,
    )
    self._btn_page3_history = rl.Rectangle(
      bot_card.x + bot_card.width - btn_w - 28.0 * s,
      bot_card.y + (bot_card.height - 96.0 * s) * 0.5,
      btn_w,
      96.0 * s,
    )
    rl.draw_rectangle_rounded(self._btn_page3_history, 0.2, 8, BTN_PRIMARY)
    gui_label(
      self._btn_page3_history,
      "저장 목록 >",
      int(30 * s),
      rl.WHITE,
      FontWeight.BOLD,
      TextAlignment.CENTER,
    )

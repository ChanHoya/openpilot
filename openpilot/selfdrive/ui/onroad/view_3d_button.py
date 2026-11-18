import time
import pyray as rl
from openpilot.common.params import Params
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget


class View3dButton(Widget):
  def __init__(self, width: int = 150, height: int = 68):
    super().__init__()
    self._params = Params()
    self._rect = rl.Rectangle(0, 0, width, height)
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi_bold = gui_app.font(FontWeight.SEMI_BOLD)
    self._last_read_t = 0.0
    v3d = self._try_get_bool("View3dMode")
    fsd = self._try_get_bool("FsdViewMode")
    self._view_3d_mode = bool(v3d or fsd)
    self._write_param(self._view_3d_mode)

  def _try_get_bool(self, key: str) -> bool | None:
    try:
      return bool(self._params.get_bool(key))
    except Exception:
      return None

  def _read_param(self) -> bool:
    v3d = self._try_get_bool("View3dMode")
    if v3d is not None:
      return v3d
    return bool(self._try_get_bool("FsdViewMode"))

  def _write_param(self, val: bool) -> None:
    for key in ("FsdViewMode", "View3dMode"):
      try:
        self._params.put_bool(key, bool(val))
      except Exception:
        pass

  def set_rect(self, rect: rl.Rectangle) -> None:
    super().set_rect(rect)
    self._rect = rl.Rectangle(rect.x, rect.y, rect.width, rect.height)

  @property
  def view_3d_mode(self) -> bool:
    return self._view_3d_mode

  @property
  def fsd_mode(self) -> bool:
    return self._view_3d_mode

  def _handle_mouse_release(self, _):
    super()._handle_mouse_release(_)
    self._view_3d_mode = not self._view_3d_mode
    self._last_read_t = time.monotonic()
    self._write_param(self._view_3d_mode)

  def _render(self, rect: rl.Rectangle) -> None:
    now = time.monotonic()
    if now - self._last_read_t > 1.0:
      self._last_read_t = now
      self._view_3d_mode = self._read_param()

    draw_rect = rect if rect is not None else self._rect
    is_compact = draw_rect.width < 120

    if self._view_3d_mode:
      bg_color = rl.Color(10, 48, 80, 240) if not self.is_pressed else rl.Color(18, 70, 115, 250)
      border_color = rl.Color(0, 255, 180, 255)
      main_text = "AR VIEW" if not is_compact else "AR"
      sub_text = "3D ON" if not is_compact else "3D"
      main_color = rl.Color(0, 255, 180, 255)
      sub_color = rl.Color(0, 220, 255, 240)
    else:
      bg_color = rl.Color(18, 24, 34, 230) if not self.is_pressed else rl.Color(34, 44, 58, 245)
      border_color = rl.Color(0, 210, 255, 230)
      main_text = "3D VIEW" if not is_compact else "3D"
      sub_text = "AR ON" if not is_compact else "AR"
      main_color = rl.Color(0, 230, 255, 255)
      sub_color = rl.Color(200, 215, 235, 220)

    # Rounded pill box
    rl.draw_rectangle_rounded(draw_rect, 0.32, 8, bg_color)
    rl.draw_rectangle_rounded_lines_ex(draw_rect, 0.32, 8, 2.2, border_color)

    if is_compact:
      font_size = 17
      size = measure_text_cached(self._font_bold, main_text, font_size)
      tx = draw_rect.x + (draw_rect.width - size.x) / 2
      ty = draw_rect.y + 4
      rl.draw_text_ex(self._font_bold, main_text, rl.Vector2(tx, ty), font_size, 0, main_color)

      sub_font_size = 12
      sub_size = measure_text_cached(self._font_semi_bold, sub_text, sub_font_size)
      stx = draw_rect.x + (draw_rect.width - sub_size.x) / 2
      sty = draw_rect.y + draw_rect.height - sub_size.y - 4
      rl.draw_text_ex(self._font_semi_bold, sub_text, rl.Vector2(stx, sty), sub_font_size, 0, sub_color)
    else:
      font_size = 28
      size = measure_text_cached(self._font_bold, main_text, font_size)
      tx = draw_rect.x + (draw_rect.width - size.x) / 2
      ty = draw_rect.y + 12
      rl.draw_text_ex(self._font_bold, main_text, rl.Vector2(tx, ty), font_size, 0, main_color)

      sub_font_size = 20
      sub_size = measure_text_cached(self._font_semi_bold, sub_text, sub_font_size)
      stx = draw_rect.x + (draw_rect.width - sub_size.x) / 2
      sty = draw_rect.y + draw_rect.height - sub_size.y - 12
      rl.draw_text_ex(self._font_semi_bold, sub_text, rl.Vector2(stx, sty), sub_font_size, 0, sub_color)


# Backward compatibility alias
FsdButton = View3dButton


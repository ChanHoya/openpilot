import os
import subprocess
import time
import pyray as rl
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.confirm_dialog import ConfirmDialog
from openpilot.system.ui.widgets.option_dialog import MultiOptionDialog

SCC_MODE_NAMES = {
  0: "순정(0)",
  1: "오파(1)",
  2: "동기(2)",
  3: "연결(3)",
}

SCC_MODE_COLORS = {
  0: rl.Color(160, 160, 160, 240),   # Grey (Stock)
  1: rl.Color(0, 215, 80, 255),      # Green (Openpilot Long)
  2: rl.Color(255, 175, 20, 240),    # Orange (Sync)
  3: rl.Color(199, 125, 255, 240),   # Purple/Lavender (Link)
}

OPTIONS_LIST = [
  "0: 순정 크루즈 (Stock SCC)",
  "1: 오파 롱컨 (Openpilot Long)",
  "2: 크루즈 상태 동기화 (Cruise Sync)",
  "3: 연결 크루즈 (CAN-FD 롱컨 배선)",
]


class CameraSCCButton(Widget):
  def __init__(self, width: int = 210, height: int = 114):
    super().__init__()
    self._params = Params()
    self._rect = rl.Rectangle(0, 0, width, height)
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._font_semi_bold = gui_app.font(FontWeight.SEMI_BOLD)
    self._font_medium = gui_app.font(FontWeight.MEDIUM)

    self._cached_mode: int = 0
    self._last_mode_read_time: float = 0.0
    self._pending_scc_mode: int | None = None
    self._is_restarting: bool = False

  def set_rect(self, rect: rl.Rectangle) -> None:
    self._rect = rl.Rectangle(rect.x, rect.y, rect.width, rect.height)

  def _get_current_scc_val(self) -> int:
    now = time.monotonic()
    if now - self._last_mode_read_time > 0.5:
      self._last_mode_read_time = now
      try:
        self._cached_mode = self._params.get_int("HyundaiCameraSCC")
      except Exception:
        self._cached_mode = 0
    return self._cached_mode

  def _is_standstill(self) -> bool:
    """Check if the vehicle is currently stopped or parked."""
    try:
      sm = ui_state.sm
      if sm is not None and "carState" in sm.recv_frame and sm.recv_frame["carState"] > 0:
        cs = sm["carState"]
        # vEgo < 0.1 m/s (0.36 km/h) or standstill flag or GearShifter.park (1)
        if cs.standstill or cs.vEgo < 0.1 or getattr(cs, "gearShifter", 0) == 1:
          return True
        return False
    except Exception:
      pass
    return True

  def update_state(self) -> None:
    """Check for standstill condition when a mode change is pending."""
    if self._pending_scc_mode is not None and not self._is_restarting:
      if self._is_standstill():
        target_mode = self._pending_scc_mode
        self._pending_scc_mode = None
        self._is_restarting = True
        self._show_standstill_restart_popup_and_restart(target_mode)

  def _show_standstill_restart_popup_and_restart(self, target_mode: int) -> None:
    opt_name = OPTIONS_LIST[target_mode] if 0 <= target_mode < len(OPTIONS_LIST) else f"모드 {target_mode}"
    confirm_msg = (
      "🛑 정지 상태가 감지되었습니다.\n\n"
      f"▶ 적용 모드: {opt_name}\n\n"
      "HyundaiCameraSCC 설정을 적용하기 위해\n"
      "오픈파일럿을 소프트 리셋(재부팅)합니다.\n"
      "(약 3~5초 소요)"
    )

    confirm_dialog = ConfirmDialog(confirm_msg, "재부팅 진행", None)
    gui_app.push_widget(confirm_dialog)
    self._apply_mode_and_restart(target_mode)

  def _handle_mouse_release(self, _):
    super()._handle_mouse_release(_)
    self._show_scc_selection_dialog()

  def _show_scc_selection_dialog(self) -> None:
    current_val = self._get_current_scc_val()
    current_str = OPTIONS_LIST[current_val] if 0 <= current_val < len(OPTIONS_LIST) else OPTIONS_LIST[0]

    def on_option_selected(result: DialogResult):
      if result == DialogResult.CONFIRM:
        selected_opt = dialog.selection
        try:
          new_val = int(selected_opt.split(":")[0])
        except Exception:
          return

        if new_val == current_val:
          if self._pending_scc_mode is not None:
            self._pending_scc_mode = None
          return

        is_stopped = self._is_standstill()
        if is_stopped:
          # Already in standstill: direct soft restart confirmation
          confirm_msg = (
            f"HyundaiCameraSCC 설정을 변경합니다.\n\n"
            f"▶ 변경 모드: {selected_opt}\n\n"
            "🛑 정지 상태가 확인되었습니다.\n"
            "오픈파일럿을 즉시 소프트 리셋(재부팅)합니다.\n"
            "(약 3~5초 소요)\n\n"
            "소프트 리셋을 진행하시겠습니까?"
          )

          def on_confirm(confirm_result: DialogResult):
            if confirm_result == DialogResult.CONFIRM:
              self._apply_mode_and_restart(new_val)

          confirm_dialog = ConfirmDialog(confirm_msg, "확인 (소프트 리셋)", "취소", callback=on_confirm)
          gui_app.push_widget(confirm_dialog)
        else:
          # Vehicle is moving: schedule pending mode change on standstill
          self._pending_scc_mode = new_val
          notice_msg = (
            f"HyundaiCameraSCC 변경 모드가 예약되었습니다.\n\n"
            f"▶ 예약 모드: {selected_opt}\n\n"
            "⚠️ 안전을 위해 주행 중에는 재부팅되지 않습니다.\n"
            "차량이 신호 대기 등으로 완전히 정지(0 km/h)하면\n"
            "안내 팝업과 함께 자동으로 소프트 리셋(재부팅)됩니다."
          )

          def on_notice_confirm(confirm_result: DialogResult):
            if confirm_result == DialogResult.CANCEL:
              self._pending_scc_mode = None

          notice_dialog = ConfirmDialog(notice_msg, "확인 (정지 시 자동 재부팅)", "예약 취소", callback=on_notice_confirm)
          gui_app.push_widget(notice_dialog)

    dialog = MultiOptionDialog(
      "HYUNDAI CAMERA SCC 모드 선택",
      OPTIONS_LIST,
      current=current_str,
      callback=on_option_selected,
    )
    gui_app.push_widget(dialog)

  def _apply_mode_and_restart(self, new_val: int) -> None:
    try:
      self._params.put_int("HyundaiCameraSCC", new_val)
      self._params.put_nonblocking("HyundaiCameraSCC", str(new_val))
    except Exception as e:
      print(f"Failed to write HyundaiCameraSCC param: {e}")

    # Trigger soft restart
    restart_script = os.path.join(BASEDIR, "restart.sh")
    if os.path.exists(restart_script):
      try:
        subprocess.Popen(["bash", restart_script])
      except Exception as e:
        print(f"Failed to execute restart.sh: {e}")
        os.system(f"bash {restart_script} &")
    else:
      os.system("pkill -2 -f openpilot.system.manager || pkill -f manager.py")

  def _render(self, rect: rl.Rectangle) -> None:
    mode = self._get_current_scc_val()
    mode_name = SCC_MODE_NAMES.get(mode, f"SCC({mode})")

    if self._pending_scc_mode is not None:
      pending_name = SCC_MODE_NAMES.get(self._pending_scc_mode, f"{self._pending_scc_mode}")
      accent_color = rl.Color(255, 190, 20, 255)  # Amber warning color
      top_text = "정지시 재부팅"
      main_text = f"대기:{pending_name}"
    else:
      accent_color = SCC_MODE_COLORS.get(mode, rl.Color(160, 160, 160, 240))
      top_text = "CAMERA SCC"
      main_text = mode_name

    # Background box
    bg_color = rl.Color(45, 45, 45, 220) if self.is_pressed else rl.Color(0, 0, 0, 170)
    rl.draw_rectangle_rounded(self._rect, 0.28, 8, bg_color)
    rl.draw_rectangle_rounded_lines_ex(self._rect, 0.28, 8, 2.5, accent_color)

    # Top Header Label
    top_font_size = 26 if len(top_text) > 10 else 28
    top_size = measure_text_cached(self._font_semi_bold, top_text, top_font_size)
    top_x = self._rect.x + (self._rect.width - top_size.x) / 2
    top_y = self._rect.y + 14
    rl.draw_text_ex(
      self._font_semi_bold,
      top_text,
      rl.Vector2(top_x, top_y),
      top_font_size,
      0,
      rl.Color(220, 220, 220, 230) if self._pending_scc_mode is None else accent_color,
    )

    # Main Mode Label: e.g. "오파(1)", "순정(0)", "대기:오파(1)"
    main_font_size = 38 if len(main_text) > 6 else 44
    main_size = measure_text_cached(self._font_bold, main_text, main_font_size)
    main_x = self._rect.x + (self._rect.width - main_size.x) / 2
    main_y = self._rect.y + self._rect.height - main_size.y - 14
    rl.draw_text_ex(
      self._font_bold,
      main_text,
      rl.Vector2(main_x, main_y),
      main_font_size,
      0,
      accent_color,
    )


"""
system_actions.py — Windows system control actions
====================================================
Maps gesture names to OS-level actions: mouse, keyboard,
volume, brightness, media keys, window management, etc.
All actions use pyautogui + ctypes for native Windows control.
"""

import ctypes
import ctypes.wintypes
import time
import math
import numpy as np
import pyautogui

# Disable pyautogui's safety pause and failsafe
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


# ---------------------------------------------------------------------------
# Windows Virtual-Key constants
# ---------------------------------------------------------------------------
VK_VOLUME_UP        = 0xAF
VK_VOLUME_DOWN      = 0xAE
VK_VOLUME_MUTE      = 0xAD
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_MEDIA_NEXT       = 0xB0
VK_MEDIA_PREV       = 0xB1
KEYEVENTF_KEYUP     = 0x0002


def _press_vk(vk_code: int):
    """Send a virtual-key press and release."""
    ctypes.windll.user32.keybd_event(vk_code, 0, 0, 0)
    ctypes.windll.user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)


# ---------------------------------------------------------------------------
# Smoother for cursor movement
# ---------------------------------------------------------------------------

class CursorSmoother:
    """Exponential moving-average smoother for jitter-free cursor control."""

    def __init__(self, factor: int = 5):
        self.factor = factor
        self._px: float = None
        self._py: float = None

    def smooth(self, x: float, y: float) -> tuple[int, int]:
        if self._px is None:
            self._px, self._py = x, y
        else:
            self._px += (x - self._px) / self.factor
            self._py += (y - self._py) / self.factor
        return int(self._px), int(self._py)

    def reset(self):
        self._px = self._py = None


# ---------------------------------------------------------------------------
# Coordinate mapping
# ---------------------------------------------------------------------------

SCREEN_W, SCREEN_H = pyautogui.size()

# Active region of the camera frame that maps to the full screen
# (margins stop the user from having to reach the extreme edges)
MARGIN_X = 0.12
MARGIN_Y = 0.12


def hand_to_screen(nx: float, ny: float) -> tuple[int, int]:
    """
    Convert normalized hand coordinates (0–1) to screen pixels.
    The active zone is the camera frame minus the margins.
    """
    sx = np.interp(nx, [MARGIN_X, 1.0 - MARGIN_X], [0, SCREEN_W])
    sy = np.interp(ny, [MARGIN_Y, 1.0 - MARGIN_Y], [0, SCREEN_H])
    return (
        int(np.clip(sx, 0, SCREEN_W - 1)),
        int(np.clip(sy, 0, SCREEN_H - 1)),
    )


# ---------------------------------------------------------------------------
# Action Controller
# ---------------------------------------------------------------------------

class ActionController:
    """
    Receives a gesture name + hand data each frame and executes
    the corresponding system action (move cursor, click, scroll, etc.).

    Manages cooldowns, drag state, and scroll tracking internally.
    """

    def __init__(self, scroll_speed: int = 18, smoothing: int = 5):
        self.smoother = CursorSmoother(factor=smoothing)
        self.scroll_speed = scroll_speed

        # Cooldowns (seconds)
        self.click_cd      = 0.45
        self.dbl_click_cd  = 0.70
        self.rclick_cd     = 0.50
        self.volume_cd     = 0.25
        self.media_cd      = 0.80
        self.swipe_cd      = 0.80
        self.screenshot_cd = 2.0

        # Timestamps of last action
        self._ts: dict[str, float] = {}

        # Drag state
        self._dragging = False

        # Scroll reference
        self._scroll_ref_y: float = None

        # Action log for HUD
        self.last_action: str = ""

    # ----- public entry point ----------------------------------------- #

    def execute(self, gesture: str, result) -> str:
        """
        Execute the action for `gesture`. `result` is a GestureResult.
        Returns the action description string for the HUD.
        """
        now = time.time()

        # --- Pointer: move cursor ---
        if gesture == "pointer":
            self._end_drag()
            sx, sy = hand_to_screen(*result.index_tip)
            mx, my = self.smoother.smooth(sx, sy)
            pyautogui.moveTo(mx, my, _pause=False)
            self._scroll_ref_y = None
            self.last_action = f"MOVE  ({mx}, {my})"
            return self.last_action

        # --- Left click ---
        if gesture == "click":
            self._end_drag()
            sx, sy = hand_to_screen(*result.index_tip)
            mx, my = self.smoother.smooth(sx, sy)
            pyautogui.moveTo(mx, my, _pause=False)
            if self._cooldown_ok("click", self.click_cd, now):
                pyautogui.click()
                self.last_action = "LEFT CLICK"
            self._scroll_ref_y = None
            return self.last_action

        # --- Right click ---
        if gesture == "right_click":
            self._end_drag()
            sx, sy = hand_to_screen(*result.index_tip)
            mx, my = self.smoother.smooth(sx, sy)
            pyautogui.moveTo(mx, my, _pause=False)
            if self._cooldown_ok("rclick", self.rclick_cd, now):
                pyautogui.rightClick()
                self.last_action = "RIGHT CLICK"
            self._scroll_ref_y = None
            return self.last_action

        # --- Double click (pinch) ---
        if gesture == "pinch":
            self._end_drag()
            sx, sy = hand_to_screen(*result.index_tip)
            mx, my = self.smoother.smooth(sx, sy)
            pyautogui.moveTo(mx, my, _pause=False)
            if self._cooldown_ok("dblclick", self.dbl_click_cd, now):
                pyautogui.doubleClick()
                self.last_action = "DOUBLE CLICK"
            self._scroll_ref_y = None
            return self.last_action

        # --- Drag (fist) ---
        if gesture == "drag":
            sx, sy = hand_to_screen(*result.wrist)
            mx, my = self.smoother.smooth(sx, sy)
            if not self._dragging:
                pyautogui.moveTo(mx, my, _pause=False)
                pyautogui.mouseDown()
                self._dragging = True
                self.last_action = "DRAG START"
            else:
                pyautogui.moveTo(mx, my, _pause=False)
                self.last_action = f"DRAGGING  ({mx}, {my})"
            self._scroll_ref_y = None
            return self.last_action

        # --- Scroll (4 fingers) ---
        if gesture == "scroll":
            self._end_drag()
            wrist_y = result.wrist[1]
            if self._scroll_ref_y is not None:
                delta = self._scroll_ref_y - wrist_y
                amount = int(delta * self.scroll_speed * 80)
                if abs(amount) > 1:
                    pyautogui.scroll(amount)
                    direction = "UP" if amount > 0 else "DOWN"
                    self.last_action = f"SCROLL {direction} ({abs(amount)})"
            self._scroll_ref_y = wrist_y
            return self.last_action

        # --- Thumbs up → volume up ---
        if gesture == "thumbs_up":
            self._end_drag()
            if self._cooldown_ok("volume", self.volume_cd, now):
                _press_vk(VK_VOLUME_UP)
                self.last_action = "VOLUME UP"
            self._scroll_ref_y = None
            return self.last_action

        # --- Thumbs down → volume down ---
        if gesture == "thumbs_down":
            self._end_drag()
            if self._cooldown_ok("volume", self.volume_cd, now):
                _press_vk(VK_VOLUME_DOWN)
                self.last_action = "VOLUME DOWN"
            self._scroll_ref_y = None
            return self.last_action

        # --- Peace sign → media play/pause ---
        if gesture == "peace":
            self._end_drag()
            if self._cooldown_ok("media", self.media_cd, now):
                _press_vk(VK_MEDIA_PLAY_PAUSE)
                self.last_action = "PLAY / PAUSE"
            self._scroll_ref_y = None
            return self.last_action

        # --- Rock sign → mute/unmute ---
        if gesture == "rock":
            self._end_drag()
            if self._cooldown_ok("mute", self.media_cd, now):
                _press_vk(VK_VOLUME_MUTE)
                self.last_action = "MUTE TOGGLE"
            self._scroll_ref_y = None
            return self.last_action

        # --- Idle (open palm) ---
        if gesture == "idle":
            self._end_drag()
            self.smoother.reset()
            self._scroll_ref_y = None
            self.last_action = "IDLE"
            return self.last_action

        # --- No hand ---
        if gesture == "none":
            self._end_drag()
            self.smoother.reset()
            self._scroll_ref_y = None
            self.last_action = ""
            return self.last_action

        return ""

    def execute_swipe(self, direction: str):
        """Execute a swipe action (Alt+Tab or media skip)."""
        now = time.time()
        if not self._cooldown_ok("swipe", self.swipe_cd, now):
            return
        if direction == "left":
            # Alt+Tab (switch window)
            pyautogui.hotkey("alt", "tab")
            self.last_action = "ALT+TAB (Swipe Left)"
        elif direction == "right":
            # Next media track
            _press_vk(VK_MEDIA_NEXT)
            self.last_action = "NEXT TRACK (Swipe Right)"

    def take_screenshot(self):
        """Win+Shift+S screenshot (Windows snipping tool)."""
        now = time.time()
        if self._cooldown_ok("screenshot", self.screenshot_cd, now):
            pyautogui.hotkey("win", "shift", "s")
            self.last_action = "SCREENSHOT"

    def cleanup(self):
        """Release any held mouse buttons."""
        self._end_drag()

    # ----- internal --------------------------------------------------- #

    def _end_drag(self):
        if self._dragging:
            pyautogui.mouseUp()
            self._dragging = False

    def _cooldown_ok(self, key: str, cd: float, now: float) -> bool:
        last = self._ts.get(key, 0)
        if now - last >= cd:
            self._ts[key] = now
            return True
        return False

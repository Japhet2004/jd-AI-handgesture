"""
ui_overlay.py — Camera window rendering and floating skeleton overlay
======================================================================
Provides the HUD (heads-up display) drawn on the camera frame and the
small always-on-top transparent Tkinter overlay that mirrors the hand.
"""

import cv2
import numpy as np
import tkinter as tk
import mediapipe as mp

HAND_CONNECTIONS = list(mp.solutions.hands.HAND_CONNECTIONS)


# ---------------------------------------------------------------------------
# Color palette  (BGR for OpenCV, hex for Tkinter)
# ---------------------------------------------------------------------------

GESTURE_COLORS_BGR = {
    "pointer":     (0, 230, 0),        # green
    "click":       (255, 230, 0),      # cyan-ish
    "right_click": (0, 150, 255),      # orange
    "scroll":      (255, 0, 220),      # magenta
    "idle":        (160, 160, 160),    # gray
    "drag":        (0, 0, 255),        # red
    "pinch":       (255, 60, 0),       # blue
    "thumbs_up":   (0, 255, 255),      # yellow
    "thumbs_down": (40, 180, 255),     # gold
    "peace":       (255, 180, 0),      # light blue
    "rock":        (80, 0, 200),       # dark red
    "none":        (80, 80, 80),       # dark gray
}

GESTURE_COLORS_HEX = {
    "pointer":     "#00E600",
    "click":       "#00E6FF",
    "right_click": "#FF9600",
    "scroll":      "#DC00FF",
    "idle":        "#A0A0A0",
    "drag":        "#FF0000",
    "pinch":       "#003CFF",
    "thumbs_up":   "#FFFF00",
    "thumbs_down": "#FFB428",
    "peace":       "#00B4FF",
    "rock":        "#C80050",
    "none":        "#505050",
}

GESTURE_ICONS = {
    "pointer":     "☝",
    "click":       "✌ CLICK",
    "right_click": "🤟 R-CLICK",
    "scroll":      "🖐 SCROLL",
    "idle":        "✋ IDLE",
    "drag":        "✊ DRAG",
    "pinch":       "🤏 DBL-CLICK",
    "thumbs_up":   "👍 VOL+",
    "thumbs_down": "👎 VOL-",
    "peace":       "✌ PLAY/PAUSE",
    "rock":        "🤘 MUTE",
    "none":        "—",
}

# Active zone margins (must match system_actions.py)
MARGIN_X = 0.12
MARGIN_Y = 0.12


# ---------------------------------------------------------------------------
# HUD drawing (on the OpenCV camera frame)
# ---------------------------------------------------------------------------

class HUD:
    """Draws mode indicator, finger count, action log, and guides onto the frame."""

    def __init__(self):
        self._action_log: list[str] = []

    def draw(self, frame: np.ndarray, gesture: str, finger_count: int,
             confidence: float, action: str, fps: float):
        """Draw the full HUD onto `frame`."""
        h, w = frame.shape[:2]
        color = GESTURE_COLORS_BGR.get(gesture, (128, 128, 128))

        # --- Active zone rectangle ---
        ax1 = int(MARGIN_X * w)
        ay1 = int(MARGIN_Y * h)
        ax2 = int((1 - MARGIN_X) * w)
        ay2 = int((1 - MARGIN_Y) * h)
        cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), (60, 60, 60), 1)

        # --- Mode badge (top-left) ---
        icon = GESTURE_ICONS.get(gesture, gesture)
        label = f" {icon} "
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, 0.65, 2)
        pad = 8
        cv2.rectangle(frame, (8, 6), (8 + tw + pad * 2, 6 + th + pad * 2), (20, 20, 20), -1)
        cv2.rectangle(frame, (8, 6), (8 + tw + pad * 2, 6 + th + pad * 2), color, 2)
        cv2.putText(frame, label, (8 + pad, 6 + th + pad),
                    font, 0.65, color, 2, cv2.LINE_AA)

        # --- Finger count + confidence bar ---
        y_off = 6 + th + pad * 2 + 28
        fc_text = f"Fingers: {finger_count}   Conf: {confidence:.0%}"
        cv2.putText(frame, fc_text, (14, y_off),
                    font, 0.42, (190, 190, 190), 1, cv2.LINE_AA)

        # Confidence bar
        bar_x = 14
        bar_y = y_off + 8
        bar_w = 160
        bar_h = 6
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (60, 60, 60), -1)
        fill_w = int(bar_w * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                      color, -1)

        # --- Last action (top-right) ---
        if action:
            (aw, ah), _ = cv2.getTextSize(action, font, 0.45, 1)
            ax = w - aw - 16
            cv2.putText(frame, action, (ax, 28),
                        font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # --- FPS (bottom-right) ---
        fps_text = f"FPS: {fps:.0f}"
        cv2.putText(frame, fps_text, (w - 90, h - 14),
                    font, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

        # --- Help bar (bottom) ---
        hints = "Q=Quit  H=Hide  O=Overlay  F=Fullscreen  S=Screenshot"
        cv2.putText(frame, hints, (10, h - 14),
                    font, 0.35, (80, 80, 80), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Glowing skeleton drawing on camera frame
# ---------------------------------------------------------------------------

def draw_skeleton(frame: np.ndarray, landmarks, color_bgr: tuple):
    """Draw hand skeleton with glow effect on the camera frame."""
    h, w = frame.shape[:2]

    # Glow pass
    glow = frame.copy()
    for a, b in HAND_CONNECTIONS:
        la, lb = landmarks.landmark[a], landmarks.landmark[b]
        x1, y1 = int(la.x * w), int(la.y * h)
        x2, y2 = int(lb.x * w), int(lb.y * h)
        cv2.line(glow, (x1, y1), (x2, y2), color_bgr, 10)
    cv2.addWeighted(glow, 0.3, frame, 0.7, 0, frame)

    # Thin lines
    for a, b in HAND_CONNECTIONS:
        la, lb = landmarks.landmark[a], landmarks.landmark[b]
        x1, y1 = int(la.x * w), int(la.y * h)
        x2, y2 = int(lb.x * w), int(lb.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), color_bgr, 2, cv2.LINE_AA)

    # Joints
    for lm in landmarks.landmark:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 7, color_bgr, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Floating Skeleton Overlay (Tkinter)
# ---------------------------------------------------------------------------

OVERLAY_W = 220
OVERLAY_H = 180
_KEY_COLOR = "black"


class SkeletonOverlay:
    """Small always-on-top floating window that mirrors the hand skeleton."""

    def __init__(self):
        self._root = tk.Tk()
        self._root.withdraw()

        self._win = tk.Toplevel(self._root)
        self._win.title("Gesture Overlay")
        self._win.geometry(f"{OVERLAY_W}x{OVERLAY_H}+40+40")
        self._win.resizable(False, False)
        self._win.overrideredirect(True)
        self._win.wm_attributes("-topmost", True)

        try:
            self._win.wm_attributes("-transparentcolor", _KEY_COLOR)
        except tk.TclError:
            pass

        self._canvas = tk.Canvas(
            self._win, width=OVERLAY_W, height=OVERLAY_H,
            bg=_KEY_COLOR, highlightthickness=0,
        )
        self._canvas.pack(fill="both", expand=True)

        # Drag support
        self._dx = self._dy = 0
        self._canvas.bind("<ButtonPress-1>", self._drag_start)
        self._canvas.bind("<B1-Motion>", self._drag_move)

        # Close button
        close_btn = tk.Label(
            self._win, text="×", fg="#FF4444", bg=_KEY_COLOR,
            font=("Segoe UI", 13, "bold"), cursor="hand2",
        )
        close_btn.place(relx=1.0, rely=0.0, x=-4, y=2, anchor="ne")
        close_btn.bind("<ButtonPress-1>", lambda _: self.hide())

        self._hands = []
        self._gesture = "none"
        self._visible = True
        self._quit = False

        self._win.bind("<KeyPress-q>", lambda _: self._request_quit())
        self._win.bind("<Escape>", lambda _: self._request_quit())

    # --- public ----------------------------------------------------- #

    def set_data(self, hands_colors: list, gesture: str = "none"):
        """hands_colors: list of (landmarks, hex_color) tuples."""
        self._hands = hands_colors or []
        self._gesture = gesture

    def refresh(self):
        self._canvas.delete("all")
        for lm, color in self._hands:
            self._draw_skeleton(lm, color)
        # Mode label
        mode_color = GESTURE_COLORS_HEX.get(self._gesture, "#808080")
        self._canvas.create_text(
            OVERLAY_W // 2, OVERLAY_H - 10,
            text=GESTURE_ICONS.get(self._gesture, self._gesture),
            fill=mode_color, font=("Segoe UI", 8, "bold"),
        )
        self._root.update_idletasks()

    def show(self):
        if not self._visible:
            self._win.deiconify()
            self._visible = True

    def hide(self):
        if self._visible:
            self._win.withdraw()
            self._visible = False

    def toggle(self):
        self.hide() if self._visible else self.show()

    def quit_requested(self) -> bool:
        return self._quit

    def destroy(self):
        try:
            self._root.destroy()
        except Exception:
            pass

    # --- internal --------------------------------------------------- #

    def _request_quit(self):
        self._quit = True

    def _drag_start(self, e):
        self._dx = e.x_root - self._win.winfo_x()
        self._dy = e.y_root - self._win.winfo_y()

    def _drag_move(self, e):
        self._win.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")

    def _draw_skeleton(self, landmarks, color_hex):
        w, h = OVERLAY_W, OVERLAY_H
        # Glow
        for a, b in HAND_CONNECTIONS:
            la, lb = landmarks.landmark[a], landmarks.landmark[b]
            x1, y1 = int(la.x * w), int(la.y * h)
            x2, y2 = int(lb.x * w), int(lb.y * h)
            self._canvas.create_line(x1, y1, x2, y2, fill="#003060",
                                     width=6, capstyle=tk.ROUND)
        # Lines
        for a, b in HAND_CONNECTIONS:
            la, lb = landmarks.landmark[a], landmarks.landmark[b]
            x1, y1 = int(la.x * w), int(la.y * h)
            x2, y2 = int(lb.x * w), int(lb.y * h)
            self._canvas.create_line(x1, y1, x2, y2, fill=color_hex,
                                     width=2, capstyle=tk.ROUND)
        # Joints
        for lm in landmarks.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            r = 3
            self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                     fill="white", outline=color_hex, width=1)

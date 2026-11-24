"""
Hand Gesture Control for Windows — Main Application
=====================================================
Full-featured webcam-based hand gesture controller.

Gestures:
  ☝  1 finger (index)        →  Move cursor
  ✌  2 fingers (index+mid)   →  Left click
  🤟 3 fingers               →  Right click
  🖐 4 fingers               →  Scroll (move hand up/down)
  ✋ 5 fingers (open palm)   →  Idle / pause
  ✊ Fist (0 fingers)        →  Click-and-drag
  🤏 Pinch (thumb+index)     →  Double click
  👍 Thumbs up               →  Volume up
  👎 Thumbs down             →  Volume down
  ✌  Peace / V-sign          →  Play / Pause media
  🤘 Rock sign               →  Mute toggle
  ← Swipe left               →  Alt+Tab
  → Swipe right              →  Next track

Keyboard:
  Q / Esc  →  Quit
  H        →  Hide / show camera window
  O        →  Toggle skeleton overlay
  F        →  Fullscreen camera
  S        →  Screenshot (Win+Shift+S)
"""

import cv2
import numpy as np
import time
import json
import os
import sys

from gesture_detector import GestureDetector
from system_actions import ActionController
from ui_overlay import (
    HUD, SkeletonOverlay, draw_skeleton,
    GESTURE_COLORS_BGR, GESTURE_COLORS_HEX,
)


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = r"""
╔═══════════════════════════════════════════════════╗
║       ✋  Hand Gesture Control for Windows  ✋    ║
╠═══════════════════════════════════════════════════╣
║                                                   ║
║  GESTURES                                         ║
║  ───────                                          ║
║  ☝  Index finger only     →  Move cursor          ║
║  ✌  Index + Middle        →  Left click           ║
║  🤟 Three fingers          →  Right click          ║
║  🖐 Four fingers           →  Scroll up / down     ║
║  ✋ Open palm (5)          →  Idle / pause         ║
║  ✊ Fist (0)               →  Drag                 ║
║  🤏 Pinch                  →  Double click          ║
║  👍 Thumbs up              →  Volume up            ║
║  👎 Thumbs down            →  Volume down          ║
║  ✌  Peace sign             →  Play / Pause         ║
║  🤘 Rock sign              →  Mute toggle          ║
║  ← Swipe left             →  Alt+Tab              ║
║  → Swipe right            →  Next track            ║
║                                                   ║
║  KEYS                                             ║
║  ────                                             ║
║  Q / Esc = Quit    H = Hide cam    O = Overlay    ║
║  F = Fullscreen    S = Screenshot                 ║
║                                                   ║
╚═══════════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load config.json from the script directory."""
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.json"
    )
    defaults = {
        "camera_index": 0,
        "camera_width": 640,
        "camera_height": 480,
        "smoothing": 5,
        "scroll_speed": 18,
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.6,
    }
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                user = json.load(f)
            defaults.update(user)
        except Exception as e:
            print(f"[warn] Could not read config.json: {e}")
    return defaults


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print(BANNER)

    cfg = load_config()

    # --- Open camera ---
    cap = cv2.VideoCapture(cfg["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["camera_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["camera_height"])

    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        print("  • Make sure no other app is using it")
        print(f"  • Camera index = {cfg['camera_index']}  (change in config.json)")
        return

    print("[✓] Camera opened — starting gesture control ...\n")

    # --- Components ---
    detector = GestureDetector(
        min_detection_confidence=cfg["min_detection_confidence"],
        min_tracking_confidence=cfg["min_tracking_confidence"],
    )
    controller = ActionController(
        scroll_speed=cfg["scroll_speed"],
        smoothing=cfg["smoothing"],
    )
    overlay = SkeletonOverlay()
    hud = HUD()

    # --- Window ---
    WIN = "Hand Gesture Control"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    camera_hidden = False
    fullscreen = False
    first_frame = True

    # FPS counter
    fps_time = time.time()
    fps_frames = 0
    fps = 0.0

    # ------------------------------------------------------------------ #
    #  Main loop                                                          #
    # ------------------------------------------------------------------ #
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror for selfie view
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # --- Detect gesture ---
            result = detector.process(frame)
            gesture = result.gesture

            # --- Execute action ---
            action = controller.execute(gesture, result)

            # Handle swipe
            if result.swipe_direction:
                controller.execute_swipe(result.swipe_direction)

            # --- Draw skeleton on frame ---
            if result.landmarks is not None:
                skel_color = GESTURE_COLORS_BGR.get(gesture, (128, 255, 0))
                draw_skeleton(frame, result.landmarks, skel_color)

            # --- Update overlay ---
            if result.landmarks is not None:
                hex_color = GESTURE_COLORS_HEX.get(gesture, "#00FF88")
                overlay.set_data([(result.landmarks, hex_color)], gesture)
            else:
                overlay.set_data([], "none")
            overlay.refresh()

            # --- FPS ---
            fps_frames += 1
            elapsed = time.time() - fps_time
            if elapsed >= 0.5:
                fps = fps_frames / elapsed
                fps_frames = 0
                fps_time = time.time()

            # --- Display ---
            if not camera_hidden:
                hud.draw(frame, gesture, result.finger_count,
                         result.confidence, controller.last_action, fps)
                cv2.imshow(WIN, frame)
                if first_frame:
                    cv2.resizeWindow(WIN, w, h)
                    first_frame = False
            else:
                cv2.imshow(WIN, np.zeros((1, 1, 3), dtype=np.uint8))

            # --- Keyboard ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27 or overlay.quit_requested():
                break
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WIN, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
            elif key == ord("h"):
                camera_hidden = not camera_hidden
            elif key == ord("o"):
                overlay.toggle()
            elif key == ord("s"):
                controller.take_screenshot()

    except KeyboardInterrupt:
        pass
    finally:
        controller.cleanup()
        cap.release()
        cv2.destroyAllWindows()
        overlay.destroy()
        detector.release()
        print("\n[✓] Hand Gesture Control stopped. Goodbye!\n")


if __name__ == "__main__":
    main()

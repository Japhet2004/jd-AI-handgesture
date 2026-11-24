"""
gesture_detector.py — Core gesture detection engine
=====================================================
Wraps MediaPipe Hands, classifies hand poses into named gestures,
and applies stability/debouncing so actions don't fire on flickery frames.
"""

from dataclasses import dataclass, field
from collections import deque
import math
import cv2
import mediapipe as mp
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GestureResult:
    """Result of gesture detection for a single frame."""
    gesture: str = "none"               # confirmed gesture name
    raw_gesture: str = "none"           # unfiltered gesture this frame
    finger_count: int = 0               # number of extended fingers
    fingers: list = field(default_factory=lambda: [0, 0, 0, 0, 0])
    landmarks: object = None            # MediaPipe NormalizedLandmarkList
    index_tip: tuple = (0.0, 0.0)       # normalized (x, y) of index tip
    middle_tip: tuple = (0.0, 0.0)      # normalized (x, y) of middle tip
    wrist: tuple = (0.0, 0.0)           # normalized (x, y) of wrist
    hand_center: tuple = (0.0, 0.0)     # center of hand bounding box
    confidence: float = 0.0             # stability 0–1
    hand_label: str = "Right"           # "Left" or "Right"
    swipe_direction: str = ""           # "left", "right", or ""
    pinch_distance: float = 1.0         # thumb-index distance


# ---------------------------------------------------------------------------
# Gesture Detector
# ---------------------------------------------------------------------------

class GestureDetector:
    """
    Processes camera frames through MediaPipe Hands and classifies
    the detected hand pose into one of the supported gestures.

    Supported gestures:
        pointer, click, right_click, scroll, idle, drag,
        pinch, thumbs_up, thumbs_down, peace, rock, none
    """

    # Landmark indices
    TIPS = [4, 8, 12, 16, 20]        # thumb, index, middle, ring, pinky tips
    PIPS = [3, 6, 10, 14, 18]        # joints just below each tip

    # How many consecutive identical frames before we confirm a gesture
    CONFIRM_FRAMES = {
        "default": 3,
        "pinch": 2,
        "thumbs_up": 2,
        "thumbs_down": 2,
        "drag": 3,
        "peace": 2,
        "rock": 2,
    }

    BUFFER_SIZE = 7  # rolling window for stability scoring

    def __init__(
        self,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
        max_num_hands: int = 1,
        pinch_threshold: float = 0.055,
    ):
        self._hands = mp.solutions.hands.Hands(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=max_num_hands,
        )
        self._pinch_threshold = pinch_threshold

        # Stability buffers
        self._gesture_buffer: deque = deque(maxlen=self.BUFFER_SIZE)
        self._confirmed_gesture: str = "none"
        self._consecutive_count: int = 0
        self._last_raw: str = "none"

        # Swipe tracking
        self._wrist_x_history: deque = deque(maxlen=12)
        self._swipe_cooldown: int = 0

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def process(self, bgr_frame: np.ndarray) -> GestureResult:
        """Process a BGR camera frame and return a GestureResult."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        if not results.multi_hand_landmarks:
            self._gesture_buffer.append("none")
            self._wrist_x_history.clear()
            if self._swipe_cooldown > 0:
                self._swipe_cooldown -= 1
            return self._make_result("none", "none")

        # Use the first detected hand
        hand_lm = results.multi_hand_landmarks[0]
        hand_label = "Right"
        if results.multi_handedness:
            hand_label = results.multi_handedness[0].classification[0].label

        # Key points
        lm = hand_lm.landmark
        index_tip = (lm[8].x, lm[8].y)
        middle_tip = (lm[12].x, lm[12].y)
        wrist = (lm[0].x, lm[0].y)
        hand_center = self._bounding_center(lm)

        # Finger states
        finger_count, fingers = self._count_fingers(hand_lm, hand_label)

        # Classify raw gesture
        raw_gesture = self._classify(hand_lm, finger_count, fingers, hand_label)

        # Swipe detection
        self._wrist_x_history.append(lm[0].x)
        swipe_dir = self._detect_swipe()

        # Pinch distance
        pinch_dist = self._pinch_distance(hand_lm)

        # Stability filtering
        self._gesture_buffer.append(raw_gesture)
        confirmed = self._apply_stability(raw_gesture)

        return GestureResult(
            gesture=confirmed,
            raw_gesture=raw_gesture,
            finger_count=finger_count,
            fingers=fingers,
            landmarks=hand_lm,
            index_tip=index_tip,
            middle_tip=middle_tip,
            wrist=wrist,
            hand_center=hand_center,
            confidence=self._confidence(),
            hand_label=hand_label,
            swipe_direction=swipe_dir,
            pinch_distance=pinch_dist,
        )

    def release(self):
        """Release MediaPipe resources."""
        self._hands.close()

    # ------------------------------------------------------------------ #
    #  Finger counting                                                    #
    # ------------------------------------------------------------------ #

    def _count_fingers(self, hand_landmarks, hand_label: str):
        """Count extended fingers. Returns (count, [thumb,idx,mid,ring,pinky])."""
        lm = hand_landmarks.landmark
        fingers = []

        # Thumb — x-axis comparison, depends on which hand
        # MediaPipe labels are from the camera's perspective (mirrored),
        # but we already flip the frame, so "Right" in label = user's right hand.
        if hand_label == "Right":
            fingers.append(1 if lm[4].x < lm[3].x else 0)
        else:
            fingers.append(1 if lm[4].x > lm[3].x else 0)

        # Index, middle, ring, pinky — y-axis (lower y = higher = extended)
        for tip_idx, pip_idx in zip(self.TIPS[1:], self.PIPS[1:]):
            fingers.append(1 if lm[tip_idx].y < lm[pip_idx].y else 0)

        return sum(fingers), fingers

    # ------------------------------------------------------------------ #
    #  Gesture classification                                             #
    # ------------------------------------------------------------------ #

    def _classify(self, hand_lm, finger_count: int, fingers: list, hand_label: str) -> str:
        """Classify raw gesture from finger states and landmark geometry."""
        lm = hand_lm.landmark

        # --- Special gestures checked first (geometry-based) ---

        # Pinch: thumb tip close to index tip, and not all fingers open
        pinch_dist = self._pinch_distance(hand_lm)
        if pinch_dist < self._pinch_threshold and finger_count <= 3:
            return "pinch"

        # Thumbs up: only thumb extended, tip above MCP
        if fingers[0] == 1 and sum(fingers[1:]) == 0:
            if lm[4].y < lm[2].y:  # tip above MCP
                return "thumbs_up"
            else:
                return "thumbs_down"

        # Rock / horns: index + pinky up, middle + ring down
        if fingers == [0, 1, 0, 0, 1] or fingers == [1, 1, 0, 0, 1]:
            return "rock"

        # Peace / V-sign: index + middle up, spread apart
        if fingers[1] == 1 and fingers[2] == 1 and sum(fingers[3:]) == 0:
            spread = self._finger_spread(lm, 8, 12)
            if spread > 0.06:
                return "peace"

        # --- Finger-count-based gestures ---
        if finger_count == 0:
            return "drag"
        elif finger_count == 1:
            if fingers[1] == 1:  # index finger
                return "pointer"
            return "idle"
        elif finger_count == 2:
            if fingers[1] == 1 and fingers[2] == 1:
                return "click"
            return "idle"
        elif finger_count == 3:
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
                return "right_click"
            return "idle"
        elif finger_count == 4:
            return "scroll"
        elif finger_count == 5:
            return "idle"

        return "idle"

    # ------------------------------------------------------------------ #
    #  Stability / debouncing                                             #
    # ------------------------------------------------------------------ #

    def _apply_stability(self, raw_gesture: str) -> str:
        """Confirm a gesture only after enough consecutive identical frames."""
        if raw_gesture == self._last_raw:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 1
            self._last_raw = raw_gesture

        needed = self.CONFIRM_FRAMES.get(
            raw_gesture, self.CONFIRM_FRAMES["default"]
        )

        if self._consecutive_count >= needed:
            self._confirmed_gesture = raw_gesture

        return self._confirmed_gesture

    def _confidence(self) -> float:
        """How stable the current gesture is (0–1)."""
        if not self._gesture_buffer:
            return 0.0
        most_common = max(set(self._gesture_buffer),
                          key=lambda g: list(self._gesture_buffer).count(g))
        return list(self._gesture_buffer).count(most_common) / len(self._gesture_buffer)

    # ------------------------------------------------------------------ #
    #  Geometry helpers                                                   #
    # ------------------------------------------------------------------ #

    def _pinch_distance(self, hand_lm) -> float:
        """Euclidean distance between thumb tip (4) and index tip (8)."""
        t = hand_lm.landmark[4]
        i = hand_lm.landmark[8]
        return math.sqrt((t.x - i.x) ** 2 + (t.y - i.y) ** 2)

    def _finger_spread(self, lm, idx_a: int, idx_b: int) -> float:
        """Distance between two landmark points."""
        a, b = lm[idx_a], lm[idx_b]
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def _bounding_center(self, lm) -> tuple:
        """Center of the hand's bounding box."""
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)

    # ------------------------------------------------------------------ #
    #  Swipe detection                                                    #
    # ------------------------------------------------------------------ #

    def _detect_swipe(self) -> str:
        """Detect horizontal swipe from wrist x-position history."""
        if self._swipe_cooldown > 0:
            self._swipe_cooldown -= 1
            return ""

        if len(self._wrist_x_history) < 8:
            return ""

        history = list(self._wrist_x_history)
        delta = history[-1] - history[0]

        if abs(delta) > 0.18:  # significant horizontal movement
            self._swipe_cooldown = 15  # cooldown frames
            self._wrist_x_history.clear()
            return "left" if delta < 0 else "right"

        return ""

    # ------------------------------------------------------------------ #
    #  Make a default result                                              #
    # ------------------------------------------------------------------ #

    def _make_result(self, gesture: str, raw: str) -> GestureResult:
        self._apply_stability(raw)
        return GestureResult(
            gesture=self._confirmed_gesture,
            raw_gesture=raw,
            confidence=self._confidence(),
        )

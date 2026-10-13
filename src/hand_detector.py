"""Hand detection, handedness and landmark drawing using MediaPipe (Phase 2.1).

Phase 2.1 changes:
- Up to **2 hands** are detected. 0, 1 and 2 hands are all normal outcomes and
  are handled safely (no hand is ever assumed to be present).
- MediaPipe's **handedness** output is exposed for every detected hand
  ("Left", "Right", or "Unknown" when the label is missing/ambiguous).
- **Both** hands are drawn, each with its own color and a text label.

Mirroring / coordinate system (important):
- The caller feeds frames that are already mirrored horizontally (see
  `scripts/test_hand_detection.py`, which calls `cv2.flip(frame, 1)` exactly
  once, right after the webcam frame is read).
- MediaPipe assumes a mirrored (selfie) input when it decides whether a hand
  is "Left" or "Right", so mirroring *before* detection gives correct labels.
- Because the flip happens once and before everything else, the preview, the
  drawn landmarks and the extracted ML features all use the SAME coordinate
  system. Nothing is flipped independently afterwards.
- Training and inference must use this exact same pipeline.

Note: MediaPipe removed the old `mediapipe.solutions` API in the versions that
support Python 3.13, so this module uses the newer Tasks API (`HandLandmarker`).
The Tasks API needs the pretrained model file `models/hand_landmarker.task`.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Standard MediaPipe topology: the 21 connections between the 21 hand
# landmarks. The topology is used exactly as MediaPipe defines it.
HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS

# Handedness category names reported by MediaPipe's handedness head.
HANDEDNESS_LEFT = "Left"
HANDEDNESS_RIGHT = "Right"
HANDEDNESS_UNKNOWN = "Unknown"

# A handedness label is only trusted when MediaPipe's confidence reaches this
# value. Below it (or when no label/score is returned at all) the hand is
# reported as "Unknown" instead of guessing.
MIN_HANDEDNESS_SCORE = 0.5

LANDMARKS_PER_HAND = 21
MAX_HANDS = 2

# BGR colors used to tell the two hands apart in the preview.
# Left = blue, Right = green, Unknown = orange.
HAND_COLORS = {
    HANDEDNESS_LEFT: (255, 0, 0),
    HANDEDNESS_RIGHT: (0, 255, 0),
    HANDEDNESS_UNKNOWN: (0, 165, 255),
}
POINT_COLORS = {
    HANDEDNESS_LEFT: (255, 128, 0),
    HANDEDNESS_RIGHT: (0, 200, 0),
    HANDEDNESS_UNKNOWN: (0, 200, 255),
}


@dataclass(frozen=True)
class DetectedHand:
    """One detected hand: its handedness plus its 21 MediaPipe landmarks.

    Attributes:
        label: "Left", "Right", or "Unknown" when MediaPipe's handedness
            result is missing or ambiguous (never guessed).
        handedness_score: MediaPipe's handedness confidence, or None when no
            score was returned.
        landmarks: Exactly 21 (x, y, z) tuples in MediaPipe's fixed topology
            order (0 = wrist, 1-4 thumb, 5-8 index, 9-12 middle,
            13-16 ring, 17-20 pinky), normalized to the frame that was passed
            to MediaPipe.
    """

    label: str
    handedness_score: Optional[float]
    landmarks: Tuple[Tuple[float, float, float], ...]

    @property
    def handedness_known(self) -> bool:
        """True when MediaPipe returned a usable Left/Right label."""
        return self.label in (HANDEDNESS_LEFT, HANDEDNESS_RIGHT)


def _handedness_for(handedness_per_hand: Sequence, index: int) -> Tuple[str, Optional[float]]:
    """Read MediaPipe's handedness entry for hand `index` safely.

    Returns ("Left"/"Right", score) when a confident, valid label exists and
    ("Unknown", score_or_None) otherwise.
    """
    if index >= len(handedness_per_hand):
        # MediaPipe returned no handedness entry for this hand
        return HANDEDNESS_UNKNOWN, None

    categories = handedness_per_hand[index]
    if not categories:
        return HANDEDNESS_UNKNOWN, None

    top = categories[0]  # highest-scoring category for this hand
    name = (getattr(top, "category_name", None) or "").strip()
    raw_score = getattr(top, "score", None)
    score = float(raw_score) if raw_score is not None else None

    if name not in (HANDEDNESS_LEFT, HANDEDNESS_RIGHT):
        # Unexpected category name -> do not guess
        return HANDEDNESS_UNKNOWN, score
    if score is not None and score < MIN_HANDEDNESS_SCORE:
        # Too little confidence -> report rather than guess
        return HANDEDNESS_UNKNOWN, score

    return name, score


def parse_hands(results) -> List[DetectedHand]:
    """Extract handedness and landmarks for every detected hand.

    Works for 0, 1 or 2 hands (never assumes a hand is present).

    Args:
        results: A MediaPipe `HandLandmarkerResult`, or None.

    Returns:
        A list of `DetectedHand` objects, empty when nothing was detected.
        Hands with an unexpected landmark count are skipped (with a warning)
        so that drawing and feature extraction can rely on 21 landmarks each.
    """
    if results is None:
        return []

    landmarks_per_hand = getattr(results, "hand_landmarks", None) or []
    handedness_per_hand = getattr(results, "handedness", None) or []

    hands = []
    for index, landmarks in enumerate(landmarks_per_hand):
        if len(landmarks) != LANDMARKS_PER_HAND:
            print(f"Warning: expected {LANDMARKS_PER_HAND} landmarks, "
                  f"got {len(landmarks)}. Ignoring this hand.")
            continue

        label, score = _handedness_for(handedness_per_hand, index)
        points = tuple((lm.x, lm.y, lm.z) for lm in landmarks)
        hands.append(DetectedHand(label=label, handedness_score=score,
                                  landmarks=points))

    return hands


class HandDetector:
    """Detects up to 2 hands in webcam frames and exposes their landmarks."""

    def __init__(self, model_path, max_hands=MAX_HANDS,
                 detection_confidence=0.5, tracking_confidence=0.5):
        """Load the MediaPipe hand landmarker model.

        Args:
            model_path: Path to the downloaded `hand_landmarker.task` file.
            max_hands: Maximum number of hands to detect (Phase 2.1 uses 2).
            detection_confidence: Minimum confidence for hand detection.
            tracking_confidence: Minimum confidence for hand tracking.
        """
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max(1, int(max_hands)),
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

        # Video mode requires strictly increasing timestamps, so we count frames
        self._frame_number = 0

    def detect(self, frame_bgr):
        """Detect hands in one BGR frame and return the raw MediaPipe result.

        The frame must already be mirrored horizontally if the preview is
        mirrored (see module docstring) so that handedness labels are correct.

        The result contains `.hand_landmarks` (one list of 21 landmarks per
        hand) and `.handedness` (one list of categories per hand). Use
        `parse_hands()` to turn it into `DetectedHand` objects.
        """
        # OpenCV gives BGR frames, MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Disabling writeable tells MediaPipe it does not need to copy the buffer
        rgb_frame.flags.writeable = False

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self._frame_number += 1
        timestamp_ms = self._frame_number * 33  # assume ~30 FPS

        return self._landmarker.detect_for_video(image, timestamp_ms)

    def draw_landmarks(self, frame_bgr, hands):
        """Draw every detected hand, its connections and its label (in place).

        Each hand uses its own color so Left and Right are visually distinct,
        plus a small "Left"/"Right"/"Unknown" text label above the hand. The
        MediaPipe connection topology is used unchanged.

        Args:
            frame_bgr: The frame to draw on (the mirrored frame, in this app).
            hands: A list of `DetectedHand` from `parse_hands()`. A raw
                MediaPipe result is also accepted for convenience.
        """
        if hands is None:
            return frame_bgr
        if hasattr(hands, "hand_landmarks"):  # raw MediaPipe result passed in
            hands = parse_hands(hands)

        height, width = frame_bgr.shape[:2]

        for hand in hands:
            line_color = HAND_COLORS.get(hand.label, HAND_COLORS[HANDEDNESS_UNKNOWN])
            point_color = POINT_COLORS.get(hand.label, POINT_COLORS[HANDEDNESS_UNKNOWN])

            # Landmarks are normalized (0.0 - 1.0); convert them to pixels
            points = [(int(x * width), int(y * height))
                      for (x, y, _z) in hand.landmarks]

            for connection in HAND_CONNECTIONS:
                cv2.line(frame_bgr, points[connection.start],
                         points[connection.end], line_color, 2)

            for point in points:
                cv2.circle(frame_bgr, point, 5, point_color, -1)

            # Hand label above the topmost landmark of this hand
            top_x = min(point[0] for point in points)
            top_y = min(point[1] for point in points)
            text_y = max(top_y - 15, 25)
            cv2.putText(frame_bgr, hand.label, (top_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)

        return frame_bgr

    def close(self):
        """Release the MediaPipe hand landmarker."""
        self._landmarker.close()

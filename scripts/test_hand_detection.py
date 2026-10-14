"""Phase 2.1 test: two-hand detection, handedness and the 126-value vector.

Opens the webcam, mirrors the preview horizontally (selfie view), detects up
to 2 hands with MediaPipe, draws BOTH hands with different colors/labels, and
shows the detection status on screen.

The three situations to test in front of the camera:

    TEST A - no hands visible
        Expected: "Hands detected: 0", both hand slots "Not detected".
    TEST B - one hand visible
        Expected: "Hands detected: 1", the handedness (Left/Right) is shown
        and that hand reports 21 landmarks.
    TEST C - both hands visible
        Expected: "Hands detected: 2", left/right identification shown,
        21 landmarks for each hand, and the 126-value fixed feature vector.

The program must not crash in any of the three situations.

Press Q in the video window to quit.

Usage (from the project folder):
    python scripts/test_hand_detection.py
"""

import sys
from pathlib import Path

import cv2

# Make the src/ package importable when this script is run from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import (
    HAND_FEATURE_LENGTH,
    MAX_FEATURE_LENGTH,
    extract_features,
    split_hands,
)
from src.hand_detector import HandDetector, parse_hands
from src import camera_lock

WINDOW_TITLE = "SignBridge AI - Hand Detection (Phase 2.1)"
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

# Download URL for Google's pretrained hand landmark model
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

# Colors (BGR) for the status text - blue for left, green for right
COLOR_LEFT = (255, 200, 0)
COLOR_RIGHT = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)


def describe_slot(slot_name, hand):
    """Human-readable status for one handedness slot (left or right).

    Returns (text, color, state); `state` is a short machine-friendly value
    used to detect real status changes (so a changing score alone does not
    spam the console).
    """
    if hand is None:
        return f"{slot_name} hand: Not detected", COLOR_RED, "empty"
    if not hand.handedness_known:
        return (f"{slot_name} hand: Detected (handedness unknown)",
                COLOR_YELLOW, "unknown")
    if hand.label == slot_name:
        score = hand.handedness_score
        score_text = f"{score:.2f}" if score is not None else "n/a"
        return (f"{slot_name} hand: Detected (score {score_text})",
                COLOR_LEFT if slot_name == "Left" else COLOR_RIGHT,
                f"labelled {hand.label}")
    # Rare case: MediaPipe gave this slot a hand labelled differently
    return (f"{slot_name} hand: Detected (MediaPipe label: {hand.label})",
            COLOR_YELLOW, f"slot occupied by {hand.label}")


def build_status(hands):
    """Compute the on-screen/console status for the current frame."""
    left_hand, right_hand = split_hands(hands)

    left_text, left_color, left_state = describe_slot("Left", left_hand)
    right_text, right_color, right_state = describe_slot("Right", right_hand)

    left_landmarks = len(left_hand.landmarks) if left_hand else 0
    right_landmarks = len(right_hand.landmarks) if right_hand else 0
    features = extract_features(hands)

    status = {
        "hand_count": len(hands),
        "left_text": left_text,
        "left_color": left_color,
        "left_state": left_state,
        "right_text": right_text,
        "right_color": right_color,
        "right_state": right_state,
        "left_landmarks": left_landmarks,
        "right_landmarks": right_landmarks,
        "feature_count": len(features),
    }
    return status


def print_status(status):
    """Print one readable status block to the console."""
    print(f"Hands detected: {status['hand_count']}")
    print(status["left_text"])
    print(status["right_text"])
    print(f"Landmarks: left {status['left_landmarks']}, "
          f"right {status['right_landmarks']}")
    print(f"Feature vector: {status['feature_count']} values "
          f"(left {HAND_FEATURE_LENGTH} + right {HAND_FEATURE_LENGTH})")
    if status["feature_count"] != MAX_FEATURE_LENGTH:
        print(f"Warning: expected {MAX_FEATURE_LENGTH} feature values, "
              f"got {status['feature_count']}!")
    print("-")


def main():
    # 1. Check that the MediaPipe model file was downloaded
    if not MODEL_PATH.exists():
        print("Error: hand landmark model not found.")
        print(f"Expected file: {MODEL_PATH}")
        print("Download it once with this PowerShell command:")
        print(f'  Invoke-WebRequest -Uri "{MODEL_URL}" '
              f'-OutFile "models\\hand_landmarker.task"')
        return

    # 2. Initialize MediaPipe (fails if the model file is invalid)
    try:
        detector = HandDetector(
            model_path=MODEL_PATH,
            max_hands=2,               # Phase 2.1 detects up to 2 hands
            detection_confidence=0.5,
            tracking_confidence=0.5,
        )
    except Exception as error:
        print("Error: could not initialize the MediaPipe hand detector.")
        print(error)
        return

    # 3. Open the default webcam
    # Advisory ownership (src/camera_lock.py): fail fast, naming the
    # SignBridge process that already owns the camera, instead of OpenCV
    # starving the reads.
    try:
        owned = camera_lock.acquire(0)
    except camera_lock.CameraBusyError as busy:
        print(f"Error: webcam 0 is already owned by another SignBridge "
              f"process (PID {busy.holder_pid}) - stop it first.")
        detector.close()
        return
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open the webcam.")
        print("Make sure a camera is connected and not being used by another application.")
        owned.release()
        detector.close()
        return

    print("Phase 2.1 hand detection started.")
    print("MediaPipe configuration: max 2 hands, detection confidence 0.5, "
          "tracking confidence 0.5")
    print("Preview: mirrored horizontally (cv2.flip(frame, 1) once per frame, "
          "before detection and display).")
    print("Feature vector: max 126 values = LEFT 63 + RIGHT 63 (missing hand = zeros).")
    print("Checks: TEST A = no hands, TEST B = one hand, TEST C = both hands.")
    print("Press Q in the video window to quit.")

    previous_signature = None
    total_frames = 0
    frames_by_count = {0: 0, 1: 0, 2: 0}
    frames_with_left = 0
    frames_with_right = 0

    try:
        # Read and process frames continuously
        while True:
            frame_ok, frame = camera.read()
            if not frame_ok:
                print("Error: Failed to read a frame from the webcam.")
                break

            # MIRROR THE PREVIEW (intentional, for a natural selfie view).
            # The frame is flipped exactly once, here, BEFORE detection, so the
            # preview, the drawn landmarks and the extracted ML features all
            # use the same coordinate system. MediaPipe also expects mirrored
            # input for correct Left/Right handedness.
            frame = cv2.flip(frame, 1)

            # Detect hands (0, 1 or 2), then draw both hands with labels
            results = detector.detect(frame)
            hands = parse_hands(results)
            detector.draw_landmarks(frame, hands)

            status = build_status(hands)

            # Only print to the console when the status really changes
            # (state without the score, so a flickering score is not spam)
            signature = (status["hand_count"], status["left_state"],
                         status["right_state"], status["left_landmarks"],
                         status["right_landmarks"])
            if signature != previous_signature:
                print_status(status)
                previous_signature = signature

            # --- on-screen status ---
            cv2.putText(frame, f"Hands detected: {status['hand_count']}",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        COLOR_WHITE, 2)
            cv2.putText(frame, status["left_text"], (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, status["left_color"], 2)
            cv2.putText(frame, status["right_text"], (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, status["right_color"], 2)
            cv2.putText(frame,
                        f"Landmarks: left {status['left_landmarks']}, "
                        f"right {status['right_landmarks']}",
                        (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        COLOR_WHITE, 2)
            cv2.putText(frame,
                        f"Feature vector: {status['feature_count']} values "
                        f"(left+right)",
                        (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        COLOR_YELLOW, 2)

            # Legend at the bottom of the frame
            height = frame.shape[0]
            cv2.putText(frame,
                        "Preview mirrored | Blue = left hand, "
                        "Green = right hand | Q = quit",
                        (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        COLOR_WHITE, 2)

            # Statistics for the final report
            total_frames += 1
            frames_by_count[min(status["hand_count"], 2)] = \
                frames_by_count.get(min(status["hand_count"], 2), 0) + 1
            if status["left_landmarks"] > 0:
                frames_with_left += 1
            if status["right_landmarks"] > 0:
                frames_with_right += 1

            cv2.imshow(WINDOW_TITLE, frame)

            # Wait 1 ms for a key press; quit when Q is pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        # Allows closing with Ctrl+C in the terminal as well
        print("\nStopped by user.")

    finally:
        # Always release the camera, close windows and free MediaPipe
        camera.release()
        owned.release()
        cv2.destroyAllWindows()
        detector.close()

        print("Camera released.")
        print(f"Frames processed: {total_frames}")
        print(f"Frames with 0 hands: {frames_by_count.get(0, 0)}")
        print(f"Frames with 1 hand : {frames_by_count.get(1, 0)}")
        print(f"Frames with 2 hands: {frames_by_count.get(2, 0)}")
        print(f"Frames with a left hand : {frames_with_left}")
        print(f"Frames with a right hand: {frames_with_right}")


if __name__ == "__main__":
    main()

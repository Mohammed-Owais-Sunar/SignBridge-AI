"""Phase 5 + 6: real-time gesture prediction and sentence building (webcam).

PIPELINE (identical feature representation to Phase 3 / Phase 4)

    webcam frame
      -> mirrored exactly once (cv2.flip(frame, 1), before detection)
      -> MediaPipe Hand Landmarker (Phase 2.1, up to 2 hands)
      -> handedness / LEFT-RIGHT ordering (split_hands)
      -> landmarks
      -> Phase 3 normalization (build_sample: wrist-centered, size-scaled,
         per hand, missing hand = 63 zeros)
      -> 126-value feature vector
      -> Random Forest (Phase 4, models/signbridge_random_forest.joblib)
      -> predicted gesture + confidence
      -> displayed on the preview
      -> stable displayed label -> SentenceBuilder (Phase 6) -> sentence

There is deliberately NO second feature/normalization implementation here:
the script imports build_sample() from src.feature_extractor - the exact
function that created dataset.csv and that Phase 4 evaluated.

CONFIDENCE
    Confidence = Random Forest predict_proba for the displayed class: the
    model's probability estimate for the current prediction. It is NOT an
    accuracy value. Frames whose best probability is below
    MIN_PREDICTION_CONFIDENCE (src/config.py, default 0.50) display
    "Uncertain".

STABILITY (minimal anti-flicker, display only)
    The displayed label is the majority vote of the last
    PREDICTION_VOTE_WINDOW raw predictions (src/config.py, default 5,
    0 = off). The classifier is never touched; the RAW per-frame prediction
    is always shown on the overlay and printed whenever it changes.

SENTENCE (Phase 6)
    The STABLE displayed prediction (post vote-window, post confidence
    threshold) is fed once per frame to src.sentence_builder.SentenceBuilder:
    word gestures append words, SPACE inserts exactly one separator, DELETE
    removes the most recent word. Actions are edge-triggered: a held gesture
    fires once; the SAME gesture re-arms only after a SUSTAINED release
    (hand absent >= REARM_NO_HAND_FRAMES frames, or below-threshold/
    unknown >= REARM_UNCERTAIN_FRAMES frames - so confidence flicker around
    the threshold cannot duplicate a held gesture), while a different
    gesture re-arms immediately. An action cooldown adds a final rate bound.

WHAT THIS SCRIPT DOES NOT DO (later phases)
    - no text-to-speech / audio (Phase 7), no Streamlit / web UI (Phase 8),
      no deployment.

KEYS (click the video window first)
    q    quit (Ctrl+C works too)
    c    clear the sentence

USAGE (from the project folder)
    python scripts/predict_realtime.py
    python scripts/predict_realtime.py --seconds 10   # auto-quit (testing)
"""

import argparse
import math
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np

# Make the src/ package importable when this script is run from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (CLASSES_PATH, MIN_PREDICTION_CONFIDENCE,
                        MODEL_PATH as HAND_MODEL_PATH, MODEL_URL,
                        PREDICTION_N_JOBS, PREDICTION_VOTE_WINDOW,
                        SIGN_CLASSES, TRAINED_MODEL_PATH)
from src.feature_extractor import (HAND_FEATURE_LENGTH, MAX_FEATURE_LENGTH,
                                   build_sample, split_hands,
                                   validate_feature_vector)
from src.hand_detector import DetectedHand, HandDetector, parse_hands
from src.sentence_builder import SentenceBuilder

WINDOW_TITLE = "SignBridge AI - Prediction + Sentence (Phase 6)"

# Colors (BGR)
COLOR_WHITE = (255, 255, 255)
COLOR_SHADOW = (0, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_DIM = (190, 190, 190)


# -----------------------------------------------------------------------------
# Model + class loading
# -----------------------------------------------------------------------------
def load_classifier():
    """Load the Random Forest and the class list; validate both.

    Returns (model, class_names) where class_names[i] is the label that
    predict_proba column i refers to.

    Raises SystemExit with a readable reason on any mismatch, so a wrong
    model can never silently produce wrong labels.
    """
    if not TRAINED_MODEL_PATH.exists():
        raise SystemExit(
            f"Error: trained model not found: {TRAINED_MODEL_PATH}\n"
            f"    Train it first: python scripts/train_model.py")
    if not CLASSES_PATH.exists():
        raise SystemExit(
            f"Error: class list not found: {CLASSES_PATH}\n"
            f"    Train it first: python scripts/train_model.py")

    try:
        model = joblib.load(TRAINED_MODEL_PATH)
        saved_classes = [str(label) for label in joblib.load(CLASSES_PATH)]
    except Exception as error:  # noqa: BLE001 - any load failure is fatal
        raise SystemExit(f"Error: could not load the model files: {error}")

    model_classes = [str(label) for label in getattr(model, "classes_", [])]

    if not model_classes:
        raise SystemExit("Error: the loaded model has no classes_.")
    if model_classes != saved_classes:
        raise SystemExit(
            f"Error: {CLASSES_PATH} does not match model.classes_.\n"
            f"    classes file : {saved_classes}\n"
            f"    model        : {model_classes}\n"
            f"    Re-run: python scripts/train_model.py")
    if set(model_classes) != set(SIGN_CLASSES):
        raise SystemExit(
            f"Error: model classes differ from Phase 3 SIGN_CLASSES.\n"
            f"    model : {sorted(model_classes)}\n"
            f"    Phase3: {sorted(SIGN_CLASSES)}")
    n_features = getattr(model, "n_features_in_", None)
    if n_features != MAX_FEATURE_LENGTH:
        raise SystemExit(
            f"Error: model expects {n_features} features, "
            f"the pipeline produces {MAX_FEATURE_LENGTH}.")

    # Display-time parallelism (documented in src/config.py): single-sample
    # queries are much faster serially and produce BIT-IDENTICAL
    # probabilities. Only this in-memory object is changed; the saved file
    # keeps its training settings.
    model.n_jobs = PREDICTION_N_JOBS

    return model, model_classes


# -----------------------------------------------------------------------------
# Prediction (features -> label + confidence)
# -----------------------------------------------------------------------------
def predict_gesture(model, features, class_names):
    """Predict ONE 126-value sample.

    Returns (raw_label, confidence, proba_dict):
        raw_label   - the class with the highest probability
        confidence  - that probability (float in [0, 1])
        proba_dict  - {class name: probability} for every class

    Raises ValueError when the feature vector is not exactly 126 finite
    values - the caller then shows the reason instead of predicting.
    """
    validate_feature_vector(list(features), MAX_FEATURE_LENGTH)
    probabilities = model.predict_proba([features])[0]

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("model returned non-finite probabilities")
    total = float(probabilities.sum())
    if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"probabilities sum to {total}, expected 1.0")

    best_index = int(np.argmax(probabilities))
    proba_dict = {name: float(value)
                  for name, value in zip(class_names, probabilities)}
    return class_names[best_index], float(probabilities[best_index]), proba_dict


class VoteWindow:
    """Majority vote over the last N raw predictions (display smoothing).

    Purely cosmetic: it never changes what the classifier predicted - the
    raw label is preserved separately for debugging.
    """

    def __init__(self, size: int):
        self.size = max(0, int(size))
        self._frames = deque(maxlen=self.size) if self.size else None

    def push(self, label: str) -> None:
        if self._frames is not None:
            self._frames.append(label)

    def result(self):
        """Majority label, or None when the window is off/empty.

        Ties are broken deterministically in favour of the most recent
        prediction (newest first).
        """
        if not self._frames:
            return None
        counts = Counter(self._frames)
        best = max(counts.values())
        for label in reversed(self._frames):  # newest first
            if counts[label] == best:
                return label
        return None

    def clear(self) -> None:
        if self._frames is not None:
            self._frames.clear()


# -----------------------------------------------------------------------------
# Startup self-check: feature layout + model contract (before the camera)
# -----------------------------------------------------------------------------
def _synthetic_hand(label: str, origin) -> DetectedHand:
    landmarks = tuple(
        (origin[0] + 0.011 * index,
         origin[1] + 0.017 * index,
         origin[2] - 0.001 * index)
        for index in range(21))
    return DetectedHand(label=label, handedness_score=0.99, landmarks=landmarks)


def startup_self_check(model, class_names) -> bool:
    """Verify the Phase 3 feature layout and the model contract.

    Runs instantly and deterministically (no camera, no MediaPipe inference).
    """
    print("\nStartup self-check:")
    checks = []

    left = _synthetic_hand("Left", (0.70, 0.35, 0.0))
    right = _synthetic_hand("Right", (0.30, 0.55, 0.0))

    # 126 features, finite
    features = build_sample([left, right])
    checks.append(("build_sample produces exactly 126 finite features",
                   len(features) == MAX_FEATURE_LENGTH
                   and all(math.isfinite(value) for value in features),
                   f"len={len(features)}"))

    # LEFT label -> left block (indices 0-62), RIGHT label -> right block
    left_block = features[:HAND_FEATURE_LENGTH]
    right_block = features[HAND_FEATURE_LENGTH:]
    checks.append(("LEFT-hand block (0-62) populated for a 'Left' hand",
                   any(abs(value) > 1e-12 for value in left_block), ""))
    checks.append(("RIGHT-hand block (63-125) populated for a 'Right' hand",
                   any(abs(value) > 1e-12 for value in right_block), ""))

    # Detection order must not matter (Phase 2.1 rule)
    swapped = build_sample([right, left])
    checks.append(("feature vector is identical regardless of detection order",
                   swapped == features, ""))

    # No hand -> 126 zeros -> never predicted at runtime
    checks.append(("0 hands -> 126 zeros (runtime must skip prediction)",
                   all(value == 0.0 for value in build_sample([])), ""))

    # Model contract
    checks.append((f"model expects {MAX_FEATURE_LENGTH} features",
                   getattr(model, "n_features_in_", None)
                   == MAX_FEATURE_LENGTH,
                   str(getattr(model, "n_features_in_", None))))
    checks.append(("class mapping == Phase 3 SIGN_CLASSES",
                   set(class_names) == set(SIGN_CLASSES), ""))

    try:
        raw_label, confidence, proba = predict_gesture(
            model, features, class_names)
        checks.append(("model accepts the vector, label is a known class",
                       raw_label in SIGN_CLASSES, raw_label))
        checks.append(("confidence is finite and in [0, 1]",
                       math.isfinite(confidence)
                       and 0.0 <= confidence <= 1.0,
                       f"{confidence:.4f}"))
        checks.append(("all class probabilities finite, sum to 1",
                       len(proba) == len(SIGN_CLASSES)
                       and all(math.isfinite(value) for value in proba.values())
                       and math.isclose(sum(proba.values()), 1.0,
                                        rel_tol=1e-4, abs_tol=1e-4), ""))
    except Exception as error:  # noqa: BLE001 - report as a failed check
        checks.append(("model accepts the vector, label is a known class",
                       False, str(error)))

    for name, ok, detail in checks:
        suffix = f"  ({detail})" if detail else ""
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}{suffix}")

    return all(ok for _, ok, _ in checks)


# -----------------------------------------------------------------------------
# Overlay
# -----------------------------------------------------------------------------
def put_text(frame, text, position, color, scale=0.7, thickness=2):
    """Draw text with a dark outline so it stays readable on any background."""
    x, y = position
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, COLOR_SHADOW, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_overlay(frame, state: dict) -> None:
    height = frame.shape[0]

    put_text(frame, f"Prediction: {state['prediction']}", (10, 38),
             state["prediction_color"], 1.0, 2)
    if state["confidence_text"]:
        put_text(frame, f"Confidence: {state['confidence_text']}", (10, 72),
                 state["confidence_color"], 0.75, 2)
    if state["raw_text"]:
        put_text(frame, f"Raw (this frame): {state['raw_text']}", (10, 102),
                 COLOR_DIM, 0.6, 1)

    put_text(frame,
             f"Hands: {state['hand_count']}   "
             f"slots L={state['left_landmarks']} R={state['right_landmarks']}"
             f"   features: {state['feature_count']}",
             (10, 132), COLOR_WHITE, 0.6, 2)
    put_text(frame,
             f"FPS: {state['fps']:.1f}   prediction latency: "
             f"{state['latency_ms']:.2f} ms   "
             f"(vote window: {state['window_size']}, "
             f"min confidence: {state['min_confidence']:.2f})",
             (10, 162), COLOR_WHITE, 0.55, 1)

    if state["note"]:
        put_text(frame, state["note"], (10, 196), COLOR_YELLOW, 0.6, 1)

    # Sentence (Phase 6). If it outgrows the frame width, show the tail so
    # the most recent words always stay visible (full text is in the log).
    sentence = state.get("sentence", "")
    visible = sentence if len(sentence) <= 56 else "..." + sentence[-53:]
    put_text(frame, f"Sentence: {visible if visible else '(empty)'}",
             (10, 240), COLOR_CYAN, 0.8, 2)

    put_text(frame,
             "Mirrored preview | gestures build the sentence | "
             "q = quit | c = clear sentence",
             (10, height - 14), COLOR_WHITE, 0.55, 1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time gesture prediction + sentence building "
                    "(Phases 5/6). No TTS, no web UI.")
    parser.add_argument("--seconds", type=int, default=0, metavar="S",
                        help="quit automatically after S seconds "
                             "(0 = run until q is pressed)")
    args = parser.parse_args()
    if args.seconds < 0:
        parser.error("--seconds must not be negative")
    return args


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("SIGNBRIDGE AI - PREDICTION + SENTENCE (Phases 5/6)")
    print("=" * 60)

    # 1-2. Load model + class mapping --------------------------------------
    model, class_names = load_classifier()
    print(f"Model       : {TRAINED_MODEL_PATH}")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    print(f"Features    : {MAX_FEATURE_LENGTH} "
          f"(LEFT {HAND_FEATURE_LENGTH} + RIGHT {HAND_FEATURE_LENGTH}), "
          f"Phase 3 normalization via build_sample()")
    print(f"Prediction n_jobs: {PREDICTION_N_JOBS} "
          f"(display-time only, identical probabilities - see config.py)")

    if not startup_self_check(model, class_names):
        print("\nError: startup self-check FAILED - not starting the camera.")
        return 1

    # 3. Webcam -------------------------------------------------------------
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: could not open the webcam.")
        print("Make sure a camera is connected and not used by another "
              "application.")
        return 1

    # MediaPipe hand model (Phase 2)
    if not HAND_MODEL_PATH.exists():
        print(f"Error: hand landmark model not found: {HAND_MODEL_PATH}")
        print("Download it once with this PowerShell command:")
        print(f'  Invoke-WebRequest -Uri "{MODEL_URL}" '
              f'-OutFile "models\\hand_landmarker.task"')
        camera.release()
        return 1
    try:
        detector = HandDetector(model_path=HAND_MODEL_PATH, max_hands=2,
                                detection_confidence=0.5,
                                tracking_confidence=0.5)
    except Exception as error:  # noqa: BLE001 - MediaPipe init failure
        print("Error: could not initialize the MediaPipe hand detector.")
        print(error)
        camera.release()
        return 1

    print("\nPreview: mirrored exactly once (cv2.flip(frame, 1) before "
          "detection) - same as Phase 2.1/3/4.")
    print(f"Display: confidence threshold {MIN_PREDICTION_CONFIDENCE:.2f} "
          f"('Uncertain' below it), vote window "
          f"{PREDICTION_VOTE_WINDOW} frame(s).")
    vote = VoteWindow(PREDICTION_VOTE_WINDOW)
    builder = SentenceBuilder()

    print("Sentence: word gestures add words, SPACE adds one separator, "
          "DELETE removes the last word.")
    print(f"          edge-trigger + sustained re-arm (no-hand >= "
          f"{builder.no_hand_frames} frames, uncertain >= "
          f"{builder.uncertain_frames} frames), action "
          f"cooldown {builder.cooldown_seconds:.2f}s "
          f"(src/sentence_builder.py).")
    print("Scope: no speech/audio (Phase 7), no web UI (Phase 8).")
    print("Keys: q = quit, c = clear the sentence.\n")

    start_time = time.monotonic()
    next_fps_time = start_time

    stats = {
        "frames": 0,
        "frames_no_hand": 0,
        "frames_predicted": 0,
        "frames_uncertain": 0,
        "latency_sum_ms": 0.0,
        "latency_count": 0,
        "latency_max_ms": 0.0,
        "fps_samples": [],
        "displayed": Counter(),          # displayed label -> frames
        "raw": Counter(),                # raw label -> frames
        "low_confidence_raw": Counter(), # raw label -> frames below threshold
        "actions": Counter(),            # sentence actions -> count
    }

    previous_display_signature = None
    previous_hand_signature = None
    fps_ema = None
    latency_ema = None
    exit_code = 0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Error: failed to read a frame from the webcam.")
                exit_code = 1
                break

            # MIRROR ONCE, before detection (Phase 2.1 pipeline - identical
            # for training in Phase 3/4 and prediction here).
            frame = cv2.flip(frame, 1)

            results = detector.detect(frame)
            hands = parse_hands(results)
            detector.draw_landmarks(frame, hands)

            now = time.monotonic()
            stats["frames"] += 1

            if args.seconds and now - start_time >= args.seconds:
                print(f"\nTime limit of {args.seconds} s reached.")
                break

            left_hand, right_hand = split_hands(hands)

            # Console log on hand/slot/handedness changes (left/right check)
            hand_signature = (
                len(hands),
                len(left_hand.landmarks) if left_hand else 0,
                len(right_hand.landmarks) if right_hand else 0,
                tuple(sorted(hand.label for hand in hands)),
            )
            if hand_signature != previous_hand_signature:
                print(f"[{now - start_time:6.1f}s] hands={len(hands)} "
                      f"slots: L={hand_signature[1]} R={hand_signature[2]} "
                      f"handedness={list(hand_signature[3])}")
                previous_hand_signature = hand_signature

            # --- feature extraction + prediction ---------------------------
            prediction = "No hand detected"
            confidence_text = ""
            raw_text = ""
            note = ""
            prediction_color = COLOR_RED
            confidence_color = COLOR_WHITE
            feature_count = 0

            if not hands:
                stats["frames_no_hand"] += 1
                # No prediction from an all-zero vector (Phase 5 rule)
                vote.clear()  # do not carry stale votes across absences
            else:
                prediction_started = time.perf_counter()  # features + model
                try:
                    features = build_sample(hands)   # Phase 3 pipeline
                    feature_count = len(features)
                    raw_label, raw_confidence, proba = predict_gesture(
                        model, features, class_names)
                except ValueError as error:
                    prediction = "Sample rejected"
                    prediction_color = COLOR_YELLOW
                    note = str(error)
                else:
                    latency_ms = (time.perf_counter() - prediction_started) \
                        * 1000.0
                    stats["latency_sum_ms"] += latency_ms
                    stats["latency_count"] += 1
                    stats["latency_max_ms"] = max(stats["latency_max_ms"],
                                                  latency_ms)
                    latency_ema = (latency_ms if latency_ema is None
                                   else 0.9 * latency_ema + 0.1 * latency_ms)

                    stats["frames_predicted"] += 1
                    stats["raw"][raw_label] += 1
                    raw_text = (f"{raw_label} "
                                f"({raw_confidence * 100:.1f}%, "
                                f"1/{len(class_names)} chance = "
                                f"{100.0 / len(class_names):.1f}%)")

                    vote.push(raw_label)
                    stable = vote.result()

                    if raw_confidence < MIN_PREDICTION_CONFIDENCE:
                        # Documented threshold (src/config.py)
                        stats["frames_uncertain"] += 1
                        stats["low_confidence_raw"][raw_label] += 1
                        prediction = "Uncertain"
                        prediction_color = COLOR_YELLOW
                        confidence_text = f"{raw_confidence * 100:.1f}%"
                        confidence_color = COLOR_YELLOW
                        note = (f"below threshold "
                                f"{MIN_PREDICTION_CONFIDENCE:.2f} - raw "
                                f"prediction shown for debugging")
                    else:
                        displayed = stable or raw_label
                        prediction = displayed
                        prediction_color = COLOR_GREEN
                        confidence = proba.get(displayed, raw_confidence)
                        confidence_text = f"{confidence * 100:.1f}%"
                        confidence_color = COLOR_WHITE
                        if stable and stable != raw_label:
                            note = (f"stable = majority of last "
                                    f"{PREDICTION_VOTE_WINDOW} frames "
                                    f"(raw this frame: {raw_label})")
                    stats["displayed"][prediction] += 1

                    # Console log only when the displayed prediction changes
                    display_signature = (prediction,
                                         round(raw_confidence, 2))
                    if display_signature != previous_display_signature:
                        print(f"[{now - start_time:6.1f}s] displayed="
                              f"{prediction}  raw={raw_label} "
                              f"conf={raw_confidence:.3f}")
                        previous_display_signature = display_signature

            # --- Phase 6: sentence building from the STABLE displayed label.
            # observe() is edge-triggered: a held gesture fires once and only
            # re-arms after a different gesture or a release state
            # ("Uncertain"/"No hand detected"/unknown -> nothing, re-arm).
            action = builder.observe(prediction)
            if action:
                print(f"[{now - start_time:6.1f}s] action: {action}  ->  "
                      f"\"{builder.sentence}\"")
                stats["actions"][action.split(maxsplit=1)[0]] += 1

            # --- FPS (real, measured from frame-to-frame time) -------------
            if fps_ema is None:
                fps_ema = 0.0
            if stats["frames"] > 1:
                instantaneous = 1.0 / max(now - previous_frame_time, 1e-6)
                fps_ema = (instantaneous if fps_ema == 0.0
                           else 0.9 * fps_ema + 0.1 * instantaneous)
            previous_frame_time = now
            if now >= next_fps_time:
                stats["fps_samples"].append(fps_ema)
                next_fps_time = now + 1.0

            draw_overlay(frame, {
                "prediction": prediction,
                "prediction_color": prediction_color,
                "confidence_text": confidence_text,
                "confidence_color": confidence_color,
                "raw_text": raw_text,
                "hand_count": len(hands),
                "left_landmarks": len(left_hand.landmarks) if left_hand else 0,
                "right_landmarks": len(right_hand.landmarks) if right_hand else 0,
                "feature_count": feature_count,
                "fps": fps_ema or 0.0,
                "latency_ms": latency_ema or 0.0,
                "window_size": PREDICTION_VOTE_WINDOW,
                "min_confidence": MIN_PREDICTION_CONFIDENCE,
                "note": note,
                "sentence": builder.sentence,
            })

            cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(1) & 0xFF
            if key < 256:
                char = chr(key).lower()
                if char == "q":
                    break
                if char == "c":
                    builder.clear()
                    print(f"[{now - start_time:6.1f}s] CLEAR (c) -> "
                          f"sentence: \"{builder.sentence}\"")

    except KeyboardInterrupt:
        print("\nInterrupted - shutting down safely.")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.close()

        # ---------------- session summary (all values measured) ------------
        elapsed = time.monotonic() - start_time
        print("\n" + "=" * 60)
        print("PHASE 5 + 6 SESSION SUMMARY")
        print("=" * 60)
        print(f"Session length        : {elapsed:.1f} s")
        print(f"Frames processed      : {stats['frames']}")
        if stats["frames"] > 0:
            print(f"Frames without a hand : {stats['frames_no_hand']} "
                  f"(displayed 'No hand detected', no prediction)")
            print(f"Frames with prediction: {stats['frames_predicted']}")
            print(f"Frames 'Uncertain'    : {stats['frames_uncertain']} "
                  f"(below {MIN_PREDICTION_CONFIDENCE:.2f})")
        if stats["latency_count"]:
            print(f"Prediction latency    : avg "
                  f"{stats['latency_sum_ms'] / stats['latency_count']:.2f} ms "
                  f"max {stats['latency_max_ms']:.2f} ms "
                  f"({stats['latency_count']} predictions, "
                  f"features + model only)")
        if stats["fps_samples"]:
            average_fps = sum(stats["fps_samples"]) / len(stats["fps_samples"])
            print(f"FPS (per-second EMA)  : avg {average_fps:.1f}  "
                  f"last {fps_ema or 0.0:.1f}")
        if stats["raw"]:
            print("Raw predictions       : "
                  + ", ".join(f"{name} {count}"
                              for name, count in stats["raw"].most_common()))
        if stats["displayed"]:
            print("Displayed predictions : "
                  + ", ".join(f"{name} {count}"
                              for name, count in stats["displayed"].most_common()))
        if stats["low_confidence_raw"]:
            print("Low-confidence frames : "
                  + ", ".join(f"{name} {count}" for name, count in
                              stats["low_confidence_raw"].most_common()))
        print(f"Sentence              : \"{builder.sentence}\"")
        if stats["actions"]:
            print("Sentence actions       : "
                  + ", ".join(f"{name} {count}"
                              for name, count in stats["actions"].most_common()))
        print("No speech/audio was produced (TTS belongs to Phase 7).")
        print("=" * 60)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

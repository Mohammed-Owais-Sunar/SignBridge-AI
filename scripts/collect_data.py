"""Phase 3 data collection: save landmark feature samples from the webcam.

WHAT THIS SCRIPT DOES
- Shows the MIRRORED webcam preview with the detected hand(s) drawn on it
  (the exact Phase 2.1 pipeline: `cv2.flip(frame, 1)` once, right after
  capture, before detection - no second flip anywhere).
- Lets you select one of the 12 sign classes and start/pause collection.
- Every SAMPLE_INTERVAL_MS (default 150 ms) it saves ONE sample:
  the normalized 126-value landmark vector + the label, appended to
  data/processed/dataset.csv. Nothing else is stored - no images, no video.

WHAT THIS SCRIPT DOES NOT DO
- No model is trained, no accuracy is computed (that is Phase 4).

SAFETY RULES (a sample is rejected - and the reason shown - when ...)
- no class is selected / the label is not one of SIGN_CLASSES
- no hand is detected in the frame
- landmark/feature extraction fails (e.g. hand size reference is zero)
- the vector is not exactly 126 finite numbers
- the class already reached SAMPLES_PER_CLASS samples
- the row would be an EXACT duplicate of an already saved row

RESUME SUPPORT
- Existing samples are loaded at startup and shown as progress; they are
  never deleted or overwritten. If the terminal is closed mid-run, every
  sample already written is still in the CSV (each row is flushed
  immediately), and the next run simply continues counting from there.

KEYS (click the video window first so it receives the keys)
    n / p        next / previous sign class
    1..9, 0, -,  jump directly to class 1..12
    c            start / pause collection
    q            quit (saved samples are kept)

USAGE (from the project folder):
    python scripts/collect_data.py
    python scripts/collect_data.py --class HELLO --auto-collect --max-new 20

The optional flags make small, resumable batches easy to run:
    --class NAME       pre-select a sign class
    --auto-collect     start collecting immediately (requires --class)
    --max-new N        quit after N NEW samples were saved in this run
    --seconds S        quit automatically after S seconds
"""

import argparse
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2

# Make the src/ package importable when this script is run from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    COLLECTION_TIPS,
    DATASET_CSV,
    MODEL_PATH,
    MODEL_URL,
    SAMPLE_INTERVAL_MS,
    SAMPLES_PER_CLASS,
    SIGN_CLASSES,
)
from src.dataset_io import DatasetError, DatasetWriter, load_progress
from src.feature_extractor import (
    HAND_FEATURE_LENGTH,
    MAX_FEATURE_LENGTH,
    build_sample,
    split_hands,
)
from src.hand_detector import HandDetector, parse_hands

WINDOW_TITLE = "SignBridge AI - Data Collection (Phase 3)"

# Direct class selection keys -> index inside SIGN_CLASSES
QUICK_SELECT_KEYS = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5,
                     "7": 6, "8": 7, "9": 8, "0": 9, "-": 10, "=": 11}

# Seconds each collection tip stays on screen
TIP_SECONDS = 4.0
# Minimum seconds between two identical rejection messages on the console
REJECT_LOG_INTERVAL_S = 5.0

# Colors (BGR)
COLOR_WHITE = (255, 255, 255)
COLOR_SHADOW = (0, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_DIM = (190, 190, 190)

VALID_LABELS = set(SIGN_CLASSES)
TARGET_SAMPLES = SAMPLES_PER_CLASS  # per class


# -----------------------------------------------------------------------------
# Sample acceptance (the one place that decides what may enter dataset.csv)
# -----------------------------------------------------------------------------
def assess_sample(hands, label, counts, seen):
    """Decide whether the current frame may become a dataset row.

    Args:
        hands: parsed hands from this frame (0, 1 or 2 DetectedHand).
        label: the selected sign class, or None when nothing is selected.
        counts: mapping label -> number of samples already saved.
        seen: set of exact rows already saved, as (label, tuple(features)).

    Returns:
        (accepted, reason, sample):
        accepted=True  -> `reason` is a message, `sample` holds the 126
                          normalized values ready to be written.
        accepted=False -> `reason` explains why the sample must NOT be saved.
    """
    if not label:
        return False, "no class selected", None
    if label not in VALID_LABELS:
        return False, f"unknown class '{label}'", None
    if not hands:
        return False, "no hand detected", None

    try:
        sample = build_sample(hands)
    except ValueError as error:
        return False, str(error), None

    if len(sample) != MAX_FEATURE_LENGTH:
        return False, (f"feature vector has {len(sample)} values, expected "
                       f"{MAX_FEATURE_LENGTH}"), None
    if any(not math.isfinite(value) for value in sample):
        return False, "feature vector contains NaN or infinite values", None

    current = counts.get(label, 0)
    if current >= TARGET_SAMPLES:
        return False, (f"{label} already has {current} samples "
                       f"(target {TARGET_SAMPLES} reached)"), None

    if (label, tuple(sample)) in seen:
        return False, "exact duplicate of an already saved row", None

    return True, f"ready: {label} {current + 1}/{TARGET_SAMPLES}", sample


# -----------------------------------------------------------------------------
# Console output
# -----------------------------------------------------------------------------
def print_startup_info(args, progress) -> None:
    """Print the class list, guidance and existing progress once."""
    print("=" * 50)
    print("SIGNBRIDGE AI - DATA COLLECTION (Phase 3)")
    print("=" * 50)
    print(f"Dataset file : {DATASET_CSV}")
    print(f"Classes ({len(SIGN_CLASSES)}):")
    for index, name in enumerate(SIGN_CLASSES, start=1):
        print(f"    {index:>2}. {name}")
    print(f"Target       : {TARGET_SAMPLES} samples per class "
          f"({TARGET_SAMPLES * len(SIGN_CLASSES)} total)")
    print(f"Capture every: {SAMPLE_INTERVAL_MS} ms, "
          f"{MAX_FEATURE_LENGTH} features per sample "
          f"(LEFT {HAND_FEATURE_LENGTH} + RIGHT {HAND_FEATURE_LENGTH})")
    print("Features are NORMALIZED (wrist-centered, hand-size scaled) before "
          "saving.")
    print("Only numerical landmarks are saved - no images.")
    print()
    print("Tips for good samples:")
    for tip in COLLECTION_TIPS:
        print(f"    - {tip}")
    print()
    print("Keys: n/p = change class | 1-9,0,-,= = jump to class | "
          "c = start/pause | q = quit")

    # Existing progress (resume support)
    if progress.total_rows == 0 and not progress.unknown_labels:
        print("\nExisting progress: none (a new dataset.csv will be created "
              "when the first sample is saved).")
    else:
        print(f"\nExisting progress: {progress.valid_rows} saved sample(s) "
              f"found - collection will RESUME from there.")
        for name in SIGN_CLASSES:
            if progress.counts.get(name):
                print(f"    {name:<12} {progress.counts[name]:>4} / "
                      f"{TARGET_SAMPLES}")
    if progress.malformed:
        lines = ", ".join(str(line) for line, _ in progress.malformed[:5])
        print(f"Warning: {len(progress.malformed)} malformed row(s) in the "
              f"CSV (line {lines}{'...' if len(progress.malformed) > 5 else ''}). "
              f"They are ignored for progress and left untouched - run "
              f"scripts/validate_dataset.py for details.")
    if progress.unknown_labels:
        print(f"Warning: unknown label(s) found: "
              f"{', '.join(sorted(progress.unknown_labels))} "
              f"(kept, but not counted towards any class).")
    if args.class_name:
        print(f"\nPre-selected class: {args.class_name} (--class)")


def print_dataset_summary(counts) -> None:
    """The end-of-session dataset summary (Phase 3 spec, section 23)."""
    total = sum(counts.get(name, 0) for name in SIGN_CLASSES)
    target_total = TARGET_SAMPLES * len(SIGN_CLASSES)

    print()
    print("=" * 32)
    print("SIGNBRIDGE AI DATASET")
    print("=" * 32)
    for name in SIGN_CLASSES:
        print(f"{name:<12}{counts.get(name, 0):>4} / {TARGET_SAMPLES}")
    print("-" * 32)
    print(f"Total samples: {total}")
    print(f"Target samples: {target_total}")
    print("=" * 32)

    incomplete = [name for name in SIGN_CLASSES
                  if counts.get(name, 0) < TARGET_SAMPLES]
    if incomplete:
        print(f"INCOMPLETE: {len(incomplete)} class(es) still below target: "
              f"{', '.join(incomplete)}")
        print("Do NOT start model training yet - collect the missing "
              "samples first.")
    else:
        print("All classes reached the target sample count.")
    print(f"Dataset file: {DATASET_CSV}")


# -----------------------------------------------------------------------------
# On-screen overlay
# -----------------------------------------------------------------------------
def put_text(frame, text, position, color, scale=0.6, thickness=2):
    """Draw text with a dark outline so it stays readable on any background."""
    x, y = position
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, COLOR_SHADOW, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


@dataclass
class Overlay:
    """Everything drawn on top of the current frame."""

    selected_index: Optional[int]
    counts: Dict[str, int]
    collecting: bool
    hand_count: int
    left_landmarks: int
    right_landmarks: int
    feature_count: int
    total_saved: int
    event: str = ""
    event_color: Tuple[int, int, int] = COLOR_WHITE
    tip: str = ""


def draw_overlay(frame, state: Overlay) -> None:
    """Draw class, progress, status, tips and the key legend on the frame."""
    height = frame.shape[0]

    # Selected class
    if state.selected_index is None:
        class_text = "Class: none selected  (press n / p or 1-9)"
        class_color = COLOR_YELLOW
    else:
        number = state.selected_index + 1
        name = SIGN_CLASSES[state.selected_index]
        class_text = (f"Class [{number}/{len(SIGN_CLASSES)}]: {name}")
        class_color = COLOR_CYAN

    # Progress for the selected class
    if state.selected_index is None:
        progress_text = f"Samples: - / {TARGET_SAMPLES}"
        progress_color = COLOR_WHITE
    else:
        name = SIGN_CLASSES[state.selected_index]
        saved = state.counts.get(name, 0)
        progress_text = f"Samples: {saved} / {TARGET_SAMPLES}"
        progress_color = (COLOR_GREEN if saved >= TARGET_SAMPLES
                          else COLOR_WHITE)

    status_text = ("Status: COLLECTING (one sample every "
                   f"{SAMPLE_INTERVAL_MS} ms)" if state.collecting
                   else "Status: PAUSED  (press c to start)")
    status_color = COLOR_GREEN if state.collecting else COLOR_YELLOW

    put_text(frame, class_text, (10, 30), class_color, 0.8, 2)
    put_text(frame, progress_text, (10, 62), progress_color, 0.7, 2)
    put_text(frame, status_text, (10, 94), status_color, 0.65, 2)
    put_text(frame,
             f"Hands: {state.hand_count}  (left {state.left_landmarks}, "
             f"right {state.right_landmarks})   "
             f"Features: {state.feature_count}",
             (10, 124), COLOR_WHITE, 0.6, 2)
    put_text(frame, f"Dataset total: {state.total_saved} samples",
             (10, 152), COLOR_DIM, 0.55, 1)
    if state.event:
        put_text(frame, state.event, (10, 182), state.event_color, 0.62, 2)

    # Rotating tip
    if state.tip:
        put_text(frame, f"Tip: {state.tip}", (10, height - 118),
                 COLOR_YELLOW, 0.55, 1)

    # Available classes (4 per row), selected one highlighted
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    row_height = 19
    per_row = 4
    for row in range(3):
        x = 10
        y = height - 96 + row * row_height
        for column in range(per_row):
            index = row * per_row + column
            if index >= len(SIGN_CLASSES):
                break
            text = f"{index + 1}.{SIGN_CLASSES[index]}"
            selected = (state.selected_index == index)
            color = COLOR_YELLOW if selected else COLOR_DIM
            put_text(frame, (">" if selected else " ") + text, (x, y),
                     color, scale, thickness if not selected else 2)
            text_width = cv2.getTextSize(" " + text, font, scale,
                                         thickness)[0][0]
            x += text_width + 14

    # Key legend
    put_text(frame,
             "n/p = change class   c = start/pause   q = quit   "
             "(mirrored preview)",
             (10, height - 14), COLOR_WHITE, 0.55, 1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect normalized 126-value landmark samples for "
                    "SignBridge AI (Phase 3). No model is trained here.")
    parser.add_argument("--class", dest="class_name", choices=SIGN_CLASSES,
                        default=None, help="pre-select a sign class")
    parser.add_argument("--auto-collect", action="store_true",
                        help="start collecting immediately (requires --class)")
    parser.add_argument("--max-new", type=int, default=0, metavar="N",
                        help="quit after N new samples are saved "
                             "(0 = no limit)")
    parser.add_argument("--seconds", type=int, default=0, metavar="S",
                        help="quit automatically after S seconds "
                             "(0 = no limit)")
    args = parser.parse_args()
    if args.auto_collect and not args.class_name:
        parser.error("--auto-collect requires --class NAME")
    if args.max_new < 0 or args.seconds < 0:
        parser.error("--max-new and --seconds must not be negative")
    return args


def main():
    args = parse_args()

    # 1. Verify the webcam --------------------------------------------------
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: could not open the webcam.")
        print("Make sure a camera is connected and not used by another "
              "application.")
        return 1

    # 2. Verify the MediaPipe model ----------------------------------------
    if not MODEL_PATH.exists():
        print(f"Error: hand landmark model not found: {MODEL_PATH}")
        print("Download it once with this PowerShell command:")
        print(f'  Invoke-WebRequest -Uri "{MODEL_URL}" '
              f'-OutFile "models\\hand_landmarker.task"')
        camera.release()
        return 1

    try:
        detector = HandDetector(model_path=MODEL_PATH, max_hands=2,
                                detection_confidence=0.5,
                                tracking_confidence=0.5)
    except Exception as error:
        print("Error: could not initialize the MediaPipe hand detector.")
        print(error)
        camera.release()
        return 1

    # 3. Load existing samples (resume) and validate the CSV header ---------
    try:
        progress = load_progress(DATASET_CSV)
        writer = DatasetWriter(DATASET_CSV)
    except DatasetError as error:
        print(f"Error: {error}")
        camera.release()
        detector.close()
        return 1

    print_startup_info(args, progress)

    counts = progress.counts          # label -> saved rows (resumed)
    seen = progress.seen              # exact rows already in the CSV

    selected_index = None
    if args.class_name:
        selected_index = SIGN_CLASSES.index(args.class_name)
    collecting = bool(args.auto_collect)

    event_text = ("Collection started (--auto-collect)" if collecting
                  else "Select a class, then press c to collect")
    event_color = COLOR_GREEN if collecting else COLOR_WHITE

    session_saved = 0
    session_rejected = Counter()
    last_reject_log: Dict[str, float] = {}
    last_attempt = 0.0                # first attempt happens immediately
    start_time = time.monotonic()
    next_tip_time = start_time + TIP_SECONDS
    tip_index = 0
    total_frames = 0

    if collecting and selected_index is None:  # guarded by argparse, defensive
        collecting = False

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Error: failed to read a frame from the webcam.")
                break

            # MIRROR THE PREVIEW - exactly once, before detection (Phase 2.1
            # pipeline). Preview, landmarks and features share one coordinate
            # system; no other flip happens anywhere in this project.
            frame = cv2.flip(frame, 1)

            results = detector.detect(frame)
            hands = parse_hands(results)
            detector.draw_landmarks(frame, hands)

            now = time.monotonic()
            total_frames += 1

            # Optional time limit (--seconds)
            if args.seconds and now - start_time >= args.seconds:
                event_text = "Time limit reached - stopping"
                event_color = COLOR_CYAN
                print(f"\nTime limit of {args.seconds} s reached.")
                break

            label = (SIGN_CLASSES[selected_index]
                     if selected_index is not None else None)
            left_hand, right_hand = split_hands(hands)

            # --- sampling (never every frame: cooldown + validation) ---
            if (collecting and label
                    and (now - last_attempt) * 1000.0 >= SAMPLE_INTERVAL_MS):
                last_attempt = now
                accepted, reason, sample = assess_sample(hands, label,
                                                         counts, seen)
                if not accepted:
                    session_rejected[reason] += 1
                    event_text = f"Rejected: {reason}"
                    event_color = COLOR_YELLOW
                    if now - last_reject_log.get(reason, 0.0) \
                            >= REJECT_LOG_INTERVAL_S:
                        print(f"Rejected: {reason}")
                        last_reject_log[reason] = now
                else:
                    try:
                        writer.write(sample, label)
                    except (DatasetError, OSError) as error:
                        collecting = False
                        event_text = f"Write failed: {error}"
                        event_color = COLOR_RED
                        print(f"Error: could not save sample: {error}")
                        print("Collection paused - fix the problem and "
                              "restart the collector.")
                    else:
                        counts[label] += 1
                        seen.add((label, tuple(sample)))
                        session_saved += 1
                        saved = counts[label]
                        event_text = (f"Saved {label} {saved}/"
                                      f"{TARGET_SAMPLES}")
                        event_color = COLOR_GREEN
                        print(f"Saved {label} sample {saved}/{TARGET_SAMPLES} "
                              f"(hands: left {len(left_hand.landmarks) if left_hand else 0}, "
                              f"right {len(right_hand.landmarks) if right_hand else 0})")

                        if saved >= TARGET_SAMPLES:
                            collecting = False
                            event_text = (f"{label} target reached "
                                          f"({TARGET_SAMPLES}/"
                                          f"{TARGET_SAMPLES})")
                            event_color = COLOR_CYAN
                            print(f"{label} reached "
                                  f"{TARGET_SAMPLES} samples - "
                                  f"collection paused for this class.")

                        if args.max_new and session_saved >= args.max_new:
                            print(f"\nBatch complete: {session_saved} new "
                                  f"sample(s) saved.")
                            break

            # Rotating tip
            if now >= next_tip_time:
                tip_index = (tip_index + 1) % len(COLLECTION_TIPS)
                next_tip_time = now + TIP_SECONDS

            draw_overlay(frame, Overlay(
                selected_index=selected_index,
                counts=counts,
                collecting=collecting,
                hand_count=len(hands),
                left_landmarks=len(left_hand.landmarks) if left_hand else 0,
                right_landmarks=len(right_hand.landmarks) if right_hand else 0,
                feature_count=MAX_FEATURE_LENGTH,
                total_saved=sum(counts.values()),
                event=event_text,
                event_color=event_color,
                tip=COLLECTION_TIPS[tip_index],
            ))

            cv2.imshow(WINDOW_TITLE, frame)

            # --- keys (click the video window first) ---
            key = cv2.waitKey(1) & 0xFF
            if key == 255 or key == -1:
                continue
            char = chr(key).lower() if key < 256 else ""

            if char == "q":
                break
            if char == "c":
                if collecting:
                    collecting = False
                    event_text = "Paused"
                    event_color = COLOR_YELLOW
                    print("Collection paused.")
                elif selected_index is None:
                    event_text = "Select a class first (n / p or 1-9)"
                    event_color = COLOR_YELLOW
                elif counts.get(label, 0) >= TARGET_SAMPLES:
                    event_text = (f"{label} already complete "
                                  f"({TARGET_SAMPLES}/{TARGET_SAMPLES})")
                    event_color = COLOR_CYAN
                else:
                    collecting = True
                    last_attempt = 0.0   # allow an immediate first sample
                    event_text = f"Collecting {label}"
                    event_color = COLOR_GREEN
                    print(f"Collecting {label} "
                          f"({counts.get(label, 0)}/{TARGET_SAMPLES} "
                          f"saved so far)")
                continue
            if char in ("n", "p"):
                if selected_index is None:
                    new_index = 0 if char == "n" else len(SIGN_CLASSES) - 1
                elif char == "n":
                    new_index = (selected_index + 1) % len(SIGN_CLASSES)
                else:
                    new_index = (selected_index - 1) % len(SIGN_CLASSES)
            elif char in QUICK_SELECT_KEYS:
                new_index = QUICK_SELECT_KEYS[char]
            else:
                continue

            if new_index != selected_index:
                selected_index = new_index
                if collecting:
                    # Never switch labels silently: pause and ask again.
                    collecting = False
                    event_text = (f"Class changed to "
                                  f"{SIGN_CLASSES[selected_index]} - "
                                  f"press c to collect it")
                    event_color = COLOR_YELLOW
                    print(f"Class changed to {SIGN_CLASSES[selected_index]} "
                          f"- collection paused.")
                else:
                    saved = counts.get(SIGN_CLASSES[selected_index], 0)
                    event_text = (f"Selected {SIGN_CLASSES[selected_index]} "
                                  f"({saved}/{TARGET_SAMPLES} saved)")
                    event_color = COLOR_CYAN

    except KeyboardInterrupt:
        print("\nInterrupted - shutting down safely.")

    finally:
        # Always release resources and report - samples already on disk stay.
        writer.close()
        camera.release()
        cv2.destroyAllWindows()
        detector.close()

        print("\nCamera released.")
        print(f"Frames processed: {total_frames}")
        print(f"New samples saved this run: {session_saved}")
        if session_rejected:
            print("Rejected this run:")
            for reason, amount in session_rejected.most_common():
                print(f"    {amount:>4}  {reason}")
        elapsed = time.monotonic() - start_time
        if elapsed > 0:
            print(f"Session length: {elapsed:.1f} s")
        print_dataset_summary(counts)

    return 0


if __name__ == "__main__":
    sys.exit(main())

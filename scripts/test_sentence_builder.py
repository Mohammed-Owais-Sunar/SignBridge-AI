"""Phase 6 test: sentence builder behaviors (no camera, no model, no audio).

Verifies - deterministic (fake clock), without OpenCV/MediaPipe - that:

  A. one gesture -> exactly one word; several gestures -> single-spaced words
  B. SPACE inserts exactly one space; holding/re-pressing SPACE never doubles
     it and never creates a leading space
  C. DELETE removes the most recent WORD; holding DELETE deletes only one
     word per intentional gesture (never the whole sentence at once)
  D. holding a gesture never repeats it (edge trigger), and the same gesture
     CAN be repeated after a SUSTAINED release (hand absent >= 8 frames or
     uncertain >= 20 frames) or after a different gesture (immediate re-arm)
  E. clear() empties the sentence and does not re-fire a held gesture
  F. deleting an empty sentence is a safe no-op
  G. unknown / "Uncertain" / no-hand labels fire nothing and re-arm only
     once sustained (brief threshold flicker must NOT duplicate a held
     gesture - the flaw found in live Phase 6 testing)
  H. the action cooldown (ACTION_COOLDOWN_SECONDS) holds back early actions
     but never drops one that stays visible
  I. scripts/predict_realtime.py is wired to the builder (observe + 'c' key
     + sentence overlay) and contains NO TTS/web-UI code (Phase 7/8 scope)

Usage (from the project folder):
    python scripts/test_sentence_builder.py
"""

import sys
from pathlib import Path

# Make the src/ package importable when this script is run from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (ACTION_COOLDOWN_SECONDS, REARM_NO_HAND_FRAMES,
                        REARM_UNCERTAIN_FRAMES, SIGN_CLASSES)
from src.sentence_builder import (ACTION_LABELS, DELETE_LABEL, GESTURE_TEXT,
                                  NO_HAND_LABEL, SPACE_LABEL, SentenceBuilder)

# ---------------------------------------------------------------------------
RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail and not condition else ""
    print(f"  [{status}] {name}{suffix}")


class FakeClock:
    """Deterministic clock - the tests control time explicitly."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def make_builder():
    clock = FakeClock()
    builder = SentenceBuilder(cooldown_seconds=ACTION_COOLDOWN_SECONDS,
                              no_hand_frames=REARM_NO_HAND_FRAMES,
                              uncertain_frames=REARM_UNCERTAIN_FRAMES,
                              clock=clock)
    return builder, clock


def step(builder, clock, label, frames=1, dt=0.04):
    """Simulate `frames` video frames (~25 fps) showing `label`.

    Returns the list of action strings fired during those frames.
    """
    fired = []
    for _ in range(frames):
        action = builder.observe(label)
        if action:
            fired.append(action)
        clock.advance(dt)
    return fired


def hold_and_release(builder, clock, label, hold=10, release=25):
    """One intentional gesture: hold it, then pass through a real release.

    The default release of 25 frames (~1 s) exceeds REARM_UNCERTAIN_FRAMES
    (20), so the same gesture could fire again afterwards; a gesture CHANGE
    re-arms immediately anyway.
    """
    fired = step(builder, clock, label, hold)
    step(builder, clock, None, release)
    return fired


def main():
    print("=" * 46)
    print("SIGNBRIDGE AI - SENTENCE BUILDER TEST (Phase 6)")
    print("=" * 46)

    # --- A: word gestures ---------------------------------------------------
    builder, clock = make_builder()
    fired = step(builder, clock, "HELLO", frames=5)
    check("1. one gesture held 5 frames -> one action, sentence 'HELLO'",
          fired == ["WORD HELLO"] and builder.sentence == "HELLO",
          f"{fired} / {builder.sentence!r}")

    builder, clock = make_builder()
    for label in ["HELLO", SPACE_LABEL, "GOOD", "PLEASE"]:
        hold_and_release(builder, clock, label, hold=10, release=3)
    check("2. several gestures -> single-spaced sentence",
          builder.sentence == "HELLO GOOD PLEASE", repr(builder.sentence))

    # --- B: SPACE -----------------------------------------------------------
    builder, clock = make_builder()
    hold_and_release(builder, clock, "HELLO", 5, 3)
    hold_and_release(builder, clock, SPACE_LABEL, 5, 3)
    hold_and_release(builder, clock, "GOOD", 10, 3)
    check("3. HELLO + SPACE + GOOD -> 'HELLO GOOD' (exactly one space)",
          builder.sentence == "HELLO GOOD" and "  " not in builder.sentence,
          repr(builder.sentence))

    builder, clock = make_builder()
    hold_and_release(builder, clock, "HELLO", 5, 3)
    fired = step(builder, clock, SPACE_LABEL, frames=100)   # held ~4 s
    check("4a. SPACE held 100 frames fires exactly once",
          fired == ["SPACE"] and builder.sentence == "HELLO ",
          f"{fired} / {builder.sentence!r}")
    step(builder, clock, None, 25)      # sustained release (>= 20 frames)
    fired = step(builder, clock, SPACE_LABEL, frames=10)
    check("4b. SPACE pressed again -> still ONE space (collapse rule)",
          fired == ["SPACE"] and builder.sentence == "HELLO "
          and "  " not in builder.sentence, repr(builder.sentence))
    step(builder, clock, None, 3)
    step(builder, clock, "GOOD", 10)
    check("4c. word after SPACE lands with a single separator",
          builder.sentence == "HELLO GOOD", repr(builder.sentence))

    builder, clock = make_builder()
    fired = step(builder, clock, SPACE_LABEL, frames=30)
    check("4d. SPACE before any word -> no leading space",
          builder.sentence == "" and fired == ["SPACE"],
          f"{fired} / {builder.sentence!r}")

    # --- C: DELETE ----------------------------------------------------------
    builder, clock = make_builder()
    for word in ["HELLO", "GOOD", "THANK_YOU"]:
        hold_and_release(builder, clock, word, 10, 3)
    check("5. three words build 'HELLO GOOD THANK_YOU'",
          builder.sentence == "HELLO GOOD THANK_YOU", repr(builder.sentence))

    builder, clock = make_builder()
    for word in ["HELLO", "GOOD", "THANK_YOU"]:
        hold_and_release(builder, clock, word, 10, 3)
    hold_and_release(builder, clock, DELETE_LABEL, 10)
    check("5a. DELETE removes the most recent WORD (word-level)",
          builder.sentence == "HELLO GOOD", repr(builder.sentence))
    hold_and_release(builder, clock, DELETE_LABEL, 10)
    check("5b. second intentional DELETE removes the next word",
          builder.sentence == "HELLO", repr(builder.sentence))

    builder, clock = make_builder()
    for word in ["HELLO", "GOOD", "THANK_YOU"]:
        hold_and_release(builder, clock, word, 10, 3)
    fired = step(builder, clock, DELETE_LABEL, frames=100)   # held ~4 s
    check("6a. DELETE held 100 frames deletes exactly ONE word",
          fired == ["DELETE"] and builder.sentence == "HELLO GOOD",
          f"{fired} / {builder.sentence!r}")
    step(builder, clock, None, 25)      # sustained release (>= 20 frames)
    step(builder, clock, DELETE_LABEL, frames=10)
    check("6b. released then DELETE again -> exactly one more word",
          builder.sentence == "HELLO", repr(builder.sentence))

    # --- D: re-arm / intentional repeats ------------------------------------
    builder, clock = make_builder()
    step(builder, clock, "HELLO", 8)
    step(builder, clock, None, 25)      # sustained release (>= 20 frames)
    step(builder, clock, "HELLO", 10)
    check("7a. HELLO -> sustained release -> HELLO gives 'HELLO HELLO'",
          builder.sentence == "HELLO HELLO", repr(builder.sentence))
    step(builder, clock, None, 10)
    hold_and_release(builder, clock, "GOOD", 10, 3)
    step(builder, clock, "HELLO", 10)
    check("7b. change re-arms: a different gesture also allows repeats",
          builder.sentence == "HELLO HELLO GOOD HELLO", repr(builder.sentence))

    # --- E: clear -----------------------------------------------------------
    builder, clock = make_builder()
    step(builder, clock, "HELLO", 5)
    builder.clear()
    check("8a. clear() empties the sentence", builder.sentence == "")
    fired = step(builder, clock, "HELLO", frames=10)   # still held, no release
    check("8b. clear does NOT re-arm a still-held gesture",
          fired == [] and builder.sentence == "",
          f"{fired} / {builder.sentence!r}")
    step(builder, clock, NO_HAND_LABEL, 10)   # hand away (>= 8 frames)
    step(builder, clock, "HELLO", 10)
    check("8c. after the hand is released the gesture can be used again",
          builder.sentence == "HELLO", repr(builder.sentence))

    # --- F: empty delete ----------------------------------------------------
    builder, clock = make_builder()
    fired = step(builder, clock, DELETE_LABEL, frames=10)
    check("9. DELETE on an empty sentence is a safe no-op",
          fired == ["DELETE"] and builder.sentence == "",
          f"{fired} / {builder.sentence!r}")

    # --- G: unknown / uncertain / release -----------------------------------
    builder, clock = make_builder()
    step(builder, clock, "HELLO", 5)
    all_quiet = True
    details = []
    for weird in ["Uncertain", "No hand detected", "Sample rejected",
                  "NOT_A_CLASS", None]:
        fired = step(builder, clock, weird, frames=5)
        if fired:
            all_quiet = False
            details.append(f"{weird}: {fired}")
    check("10a. uncertain/no-hand/unknown labels fire nothing",
          all_quiet and builder.sentence == "HELLO",
          "; ".join(details) or repr(builder.sentence))
    step(builder, clock, "HELLO", 10)
    check("10b. those states re-arm the same gesture for intentional reuse",
          builder.sentence == "HELLO HELLO", repr(builder.sentence))

    # --- H: confidence below threshold --------------------------------------
    source = (PROJECT_ROOT / "scripts" / "predict_realtime.py").read_text(
        encoding="utf-8")
    check("11a. predict_realtime feeds the stable display label to observe()",
          "builder.observe(prediction)" in source)
    check("11b. below-threshold frames show 'Uncertain' (not an action label)",
          'prediction = "Uncertain"' in source
          and "Uncertain" not in ACTION_LABELS)

    # --- I: long mixed sequence ---------------------------------------------
    builder, clock = make_builder()
    sequence = ["HELLO", SPACE_LABEL, SPACE_LABEL, "GOOD", DELETE_LABEL,
                SPACE_LABEL, "THANK_YOU"]
    for label in sequence:
        hold_and_release(builder, clock, label, 10, 3)
    check("12. sequence HELLO SPACE SPACE GOOD DELETE SPACE THANK_YOU "
          "-> 'HELLO THANK_YOU'",
          builder.sentence == "HELLO THANK_YOU", repr(builder.sentence))

    # --- J: cooldown --------------------------------------------------------
    builder, clock = make_builder()
    first = step(builder, clock, "HELLO", frames=1)
    early = step(builder, clock, "GOOD", frames=4)      # all within cooldown
    check("13a. a new action inside the cooldown is held back at first",
          first == ["WORD HELLO"] and early == []
          and builder.sentence == "HELLO",
          f"{first} + {early} / {builder.sentence!r}")
    later = step(builder, clock, "GOOD", frames=10)     # retried until allowed
    check("13b. the held-back action fires later (never silently dropped)",
          later == ["WORD GOOD"] and builder.sentence == "HELLO GOOD",
          f"{later} / {builder.sentence!r}")
    check("13c. cooldown comes from config and is applied",
          ACTION_COOLDOWN_SECONDS == 0.3
          and builder.cooldown_seconds == 0.3,
          str(builder.cooldown_seconds))

    # --- sustained re-arm: THE live-test regression --------------------------
    # (Before this rule, confidence flapping around the threshold re-armed a
    #  held gesture every few frames: one held HELP produced 5 words live.)
    builder, clock = make_builder()
    step(builder, clock, "HELLO", frames=5)            # first fire
    for _ in range(8):                                 # flicker, ~0.12 s each
        step(builder, clock, "Uncertain", frames=3)
        step(builder, clock, "HELLO", frames=4)
    check("16a. brief confidence flicker never duplicates a held gesture",
          builder.action_count == 1 and builder.sentence == "HELLO",
          f"actions={builder.action_count} / {builder.sentence!r}")
    step(builder, clock, "Uncertain", frames=REARM_UNCERTAIN_FRAMES + 5)
    step(builder, clock, "HELLO", frames=10)
    check("16b. a SUSTAINED uncertain period re-arms the same gesture",
          builder.sentence == "HELLO HELLO", repr(builder.sentence))

    builder, clock = make_builder()
    step(builder, clock, "HELLO", frames=5)
    flicker = step(builder, clock, NO_HAND_LABEL, frames=3)   # 1-3 f drop
    step(builder, clock, "HELLO", frames=10)
    check("16c. a 3-frame hand-detection drop does not re-fire",
          flicker == [] and builder.action_count == 1
          and builder.sentence == "HELLO",
          f"{flicker} / {builder.sentence!r}")
    step(builder, clock, NO_HAND_LABEL, frames=REARM_NO_HAND_FRAMES + 5)
    step(builder, clock, "HELLO", frames=10)
    check("16d. hand absent >= REARM_NO_HAND_FRAMES re-arms quickly",
          builder.sentence == "HELLO HELLO", repr(builder.sentence))

    check("16e. re-arm thresholds come from config and are applied",
          REARM_NO_HAND_FRAMES == 8 and REARM_UNCERTAIN_FRAMES == 20
          and builder.no_hand_frames == REARM_NO_HAND_FRAMES
          and builder.uncertain_frames == REARM_UNCERTAIN_FRAMES,
          f"{builder.no_hand_frames}/{builder.uncertain_frames}")
    check("16f. predict_realtime's no-hand display text matches NO_HAND_LABEL",
          NO_HAND_LABEL == "No hand detected"
          and f'"{NO_HAND_LABEL}"' in source,
          NO_HAND_LABEL)

    # --- K: mapping ---------------------------------------------------------
    word_gestures = set(SIGN_CLASSES) - {SPACE_LABEL, DELETE_LABEL}
    check("14a. GESTURE_TEXT covers exactly the 10 word gestures",
          set(GESTURE_TEXT) == word_gestures and len(GESTURE_TEXT) == 10,
          str(sorted(GESTURE_TEXT)))
    check("14b. SPACE/DELETE are actions, not word mappings",
          SPACE_LABEL in ACTION_LABELS and DELETE_LABEL in ACTION_LABELS
          and SPACE_LABEL not in GESTURE_TEXT
          and DELETE_LABEL not in GESTURE_TEXT)
    check("14c. mapping is identity - ML labels are never altered",
          all(key == value for key, value in GESTURE_TEXT.items()))

    # --- L: wiring + scope discipline ---------------------------------------
    check("15a. predict_realtime imports and uses SentenceBuilder",
          "from src.sentence_builder import SentenceBuilder" in source
          and "builder = SentenceBuilder()" in source)
    check("15b. 'c' clears the sentence, 'q' still quits",
          'char == "c"' in source and "builder.clear()" in source
          and 'char == "q"' in source)
    check("15c. the overlay displays the sentence",
          "Sentence:" in source and '"sentence": builder.sentence' in source)
    check("15d. no TTS/web-UI code anywhere in Phase 5/6 scripts",
          all(word not in source for word in
              ("pyttsx3", "import streamlit", "from streamlit", "gTTS",
               "playsound")))
    builder_source = (PROJECT_ROOT / "src" / "sentence_builder.py").read_text(
        encoding="utf-8")
    check("15e. sentence_builder is independent of OpenCV/MediaPipe",
          "cv2" not in builder_source
          and "mediapipe" not in builder_source
          and "joblib" not in builder_source)

    # --- summary ------------------------------------------------------------
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print("\n" + "=" * 46)
    print(f"Checks passed: {passed} / {total}")
    print("=" * 46)
    if passed != total:
        print("FAILED checks:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"    - {name}: {detail}")
        return 1
    print("All sentence builder checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

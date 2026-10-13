"""Phase 6: sentence builder - text composition from stable gesture labels.

This module is deliberately independent from OpenCV, MediaPipe and the ML
model. It receives the ALREADY STABLE displayed prediction from the Phase 5
pipeline (one call per video frame) and turns it into a sentence. The Random
Forest classifier is never touched by this module.

BEHAVIORS (documented contract):

1. GESTURE-TO-TEXT
   The 10 "word" gestures map to readable text via GESTURE_TEXT (identity
   mapping: HELLO -> "HELLO", I_LOVE_YOU -> "I_LOVE_YOU", ...). The ML labels
   themselves are never changed - the mapping only affects the sentence.

2. SPACE
   Inserts EXACTLY ONE space: a space is only added when the sentence is
   non-empty and does not already end with a space. Therefore SPACE can never
   produce "HELLO  " style doubles, a leading space, or an all-space line.
   SPACE before the first word is a no-op (no leading space).

3. EDGE TRIGGER + RE-ARM  (the frame-rate debounce)
   A gesture fires only when it BECOMES the active displayed label. While the
   same label stays visible, nothing repeats: "HELLO" held for 300 frames
   produces exactly one word. Re-arming (so the user CAN intentionally
   repeat) happens when:
     - the displayed gesture CHANGES to a different action -> immediately
       (the new gesture is its own action);
     - the hand is absent (NO_HAND_LABEL) for >= REARM_NO_HAND_FRAMES
       consecutive frames (~0.3 s) -> a quick "lower the hand and sign
       again" repeat works;
     - the display stays below-threshold / unknown for >=
       REARM_UNCERTAIN_FRAMES consecutive frames (~0.75 s) -> BRIEF
       confidence flicker around MIN_PREDICTION_CONFIDENCE does NOT
       re-fire a held gesture (live-tested rule: without it, one held
       gesture produced 5 duplicate words while confidence wobbled
       across the threshold).
   Examples:
       HELLO -> hand away 0.5 s -> HELLO     gives "HELLO HELLO"
       HELLO -> GOOD -> HELLO                gives "HELLO GOOD HELLO"
   A gesture is never permanently locked after its first activation.

4. ACTION COOLDOWN (config: ACTION_COOLDOWN_SECONDS, default 0.3 s)
   Two actions are at least that many seconds apart. A gesture that arrives
   too soon after the previous action is NOT consumed: it stays pending and
   fires on a later frame while it remains visible (bounded to one action per
   cooldown interval, never spamming, never silently lost).

5. DELETE - WORD LEVEL
   Removes the MOST RECENTLY ADDED WORD (trailing spaces first):
       "HELLO GOOD MORNING" -> DELETE -> "HELLO GOOD"
   Deleting an empty sentence is a no-op (no crash).
   Because DELETE is edge-triggered, holding it deletes ONE word per
   intentional gesture, never the whole sentence at once.

6. clear()
   Empties the sentence. It does NOT re-arm a held gesture: the same gesture
   still visible after a clear will not re-fire until it is released or
   changed. Clearing is a keyboard action ('c'), independent of quit ('q').

NO SPEECH: this module never produces audio (text-to-speech is Phase 7).
"""

import time

from src.config import (ACTION_COOLDOWN_SECONDS, REARM_NO_HAND_FRAMES,
                        REARM_UNCERTAIN_FRAMES)

# The 10 "word" gestures -> readable sentence text.
# Identity mapping: the ML labels are not changed anywhere in the project.
GESTURE_TEXT = {
    "HELLO": "HELLO",
    "YES": "YES",
    "NO": "NO",
    "THANK_YOU": "THANK_YOU",
    "PLEASE": "PLEASE",
    "HELP": "HELP",
    "SORRY": "SORRY",
    "I_LOVE_YOU": "I_LOVE_YOU",
    "GOOD": "GOOD",
    "BAD": "BAD",
}

SPACE_LABEL = "SPACE"
DELETE_LABEL = "DELETE"

# Display text for "no hand in view" (Phase 5 contract). MUST stay in sync
# with the prediction display text in scripts/predict_realtime.py - the
# test suite pins both to this value. Everything else that is not an action
# label ("Uncertain", "Sample rejected", None, ...) counts as an
# uncertain/unknown release state.
NO_HAND_LABEL = "No hand detected"

# Everything the builder reacts to (anything else = release/no action)
ACTION_LABELS = frozenset(list(GESTURE_TEXT) + [SPACE_LABEL, DELETE_LABEL])


class SentenceBuilder:
    """Turns stable per-frame gesture labels into a sentence.

    Feed it once per video frame with the displayed prediction
    (predict_realtime.py passes exactly that), or with None/unknown labels
    for "no usable prediction". It returns a short action description when
    an action fired, else None.
    """

    def __init__(self, cooldown_seconds=ACTION_COOLDOWN_SECONDS,
                 no_hand_frames=REARM_NO_HAND_FRAMES,
                 uncertain_frames=REARM_UNCERTAIN_FRAMES, clock=None):
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        # Consecutive release frames required before the SAME gesture may
        # fire again (config.py: REARM_NO_HAND_FRAMES / REARM_UNCERTAIN_FRAMES)
        self.no_hand_frames = max(0, int(no_hand_frames))
        self.uncertain_frames = max(0, int(uncertain_frames))
        # Injectable clock: tests drive it deterministically; the app uses
        # time.monotonic() (wall time, unaffected by clock changes).
        self._clock = clock if clock is not None else time.monotonic
        self._sentence = ""
        self._active = None          # label that already fired (edge trigger)
        self._absent = 0             # consecutive release frames since seen
        self._last_action_at = float("-inf")
        self.action_count = 0        # total fired actions (stats/tests)

    # ------------------------------------------------------------------ state
    @property
    def sentence(self) -> str:
        """Current sentence (always a single-space separated string)."""
        return self._sentence

    @property
    def active_label(self):
        """Label that most recently fired (None when released). Debug aid."""
        return self._active

    # ---------------------------------------------------------------- observe
    def observe(self, label):
        """Call ONCE PER FRAME with the stable displayed prediction.

        Returns a short action string ("WORD HELLO", "SPACE", "DELETE") when
        this frame triggered an action, otherwise None.

        Release states fire nothing. Re-arming is SUSTAINED, not instant:
        the no-hand display must persist no_hand_frames consecutive frames,
        any other non-action label (uncertain/unknown/None) must persist
        uncertain_frames consecutive frames, while a DIFFERENT visible
        gesture re-arms immediately (see module docstring). This is what
        keeps confidence flicker around the threshold from re-firing a
        gesture the user is simply holding.
        """
        if label is None or label not in ACTION_LABELS:
            # Release-ish frame. Only a SUSTAINED absence re-arms (module
            # docstring point 3); brief flickers keep the active gesture.
            required = (self.no_hand_frames if label == NO_HAND_LABEL
                        else self.uncertain_frames)
            if self._active is not None:
                self._absent += 1
                if self._absent >= required:
                    self._active = None   # re-armed
            return None

        if label == self._active:
            self._absent = 0              # still visible -> still held
            return None                   # edge-triggered, no repeat

        # A different known action: the CHANGE re-arms instantly. Cooldown:
        # do NOT commit _active yet, so an action that is too early retries
        # on later frames instead of being swallowed (and still can never
        # exceed one action per cooldown interval).
        now = self._clock()
        if now - self._last_action_at < self.cooldown_seconds:
            return None

        self._active = label
        self._absent = 0
        self._last_action_at = now
        self.action_count += 1
        return self._fire(label)

    # ------------------------------------------------------------------- fire
    def _fire(self, label):
        if label in GESTURE_TEXT:
            return self.add_word(label)
        if label == SPACE_LABEL:
            return self._insert_space()
        return self._delete_word()   # DELETE_LABEL (checked by ACTION_LABELS)

    def add_word(self, label: str) -> str:
        """Append a word gesture; words are separated by exactly one space."""
        if label not in GESTURE_TEXT:
            raise ValueError(f"{label!r} is not a word gesture")
        if self._sentence and not self._sentence.endswith(" "):
            self._sentence += " "    # exactly one separator between words
        self._sentence += GESTURE_TEXT[label]
        return f"WORD {GESTURE_TEXT[label]}"

    def _insert_space(self) -> str:
        # Collapse rule: only add a space if the sentence is non-empty and
        # does not already end with one -> never doubles, never leading.
        if self._sentence and not self._sentence.endswith(" "):
            self._sentence += " "
        return "SPACE"

    def _delete_word(self) -> str:
        """Remove the most recent WORD (word-level delete), trailing spaces first."""
        text = self._sentence.rstrip()
        if not text:
            return "DELETE"          # empty sentence: documented no-op
        separator = text.rfind(" ")
        self._sentence = text[:separator] if separator != -1 else ""
        return "DELETE"

    # ------------------------------------------------- manual UI actions
    # These bypass the edge trigger and the cooldown ON PURPOSE: those two
    # rules (module docstring points 3-4) exist to stop a HELD GESTURE
    # from repeating, not to throttle deliberate input such as a web UI
    # button (Phase 8, app.py). The text rules themselves are identical.
    def insert_space(self) -> str:
        """Insert one separator space NOW (manual action, e.g. a button).

        Same collapse rule as the SPACE gesture: only when the sentence is
        non-empty and does not already end with a space.
        """
        return self._insert_space()

    def delete_word(self) -> str:
        """Remove the most recent word NOW (manual action, e.g. a button).

        Same word-level rule as the DELETE gesture; deleting an empty
        sentence is a no-op.
        """
        return self._delete_word()

    # ------------------------------------------------------------------ clear
    def clear(self) -> None:
        """Empty the sentence. Does not re-arm a currently held gesture."""
        self._sentence = ""

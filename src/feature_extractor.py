"""Deterministic, fixed-length landmark representation (Phase 2.1).

The extractor turns the detected hands into ONE flat list of floats whose
length never changes, which is what a future classifier (e.g. a Random Forest)
will require. Nothing is trained in this phase - this module only defines and
produces the input representation.

=============================================================================
EXACT REPRESENTATION (maximum 126 values)
=============================================================================

    index   0 -  62   LEFT  hand block
              landmark 0 (x, y, z)
              landmark 1 (x, y, z)
              ...
              landmark 20 (x, y, z)          -> 21 x 3 = 63 values

    index  63 - 125   RIGHT hand block
              landmark 0 (x, y, z)
              landmark 1 (x, y, z)
              ...
              landmark 20 (x, y, z)          -> 21 x 3 = 63 values

    TOTAL: 126 values (63 + 63)

Ordering rules (deterministic - never MediaPipe's detection order):

1. Blocks are ordered by HANDEDNESS: LEFT first (0-62), then RIGHT (63-125),
   regardless of which hand MediaPipe reports first in a given frame.
2. Inside a block, the 21 landmarks are always in MediaPipe's fixed topology
   order: 0 = wrist, 1-4 = thumb, 5-8 = index finger, 9-12 = middle finger,
   13-16 = ring finger, 17-20 = little finger.
3. Each landmark contributes its coordinates in the order x, y, z.
4. Coordinates are the normalized coordinates of the frame handed to
   MediaPipe - in this project that is the *mirrored* frame (the same frame
   that is displayed), see README.md "Phase 2.1". x and y are in [0, 1];
   z is the depth relative to the wrist.

MISSING HAND HANDLING:

- 0 hands detected -> both 63-value blocks are filled with 0.0 (126 zeros).
- 1 hand detected  -> the present hand fills its block, the absent hand's
  block is filled with 0.0.
- The missing-hand value is MISSING_HAND_VALUE = 0.0, so a future model always
  receives exactly 126 numbers and can tell "hand absent" from real data
  (real landmark coordinates are never all-zero for a visible hand).

HANDEDNESS AMBIGUITY:

- `split_hands()` places hands by MediaPipe's "Left"/"Right" label.
- A hand labelled "Unknown" (handedness missing or ambiguous) is placed in the
  left slot if it is still free, otherwise in the right slot, always in
  MediaPipe's detection order. This is a fixed, documented rule - the
  representation stays deterministic and nothing is guessed silently.

=============================================================================
PHASE 3 - NORMALIZATION (feature engineering)
=============================================================================

`extract_features()` returns the RAW vector (frame coordinates, Phase 2.1).
`normalize_features()` / `build_sample()` return the NORMALIZED vector - the
126 values that are stored in data/processed/dataset.csv and that a future
model will be trained on. Normalization happens per hand:

1. ORIGIN: subtract the wrist (landmark 0) of that hand from all of its
   landmarks  -> the hand is centered on itself, so absolute position in the
   frame no longer matters.
2. SCALE: divide by the hand size, the Euclidean distance between the wrist
   (landmark 0) and the middle-finger MCP (landmark 9) -> distance from the
   camera / apparent size no longer matters, only the SHAPE of the hand.

The LEFT and RIGHT hands are normalized INDEPENDENTLY (each with its own
wrist and its own scale - they are never normalized together). A missing
hand keeps its whole 63-value block as zeros, so the vector stays exactly
126 values for 0, 1 or 2 hands.

If a hand's scale is zero or invalid the sample is REJECTED (ValueError is
raised) instead of dividing by zero.

The SAME `build_sample()` function must be used at collection time and at
prediction time, so training and inference can never drift apart.

`feature_column_names()` names the 126 CSV columns in exactly the order of
the feature vector: left_0_x ... left_20_z, right_0_x ... right_20_z.
"""

import math

# 21 landmarks per hand, each with x, y and z
LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3  # x, y, z
HAND_FEATURE_LENGTH = LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK  # 63

# Up to 2 hands (left block first, then right block)
MAX_HANDS = 2
MAX_FEATURE_LENGTH = HAND_FEATURE_LENGTH * MAX_HANDS  # 126

# Fixed value used for a hand that is not detected
MISSING_HAND_VALUE = 0.0

# Block offsets inside the feature vector
LEFT_BLOCK_OFFSET = 0                    # indices 0-62
RIGHT_BLOCK_OFFSET = HAND_FEATURE_LENGTH  # indices 63-125


def split_hands(hands):
    """Split detected hands into the (left, right) feature slots.

    Deterministic rule:
    1. The first hand labelled "Left" takes the left slot.
    2. The first hand labelled "Right" takes the right slot.
    3. Any remaining hands ("Unknown" labels, or duplicate labels) fill the
       still-empty slot in MediaPipe's detection order, left slot first.

    Args:
        hands: Iterable of DetectedHand (may be empty or None).

    Returns:
        (left_hand, right_hand); either element is None when that slot is
        empty.
    """
    if not hands:
        return None, None

    left_hand = None
    right_hand = None
    unassigned = []

    for hand in hands:
        if hand.label == "Left" and left_hand is None:
            left_hand = hand
        elif hand.label == "Right" and right_hand is None:
            right_hand = hand
        else:
            unassigned.append(hand)

    for hand in unassigned:
        if left_hand is None:
            left_hand = hand
        elif right_hand is None:
            right_hand = hand

    return left_hand, right_hand


def _hand_block(hand):
    """Flatten one hand into its 63-value block (fixed topology order).

    Always returns exactly HAND_FEATURE_LENGTH values: the block is padded
    with MISSING_HAND_VALUE if a hand ever carried fewer landmarks, so the
    total vector length can never change.
    """
    values = []
    for landmark in hand.landmarks[:LANDMARKS_PER_HAND]:
        values.extend([landmark[0], landmark[1], landmark[2]])

    if len(values) < HAND_FEATURE_LENGTH:
        values.extend([MISSING_HAND_VALUE] * (HAND_FEATURE_LENGTH - len(values)))

    return values


def extract_features(hands):
    """Build the fixed 126-value feature vector for the detected hands.

    Args:
        hands: List of DetectedHand from `src.hand_detector.parse_hands()`.
            May contain 0, 1 or 2 hands.

    Returns:
        A list of exactly MAX_FEATURE_LENGTH (126) floats:
        [63 LEFT values] + [63 RIGHT values]. A missing hand contributes a
        block of MISSING_HAND_VALUE (0.0). Never None.
    """
    left_hand, right_hand = split_hands(hands)

    features = [MISSING_HAND_VALUE] * MAX_FEATURE_LENGTH

    if left_hand is not None:
        features[LEFT_BLOCK_OFFSET:LEFT_BLOCK_OFFSET + HAND_FEATURE_LENGTH] = \
            _hand_block(left_hand)

    if right_hand is not None:
        features[RIGHT_BLOCK_OFFSET:RIGHT_BLOCK_OFFSET + HAND_FEATURE_LENGTH] = \
            _hand_block(right_hand)

    return features


# =============================================================================
# Phase 3 - normalization, validation and feature-column naming
# =============================================================================

# Landmarks used as the origin and the scale reference of a hand
# (MediaPipe topology: 0 = wrist, 9 = middle-finger MCP knuckle).
WRIST_INDEX = 0
MIDDLE_FINGER_MCP_INDEX = 9

# A wrist -> middle-MCP distance at or below this is treated as a degenerate
# hand: such a sample is rejected instead of divided by (almost) zero.
MIN_HAND_SCALE = 1e-6


def is_missing_hand_block(block):
    """True when a 63-value block represents an ABSENT hand (all zeros)."""
    if len(block) != HAND_FEATURE_LENGTH:
        return False
    return all(value == MISSING_HAND_VALUE for value in block)


def validate_feature_vector(features, expected_length=MAX_FEATURE_LENGTH):
    """Check that `features` is exactly `expected_length` finite numbers.

    This is the final gate before a sample is written to disk or fed to a
    model: wrong length, NaN and infinity are all rejected.

    Raises:
        ValueError: with a human-readable reason when the vector is invalid.
    """
    if features is None:
        raise ValueError("feature vector is missing (None)")
    if len(features) != expected_length:
        raise ValueError(f"feature vector has {len(features)} values, "
                         f"expected {expected_length}")
    for index, value in enumerate(features):
        if not math.isfinite(value):
            raise ValueError(f"feature value at index {index} is not finite "
                             f"({value!r})")


def normalize_hand_block(block):
    """Normalize ONE hand: its 63 raw values -> 63 scale-invariant values.

    Why this exists (see README "Phase 3"):
    Raw MediaPipe coordinates depend on where the hand sits in the frame, how
    far it is from the camera and how big it appears. A model trained on those
    numbers would learn "hand at the top-left, far away" instead of the shape
    of the gesture. Normalizing each hand on its own removes exactly those
    effects:

        1. subtract the wrist (landmark 0) of THIS hand from all its
           landmarks          -> position-invariant (camera position)
        2. divide by the distance wrist (0) -> middle-finger MCP (9)
                          of THIS hand       -> distance/zoom-invariant
                                                (hand size reference)

    The two hands are NEVER normalized together: each hand uses its own wrist
    and its own scale.

    Args:
        block: 63 raw values of one hand (21 landmarks x, y, z) in the fixed
            Phase 2.1 order. An all-zero block means "hand absent".

    Returns:
        Exactly 63 normalized floats. An absent hand's block is returned
        unchanged (63 zeros) so a missing hand stays recognizable.

    Raises:
        ValueError: wrong length, non-finite values, or a zero/invalid hand
            size. The caller must REJECT the sample - never divide by zero.
    """
    if len(block) != HAND_FEATURE_LENGTH:
        raise ValueError(f"hand block has {len(block)} values, "
                         f"expected {HAND_FEATURE_LENGTH}")

    for index, value in enumerate(block):
        if not math.isfinite(value):
            raise ValueError(f"hand block value at index {index} is not "
                             f"finite ({value!r})")

    # Absent hand: keep the documented all-zeros block as-is
    if is_missing_hand_block(block):
        return [MISSING_HAND_VALUE] * HAND_FEATURE_LENGTH

    wrist = block[WRIST_INDEX * COORDINATES_PER_LANDMARK:
                  WRIST_INDEX * COORDINATES_PER_LANDMARK + COORDINATES_PER_LANDMARK]
    middle_mcp = block[MIDDLE_FINGER_MCP_INDEX * COORDINATES_PER_LANDMARK:
                       MIDDLE_FINGER_MCP_INDEX * COORDINATES_PER_LANDMARK
                       + COORDINATES_PER_LANDMARK]

    scale = math.sqrt(sum((middle_mcp[i] - wrist[i]) ** 2
                          for i in range(COORDINATES_PER_LANDMARK)))
    if not math.isfinite(scale) or scale <= MIN_HAND_SCALE:
        raise ValueError("cannot normalize: hand size reference "
                         "(wrist landmark 0 -> middle-finger MCP landmark 9) "
                         "is zero or invalid")

    normalized = []
    for landmark in range(LANDMARKS_PER_HAND):
        base = landmark * COORDINATES_PER_LANDMARK
        for coordinate in range(COORDINATES_PER_LANDMARK):
            value = block[base + coordinate] - wrist[coordinate]
            normalized.append(value / scale)

    return normalized


def normalize_features(raw_features):
    """Normalize the fixed 126-value vector, hand block by hand block.

    Args:
        raw_features: the RAW 126 values from `extract_features()`
            ([63 LEFT] + [63 RIGHT]).

    Returns:
        A new list of exactly 126 floats: the same LEFT-then-RIGHT layout,
        each PRESENT hand centered/scaled on its own, each MISSING hand kept
        as 63 zeros.

    Raises:
        ValueError: wrong vector length or a hand that cannot be normalized
            (zero/invalid scale, NaN/inf). The sample must then be rejected.
    """
    if raw_features is None or len(raw_features) != MAX_FEATURE_LENGTH:
        got = "None" if raw_features is None else len(raw_features)
        raise ValueError(f"raw feature vector has {got} values, "
                         f"expected {MAX_FEATURE_LENGTH}")

    normalized = [MISSING_HAND_VALUE] * MAX_FEATURE_LENGTH

    for offset in (LEFT_BLOCK_OFFSET, RIGHT_BLOCK_OFFSET):
        block = list(raw_features[offset:offset + HAND_FEATURE_LENGTH])
        if is_missing_hand_block(block):
            # Absent hand -> its whole block stays zeros (documented rule)
            continue
        normalized[offset:offset + HAND_FEATURE_LENGTH] = \
            normalize_hand_block(block)

    return normalized


def build_sample(hands):
    """Build the 126 values that are stored in the dataset / used to predict.

    This is THE single feature-processing pipeline of the project:

        detected hands -> extract_features()   (raw, Phase 2.1)
                       -> normalize_features() (Phase 3 normalization)

    scripts/collect_data.py uses it to write data/processed/dataset.csv, and
    real-time prediction MUST use this same function later, so training and
    inference always see identical features.

    Args:
        hands: List of DetectedHand from `src.hand_detector.parse_hands()`
            (0, 1 or 2 hands).

    Returns:
        Exactly MAX_FEATURE_LENGTH (126) normalized floats.

    Raises:
        ValueError: when the sample must be rejected (e.g. a hand whose size
            reference is zero). The caller shows the reason and discards it.
    """
    return normalize_features(extract_features(hands))


def feature_column_names():
    """The 126 fixed CSV column names, in exact feature-vector order.

    Returns a NEW list of 126 names:

        left_0_x, left_0_y, left_0_z, ..., left_20_z,
        right_0_x, ..., right_20_z

    Position i of this list always corresponds to feature i of the vector,
    so the column order can never drift from the data order.
    """
    names = []
    # Block order is fixed by the feature vector: left block first, right second
    for prefix in ("left", "right"):
        for landmark in range(LANDMARKS_PER_HAND):
            for axis in ("x", "y", "z"):
                names.append(f"{prefix}_{landmark}_{axis}")
    return names

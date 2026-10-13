"""Phase 3 test: feature extraction, normalization and column naming.

Verifies - without training any model and without the webcam - that:

  A. 0 hands / 1 hand / 2 hands all produce exactly 126 raw values
  B. normalization also produces exactly 126 values, never NaN/inf
  C. a missing hand's 63-value block STAYS all zeros after normalization
  D. each present hand is centered on its own wrist and scaled to size 1
     (position/distance invariance, left and right normalized independently)
  E. invalid samples are REJECTED with a readable reason instead of being
     written (wrong length, NaN, degenerate hand size)
  F. the CSV column names match the feature order exactly
     (left_0_x ... left_20_z, right_0_x ... right_20_z, then 'label')

Usage (from the project folder):
    python scripts/test_feature_extraction.py
"""

import math
import sys
from pathlib import Path

# Make the src/ package importable when this script is run from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LABEL_COLUMN, SIGN_CLASSES
from src.dataset_io import expected_header
from src.feature_extractor import (
    HAND_FEATURE_LENGTH,
    LEFT_BLOCK_OFFSET,
    MAX_FEATURE_LENGTH,
    RIGHT_BLOCK_OFFSET,
    build_sample,
    extract_features,
    feature_column_names,
    normalize_features,
    normalize_hand_block,
)
from src.hand_detector import DetectedHand

# ---------------------------------------------------------------------------
# Synthetic hands (deterministic, no camera, no MediaPipe inference)
# ---------------------------------------------------------------------------
def synthetic_landmarks(origin=(0.5, 0.5, 0.0), size=1.0):
    """21 deterministic (x, y, z) landmarks forming a hand-like shape."""
    points = []
    for landmark in range(21):
        if landmark == 0:
            dx, dy, dz = 0.0, 0.0, 0.0
        else:
            dx = 0.02 * landmark * size
            dy = -0.03 * ((landmark % 5) + 1) * size
            dz = -0.001 * landmark * size
        points.append((origin[0] + dx, origin[1] + dy, origin[2] + dz))
    return tuple(points)


def make_hand(label, origin=(0.5, 0.5, 0.0), size=1.0, landmarks=None):
    return DetectedHand(label=label, handedness_score=0.99,
                        landmarks=landmarks if landmarks is not None
                        else synthetic_landmarks(origin, size))


def all_zero(values):
    return all(value == 0.0 for value in values)


def finite(values):
    return all(math.isfinite(value) for value in values)


# ---------------------------------------------------------------------------
# Tiny test harness
# ---------------------------------------------------------------------------
RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail and not condition else ""
    print(f"  [{status}] {name}{suffix}")


def expect_value_error(name, function, *args):
    try:
        function(*args)
    except ValueError as error:
        check(name, True, str(error))
    except Exception as error:  # noqa: BLE001 - must be ValueError specifically
        check(name, False, f"raised {type(error).__name__}: {error}")
    else:
        check(name, False, "no exception raised (sample was not rejected)")


def main():
    print("=" * 46)
    print("SIGNBRIDGE AI - FEATURE EXTRACTION TEST (Phase 3)")
    print("=" * 46)

    left = make_hand("Left", origin=(0.30, 0.35, -0.02), size=1.0)
    right = make_hand("Right", origin=(0.70, 0.45, 0.01), size=0.6)

    # --- A. raw extraction: 0 / 1 / 2 hands -------------------------------
    print("\nA) Raw feature vector (extract_features)")
    raw_none = extract_features([])
    check("0 hands -> exactly 126 values",
          len(raw_none) == MAX_FEATURE_LENGTH and
          len(raw_none) == 126, f"len={len(raw_none)}")
    check("0 hands -> all 126 values are zeros", all_zero(raw_none))

    raw_left = extract_features([left])
    check("1 hand -> exactly 126 values",
          len(raw_left) == MAX_FEATURE_LENGTH, f"len={len(raw_left)}")
    check("1 hand (Left) -> left block present",
          not all_zero(raw_left[0:HAND_FEATURE_LENGTH]))
    check("1 hand (Left) -> right block is 63 zeros",
          all_zero(raw_left[RIGHT_BLOCK_OFFSET:
                            RIGHT_BLOCK_OFFSET + HAND_FEATURE_LENGTH]))

    raw_right = extract_features([right])
    check("1 hand (Right) -> left block is 63 zeros",
          all_zero(raw_right[LEFT_BLOCK_OFFSET:
                             LEFT_BLOCK_OFFSET + HAND_FEATURE_LENGTH]))
    check("1 hand (Right) -> right block present",
          not all_zero(raw_right[RIGHT_BLOCK_OFFSET:
                                 RIGHT_BLOCK_OFFSET + HAND_FEATURE_LENGTH]))

    raw_both = extract_features([left, right])
    check("2 hands -> exactly 126 values",
          len(raw_both) == MAX_FEATURE_LENGTH, f"len={len(raw_both)}")
    check("2 hands -> both blocks present",
          not all_zero(raw_both[0:HAND_FEATURE_LENGTH]) and
          not all_zero(raw_both[RIGHT_BLOCK_OFFSET:
                                RIGHT_BLOCK_OFFSET + HAND_FEATURE_LENGTH]))
    check("2 hands -> detection order does not matter",
          extract_features([right, left]) == raw_both)

    # --- B. normalization: length + finiteness ----------------------------
    print("\nB) Normalization (build_sample / normalize_features)")
    for name, hands in (("0 hands", []), ("1 hand", [left]),
                        ("2 hands", [left, right])):
        sample = build_sample(hands)
        check(f"{name} -> normalized vector has 126 values",
              len(sample) == MAX_FEATURE_LENGTH, f"len={len(sample)}")
        check(f"{name} -> no NaN / infinite values", finite(sample))

    # --- C. missing-hand blocks stay zeros -------------------------------
    print("\nC) Missing-hand blocks after normalization")
    norm_left = build_sample([left])
    check("1 hand (Left) -> right block still 63 zeros",
          all_zero(norm_left[RIGHT_BLOCK_OFFSET:
                             RIGHT_BLOCK_OFFSET + HAND_FEATURE_LENGTH]))
    norm_right = build_sample([right])
    check("1 hand (Right) -> left block still 63 zeros",
          all_zero(norm_right[LEFT_BLOCK_OFFSET:
                              LEFT_BLOCK_OFFSET + HAND_FEATURE_LENGTH]))
    check("0 hands -> 126 zeros after normalization",
          all_zero(build_sample([])))

    # --- D. per-hand geometry --------------------------------------------
    print("\nD) Per-hand normalization geometry")
    norm_both = build_sample([left, right])

    def block_wrist_distance(vector, offset):
        base = offset  # wrist of that block
        return math.sqrt(sum(vector[base + i] ** 2 for i in range(3)))

    def block_mcp_distance(vector, offset):
        base = offset + 9 * 3
        wrist = vector[offset:offset + 3]
        return math.sqrt(sum((vector[base + i] - wrist[i]) ** 2
                             for i in range(3)))

    check("left hand is centered on its own wrist (wrist = 0,0,0)",
          abs(block_wrist_distance(norm_both, LEFT_BLOCK_OFFSET)) < 1e-12)
    check("right hand is centered on its own wrist (wrist = 0,0,0)",
          abs(block_wrist_distance(norm_both, RIGHT_BLOCK_OFFSET)) < 1e-12)
    check("left hand scaled to size 1 (wrist -> middle MCP distance = 1)",
          abs(block_mcp_distance(norm_both, LEFT_BLOCK_OFFSET) - 1.0) < 1e-9,
          f"got {block_mcp_distance(norm_both, LEFT_BLOCK_OFFSET)}")
    check("right hand scaled to size 1 (independent of left hand)",
          abs(block_mcp_distance(norm_both, RIGHT_BLOCK_OFFSET) - 1.0) < 1e-9,
          f"got {block_mcp_distance(norm_both, RIGHT_BLOCK_OFFSET)}")

    # The right hand was built at 0.6x size: independent normalization must
    # give it the SAME shape as a 1.0x hand at the same relative position.
    reference = build_sample([make_hand("Right", origin=(0.70, 0.45, 0.01),
                                        size=1.0)])[RIGHT_BLOCK_OFFSET:
                                                   RIGHT_BLOCK_OFFSET +
                                                   HAND_FEATURE_LENGTH]
    same_shape = all(abs(norm_both[RIGHT_BLOCK_OFFSET + i] - reference[i])
                     < 1e-9 for i in range(HAND_FEATURE_LENGTH))
    check("size change does not change the normalized shape", same_shape)

    # Position invariance: same hand, moved to another part of the frame
    moved = build_sample([make_hand("Left", origin=(0.10, 0.80, 0.25),
                                    size=1.0)])[0:HAND_FEATURE_LENGTH]
    base = build_sample([make_hand("Left", origin=(0.30, 0.35, -0.02),
                                   size=1.0)])[0:HAND_FEATURE_LENGTH]
    check("position change does not change the normalized shape",
          all(abs(moved[i] - base[i]) < 1e-9
              for i in range(HAND_FEATURE_LENGTH)))

    # --- E. rejection rules ----------------------------------------------
    print("\nE) Invalid samples are rejected (ValueError)")
    expect_value_error("wrong length (125 values) rejected",
                       normalize_features, [0.0] * 125)
    nan_vector = [0.0] * MAX_FEATURE_LENGTH
    nan_vector[5] = float("nan")
    expect_value_error("NaN value rejected", normalize_features, nan_vector)
    inf_vector = [0.0] * MAX_FEATURE_LENGTH
    inf_vector[10] = float("inf")
    expect_value_error("infinite value rejected", normalize_features,
                       inf_vector)
    degenerate = make_hand("Left", size=0.0)  # all landmarks on the wrist
    expect_value_error("degenerate hand (zero size) rejected",
                       build_sample, [degenerate])
    expect_value_error("wrong hand-block length rejected",
                       normalize_hand_block, [0.0] * 62)

    # A rejected sample never reaches the CSV: build_sample raised, so the
    # caller has no vector to write (collector shows the reason instead).
    try:
        build_sample([degenerate])
        wrote_anything = True
    except ValueError:
        wrote_anything = False
    check("rejected sample produces no vector to save", not wrote_anything)

    # --- F. column naming -------------------------------------------------
    print("\nF) CSV column names")
    names = feature_column_names()
    check("126 feature column names", len(names) == 126, f"len={len(names)}")
    check("first column is left_0_x", names[0] == "left_0_x", names[0])
    check("column 62 is left_20_z", names[62] == "left_20_z", names[62])
    check("column 63 is right_0_x", names[63] == "right_0_x", names[63])
    check("column 125 is right_20_z", names[125] == "right_20_z",
          names[125])
    check("no vague names like 'feature1'",
          not any(name.startswith("feature") for name in names))
    header = expected_header()
    check("full CSV header = 126 features + 'label' = 127 columns",
          len(header) == 127 and header[-1] == LABEL_COLUMN,
          f"len={len(header)}, last={header[-1]}")
    check("header order matches the feature order exactly",
          header[:-1] == names)

    # --- consistency of the shared pipeline -------------------------------
    print("\nG) Shared pipeline consistency")
    check("build_sample == normalize(extract(...))",
          build_sample([left, right])
          == normalize_features(extract_features([left, right])))
    check("12 sign classes configured centrally",
          len(SIGN_CLASSES) == 12, str(SIGN_CLASSES))
    check("all configured labels are valid CSV labels",
          all(label.strip() and label == label.strip()
              for label in SIGN_CLASSES))

    # --- summary -----------------------------------------------------------
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
    print("All feature extraction checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

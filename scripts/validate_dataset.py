"""Phase 3 dataset validation: check data/processed/dataset.csv before Phase 4.

This script ONLY inspects the dataset and prints a report. It never modifies,
repairs or deletes anything, and it does not train or evaluate any model.

Checks performed (Phase 3 spec, section 19):
     1. File exists
     2. CSV can be loaded
     3. Column count is exactly 127 (126 features + 1 label)
     4. Feature columns are numeric
     5. No unexpected NaN values
     6. No infinite values
     7. Labels are valid (one of the configured sign classes)
     8. Every feature vector has exactly 126 features
     9. Class counts / class balance (target: SAMPLES_PER_CLASS)
    10. Exact duplicate rows
    11. Missing labels
    12. Missing (and unexpected) required columns

Exit codes:
    0 = structure is valid (class balance may still be below target)
    1 = errors found (or the file is missing/unreadable)

USAGE (from the project folder):
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --path data/processed/dataset.csv
    python scripts/validate_dataset.py --target 150
"""

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

# Make the src/ package importable when this script is run from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_CSV, LABEL_COLUMN, SAMPLES_PER_CLASS, \
    SIGN_CLASSES
from src.dataset_io import expected_header, missing_columns, read_dataset, \
    unexpected_columns
from src.feature_extractor import MAX_FEATURE_LENGTH, feature_column_names

VALID_LABELS = set(SIGN_CLASSES)
MAX_SHOWN_LINES = 10


def _section(title):
    print(f"\n{title}:")


def validate(path: Path, target: int) -> int:
    """Validate the dataset at `path` and print the report.

    Returns 0 when the structure is valid, 1 when errors were found.
    """
    print("=" * 34)
    print("SIGNBRIDGE AI DATASET VALIDATION")
    print("=" * 34)
    print(f"File: {path}")

    errors = []      # structural problems -> exit code 1
    warnings = []    # e.g. classes below target -> still exit code 0

    # 1. File exists -------------------------------------------------------
    if not path.exists():
        _section("File exists")
        print("NO - the dataset has not been collected yet.")
        print(f"\nRESULT: no dataset to validate. Collect samples first:")
        print("    python scripts/collect_data.py --class HELLO")
        return 1

    # 2. CSV can be loaded -------------------------------------------------
    result = read_dataset(path)
    if result.error:
        _section("CSV can be loaded")
        print(f"NO - {result.error}")
        errors.append("CSV could not be read")
        print(f"\nRESULT: ERRORS FOUND ({len(errors)}).")
        return 1

    if not result.header and not result.rows:
        _section("CSV can be loaded")
        print("NO - the file is empty (no header, no rows).")
        errors.append("empty CSV file")
        print(f"\nRESULT: ERRORS FOUND ({len(errors)}).")
        return 1

    # 3. + 12. Column count, missing and unexpected columns ----------------
    header = result.header
    missing = missing_columns(header)
    extra = unexpected_columns(header)
    expected_count = len(expected_header())

    structure_ok = True
    _section("Column count")
    if len(header) == expected_count:
        print(f"{len(header)} (expected {expected_count}) - OK")
    else:
        print(f"{len(header)} (expected {expected_count}) - WRONG")
        errors.append(f"column count is {len(header)}, expected "
                      f"{expected_count}")
        structure_ok = False

    _section("Missing required columns")
    if missing:
        print(f"{len(missing)}: {', '.join(missing[:8])}"
              + ("..." if len(missing) > 8 else ""))
        errors.append(f"{len(missing)} missing column(s)")
        structure_ok = False
    else:
        print("none - OK")

    _section("Unexpected columns")
    if extra:
        print(f"{len(extra)}: {', '.join(extra[:8])}"
              + ("..." if len(extra) > 8 else ""))
        errors.append(f"{len(extra)} unexpected column(s)")
        structure_ok = False
    else:
        print("none - OK")

    # Row-level checks ------------------------------------------------------
    feature_names = feature_column_names()
    valid_rows = 0
    invalid = []                     # (line number, reason)
    nan_cells = 0
    inf_cells = 0
    non_numeric_cells = 0
    label_counts = Counter()         # valid rows per class
    missing_labels = 0
    unknown_labels = Counter()
    row_feature_lengths = Counter()  # how many features each row has
    duplicates = Counter()           # exact full row -> occurrences

    for line_number, cells in result.rows:
        duplicates[tuple(cells)] += 1

        if len(cells) != expected_count:
            invalid.append((line_number,
                            f"{len(cells)} columns instead of "
                            f"{expected_count}"))
            continue

        # Parse the 126 feature cells
        features = []
        row_reason = None
        non_numeric_in_row = 0
        for index, cell in enumerate(cells[:-1]):
            try:
                value = float(cell)
            except (TypeError, ValueError):
                non_numeric_in_row += 1
                row_reason = (f"column '{feature_names[index]}' is not a "
                              f"number ({cell!r})")
                break
            if math.isnan(value):
                nan_cells += 1
            elif math.isinf(value):
                inf_cells += 1
            features.append(value)
        non_numeric_cells += non_numeric_in_row

        if row_reason is None:
            if any(math.isnan(value) or math.isinf(value)
                   for value in features):
                row_reason = "row contains NaN or infinite values"

        if row_reason is None and non_numeric_in_row == 0:
            row_feature_lengths[len(features)] += 1
            if len(features) != MAX_FEATURE_LENGTH:
                row_reason = (f"{len(features)} features instead of "
                              f"{MAX_FEATURE_LENGTH}")

        # Label checks
        if row_reason is None:
            label = cells[-1].strip()
            if not label:
                missing_labels += 1
                row_reason = "missing label"
            elif label not in VALID_LABELS:
                unknown_labels[label] += 1
                row_reason = f"unknown label '{label}'"
            else:
                valid_rows += 1
                label_counts[label] += 1

        if row_reason:
            invalid.append((line_number, row_reason))

    duplicate_rows = sum(count - 1 for count in duplicates.values()
                         if count > 1)

    # --- Report ------------------------------------------------------------
    _section("Dataset shape")
    print(f"{len(result.rows)} rows x {len(header)} columns")
    print(f"Valid samples: {valid_rows}")

    _section("Class distribution")
    for name in SIGN_CLASSES:
        print(f"{name}: {label_counts.get(name, 0)}")
    for name in sorted(unknown_labels):
        print(f"{name}: {unknown_labels[name]}  (UNKNOWN LABEL)")

    _section("Feature count")
    if row_feature_lengths and set(row_feature_lengths) == \
            {MAX_FEATURE_LENGTH} and structure_ok:
        print(f"{MAX_FEATURE_LENGTH} (every row) - OK")
    else:
        print(f"expected {MAX_FEATURE_LENGTH}; observed: "
              f"{dict(sorted(row_feature_lengths.items()))}")
        if set(row_feature_lengths) - {MAX_FEATURE_LENGTH}:
            errors.append("rows with a wrong feature count")

    _section("Label column")
    print(LABEL_COLUMN if LABEL_COLUMN in header else
          f"MISSING (expected '{LABEL_COLUMN}')")

    _section("Invalid rows")
    print(len(invalid))
    for line_number, reason in invalid[:MAX_SHOWN_LINES]:
        print(f"    line {line_number}: {reason}")
    if len(invalid) > MAX_SHOWN_LINES:
        print(f"    ... and {len(invalid) - MAX_SHOWN_LINES} more")
    if invalid:
        errors.append(f"{len(invalid)} invalid row(s)")

    _section("NaN values")
    print(nan_cells if nan_cells else "0")
    if nan_cells:
        errors.append(f"{nan_cells} NaN value(s)")

    _section("Infinite values")
    print(inf_cells if inf_cells else "0")
    if inf_cells:
        errors.append(f"{inf_cells} infinite value(s)")

    _section("Non-numeric feature cells")
    print(non_numeric_cells if non_numeric_cells else "0")
    if non_numeric_cells:
        errors.append(f"{non_numeric_cells} non-numeric cell(s)")

    _section("Missing labels")
    print(missing_labels if missing_labels else "0")
    if missing_labels:
        errors.append(f"{missing_labels} row(s) without a label")

    _section("Unknown labels")
    if unknown_labels:
        print(f"{sum(unknown_labels.values())}: "
              f"{', '.join(sorted(unknown_labels))}")
        errors.append(f"{sum(unknown_labels.values())} row(s) with an "
                      f"unknown label")
    else:
        print("0")

    _section("Exact duplicate rows")
    print(duplicate_rows if duplicate_rows else "0")
    if duplicate_rows:
        errors.append(f"{duplicate_rows} exact duplicate row(s)")

    # --- Class balance (section 20) ---------------------------------------
    _section(f"Class balance (target {target} per class)")
    below = []
    above = []
    for name in SIGN_CLASSES:
        count = label_counts.get(name, 0)
        if count >= target and count != target:
            above.append(name)
            print(f"{name:<12} {count:>4} / {target}  (above target)")
        elif count < target:
            below.append(name)
            print(f"{name:<12} {count:>4} / {target}  "
                  f"(UNDER TARGET, missing {target - count})")
        else:
            print(f"{name:<12} {count:>4} / {target}  OK")

    if below:
        warnings.append(f"{len(below)} class(es) below target: "
                        f"{', '.join(below)}")
    if above:
        warnings.append(f"class(es) above target: {', '.join(above)}")

    # --- Final verdict -----------------------------------------------------
    print("\n" + "-" * 34)
    if errors:
        print(f"RESULT: ERRORS FOUND ({len(errors)}):")
        for message in errors:
            print(f"    - {message}")
        print("Fix these before Phase 4 (do NOT delete samples blindly - "
              "re-collect where possible).")
        return 1

    print("RESULT: dataset structure is VALID.")
    print(f"    {valid_rows} rows x {expected_count} columns "
          f"({MAX_FEATURE_LENGTH} numeric features + '{LABEL_COLUMN}').")
    if warnings:
        print("Warnings:")
        for message in warnings:
            print(f"    - {message}")
        print("Collect the missing samples before training a model.")
    else:
        print("All classes reached the target sample count.")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate the SignBridge AI dataset (no model training).")
    parser.add_argument("--path", default=str(DATASET_CSV),
                        help=f"CSV file to validate (default: {DATASET_CSV})")
    parser.add_argument("--target", type=int, default=SAMPLES_PER_CLASS,
                        help="target samples per class used for the balance "
                             f"report (default: {SAMPLES_PER_CLASS})")
    args = parser.parse_args()
    if args.target <= 0:
        parser.error("--target must be a positive number")
    return args


def main():
    args = parse_args()
    return validate(Path(args.path), args.target)


if __name__ == "__main__":
    sys.exit(main())

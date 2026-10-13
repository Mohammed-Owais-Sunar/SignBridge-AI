"""Phase 4: train and evaluate the gesture classifier (Random Forest).

WHAT THIS SCRIPT DOES
1.  Loads data/processed/dataset.csv through src.dataset_io - the SAME
    validated 127-column format Phase 3 wrote. Nothing is modified.
2.  Validates the required columns and every row (126 finite features,
    label one of SIGN_CLASSES).
3.  Separates X (the 126 feature columns) from y (the label column). The
    label column NEVER enters X.
4.  Reproducible stratified train/test split:
    test_size = 0.20, stratify = y, random_state = 42.
5.  Trains a RandomForestClassifier on the TRAINING split only.
6.  Predicts the held-out TEST split and computes accuracy, precision,
    recall and F1 (macro and weighted), a classification report and a
    confusion matrix.
7.  Saves the model (joblib), the class list, and the evaluation files
    under evaluation/.
8.  Reloads the saved model and re-checks it: feature count, class list,
    and prediction on real test samples.

DATA LEAKAGE CONTROLS
-   X contains only feature columns; the label column is stripped before
    the split and can never be learned as a feature.
-   The split happens BEFORE fitting. model.fit() receives X_train and
    y_train only; the test split is touched only by predict()/metrics.
-   No preprocessing is fitted on data at all: Phase 3 normalization is
    deterministic and PER SAMPLE (each hand is wrist-centered and
    hand-size-scaled on its own), so it carries no dataset statistics and
    cannot leak test information into training.
-   random_state is fixed everywhere, so the same dataset always produces
    the same split, the same model and the same metrics.

WHAT THIS SCRIPT DOES NOT DO (later phases)
-   no webcam prediction, no real-time recognition, no prediction
    smoothing, no sentence builder, no SPACE/DELETE logic, no
    text-to-speech, no Streamlit UI.

USAGE (from the project folder):
    python scripts/train_model.py
    python scripts/train_model.py --dataset data/processed/dataset.csv
"""

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

# Make the src/ package importable when this script is run from anywhere
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (CLASSES_PATH, DATASET_CSV, EVALUATION_DIR,
                        LABEL_COLUMN, RANDOM_FOREST_PARAMS, RANDOM_STATE,
                        SIGN_CLASSES, TEST_SIZE, TRAINED_MODEL_PATH)
from src.dataset_io import expected_header, parse_row, read_dataset
from src.feature_extractor import MAX_FEATURE_LENGTH, feature_column_names

CLASS_REPORT_TXT = "classification_report.txt"
CONFUSION_CSV = "confusion_matrix.csv"
CONFUSION_PNG = "confusion_matrix.png"
METRICS_JSON = "metrics.json"


# -----------------------------------------------------------------------------
# Dataset loading (read-only, reuses the Phase 3 format checks)
# -----------------------------------------------------------------------------
def load_dataset(path: Path):
    """Load dataset.csv into (X, y, stats).

    X: float64 array of shape (n, 126) in feature-column order.
    y: array of label strings.

    Raises SystemExit with a readable reason when the file is missing, has
    the wrong header, or contains rows that fail the Phase 3 format rules.
    """
    result = read_dataset(path)
    if result.error:
        raise SystemExit(f"Error: could not read {path}: {result.error}")
    if not result.exists:
        raise SystemExit(f"Error: dataset not found: {path}\n"
                         f"Collect it first: python scripts/collect_data.py")
    if not result.rows:
        raise SystemExit(f"Error: {path} contains no samples.")

    # Column validation: exactly the Phase 3 header, in the exact order
    if result.header != expected_header():
        missing = [name for name in expected_header() if name not in result.header]
        extra = [name for name in result.header if name not in expected_header()]
        raise SystemExit(
            f"Error: {path} does not match the Phase 3 dataset format.\n"
            f"    missing columns: {len(missing)}  unexpected columns: {len(extra)}\n"
            f"    Run: python scripts/validate_dataset.py")

    features = []
    labels = []
    invalid = []  # (line number, reason) - training must never use these

    for line_number, cells in result.rows:
        try:
            row_features, label = parse_row(cells)
        except ValueError as error:
            invalid.append((line_number, str(error)))
            continue
        if len(row_features) != MAX_FEATURE_LENGTH:
            invalid.append((line_number,
                            f"{len(row_features)} features instead of "
                            f"{MAX_FEATURE_LENGTH}"))
            continue
        if not all(math.isfinite(value) for value in row_features):
            invalid.append((line_number, "row contains NaN or infinite values"))
            continue
        if not label:
            invalid.append((line_number, "row has no label"))
            continue
        if label not in SIGN_CLASSES:
            invalid.append((line_number, f"unknown label '{label}'"))
            continue
        features.append(row_features)
        labels.append(label)

    if invalid:
        details = "; ".join(f"line {line}: {reason}"
                            for line, reason in invalid[:5])
        raise SystemExit(
            f"Error: {len(invalid)} invalid row(s) in {path} "
            f"({details}{'; ...' if len(invalid) > 5 else ''}).\n"
            f"    Run: python scripts/validate_dataset.py - do not train on "
            f"a broken dataset.")

    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)

    stats = {
        "path": str(path),
        "total_rows": len(result.rows),
        "valid_rows": len(y),
        "feature_count": int(X.shape[1]),
        "class_counts": dict(sorted(Counter(y.tolist()).items())),
    }
    return X, y, stats


# -----------------------------------------------------------------------------
# Evaluation outputs
# -----------------------------------------------------------------------------
def write_classification_report(path: Path, report_text: str, accuracy: float,
                                precision_macro: float, recall_macro: float,
                                f1_macro: float, precision_weighted: float,
                                recall_weighted: float, f1_weighted: float,
                                params: dict, train_n: int, test_n: int,
                                dataset_stats: dict) -> None:
    lines = [
        "SIGNBRIDGE AI - PHASE 4 CLASSIFICATION REPORT",
        "=" * 60,
        f"Dataset          : {dataset_stats['path']}",
        f"Samples (valid)  : {dataset_stats['valid_rows']} "
        f"({train_n} train / {test_n} test, test_size={TEST_SIZE}, "
        f"stratified, random_state={RANDOM_STATE})",
        f"Features         : {dataset_stats['feature_count']}",
        f"Classes          : {len(SIGN_CLASSES)}",
        f"Model            : RandomForestClassifier",
        f"Parameters       : {json.dumps(params, sort_keys=True)}",
        "",
        "OVERALL METRICS",
        f"Accuracy         : {accuracy:.4f}",
        f"Precision (macro)      : {precision_macro:.4f}",
        f"Recall (macro)         : {recall_macro:.4f}",
        f"F1 (macro)             : {f1_macro:.4f}",
        f"Precision (weighted)   : {precision_weighted:.4f}",
        f"Recall (weighted)      : {recall_weighted:.4f}",
        f"F1 (weighted)          : {f1_weighted:.4f}",
        "",
        "PER-CLASS REPORT (test set, class order = Phase 3 SIGN_CLASSES)",
        report_text,
        "",
        "Notes:",
        "- Macro/weighted averages are identical here because every class has",
        "  the same number of test samples (stratified split of a balanced",
        "  dataset).",
        "- Metrics are computed ONLY on the held-out test split; the model",
        "  was fit exclusively on the training split.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_confusion_csv(path: Path, matrix: np.ndarray, labels) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\predicted"] + list(labels))
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [int(value) for value in row])


def write_confusion_png(path: Path, matrix: np.ndarray, labels) -> bool:
    """Save a graphical confusion matrix. Returns False if matplotlib is
    unavailable (the CSV is always written, the PNG is best-effort)."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless: never opens a window
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(labels)),
           yticks=np.arange(len(labels)),
           xticklabels=labels,
           yticklabels=labels,
           ylabel="True label",
           xlabel="Predicted label",
           title="SignBridge AI - confusion matrix (held-out test set)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(int(matrix[row, column])),
                    ha="center", va="center",
                    color="white" if matrix[row, column] > threshold
                    else "black",
                    fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# -----------------------------------------------------------------------------
# Saved-model cross-check (section 7 of the Phase 4 spec)
# -----------------------------------------------------------------------------
def verify_saved_model(model_path: Path, expected_model, X_test, y_test) -> bool:
    """Reload the saved model and verify it behaves like the trained one."""
    print("\nSaved-model cross-check:")
    checks = []

    try:
        loaded = joblib.load(model_path)
        checks.append(("model loads with joblib", True, ""))
    except Exception as error:  # noqa: BLE001 - report any load failure
        checks.append(("model loads with joblib", False, str(error)))
        _print_checks(checks)
        return False

    checks.append(("is a RandomForestClassifier",
                   isinstance(loaded, RandomForestClassifier),
                   type(loaded).__name__))

    n_features = getattr(loaded, "n_features_in_", None)
    checks.append((f"feature count matches training input "
                   f"({MAX_FEATURE_LENGTH})",
                   n_features == MAX_FEATURE_LENGTH,
                   f"n_features_in_={n_features}"))

    loaded_classes = list(getattr(loaded, "classes_", []))
    checks.append(("classes identical to Phase 3 SIGN_CLASSES (as a set)",
                   set(loaded_classes) == set(SIGN_CLASSES),
                   f"{loaded_classes}"))

    sample = X_test[:32]
    checks.append(("test features contain no NaN/inf",
                   bool(np.isfinite(sample).all()), ""))
    try:
        predictions = loaded.predict(sample)
        ok = (len(predictions) == len(sample)
              and all(label in SIGN_CLASSES for label in predictions))
        checks.append(("prediction on test samples works, labels valid",
                       ok, f"{len(predictions)} predictions"))
        matches = bool(np.array_equal(predictions,
                                      expected_model.predict(sample)))
        checks.append(("reloaded model predicts identically to the trained one",
                       matches, ""))
        if hasattr(loaded, "predict_proba"):
            proba = loaded.predict_proba(sample)
            checks.append(("predict_proba shape = (samples, classes)",
                           proba.shape == (len(sample), len(SIGN_CLASSES)),
                           f"{proba.shape}"))
            checks.append(("predict_proba values finite and in [0, 1]",
                           bool(np.isfinite(proba).all()
                                and (proba >= 0).all()
                                and (proba <= 1).all()), ""))
    except Exception as error:  # noqa: BLE001 - any predict failure is a fail
        checks.append(("prediction on test samples works, labels valid",
                       False, str(error)))

    # y_test itself: sanity that evaluation labels are Phase 3 labels
    checks.append(("evaluation labels are Phase 3 classes",
                   set(y_test.tolist()) <= set(SIGN_CLASSES), ""))

    _print_checks(checks)
    return all(ok for _, ok, _ in checks)


def _print_checks(checks) -> None:
    for name, ok, detail in checks:
        suffix = f"  ({detail})" if detail and not ok else \
                 (f"  ({detail})" if detail and ok else "")
        print(f"    [{'PASS' if ok else 'FAIL'}] {name}{suffix}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the SignBridge AI gesture classifier "
                    "(Phase 4). Reads the dataset read-only; writes the model "
                    "to models/ and reports to evaluation/.")
    parser.add_argument("--dataset", default=str(DATASET_CSV),
                        help=f"dataset CSV to train on (default: {DATASET_CSV})")
    args = parser.parse_args()
    return args


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    started = time.monotonic()

    print("=" * 60)
    print("SIGNBRIDGE AI - TRAINING (Phase 4)")
    print("=" * 60)

    # 1-3. Locate, load and validate the dataset --------------------------
    X, y, stats = load_dataset(dataset_path)
    print(f"Dataset      : {stats['path']}")
    print(f"Samples      : {stats['valid_rows']} valid "
          f"(rows read: {stats['total_rows']}, invalid: 0)")
    print(f"Features     : {stats['feature_count']} per sample "
          f"(label column '{LABEL_COLUMN}' excluded from X)")
    print(f"Classes ({len(stats['class_counts'])}):")
    for name, count in stats["class_counts"].items():
        print(f"    {name:<12}{count:>5}")

    if stats["feature_count"] != MAX_FEATURE_LENGTH:
        raise SystemExit(f"Error: expected {MAX_FEATURE_LENGTH} features, "
                         f"got {stats['feature_count']}.")
    if set(stats["class_counts"]) != set(SIGN_CLASSES):
        raise SystemExit("Error: dataset classes do not match Phase 3 "
                         f"SIGN_CLASSES: {sorted(stats['class_counts'])}")

    # 4. Train/test split (stratified, reproducible) ----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    print("\nTrain/test split:")
    print(f"    test_size={TEST_SIZE}  stratify=y  random_state={RANDOM_STATE}")
    print(f"    training samples : {len(X_train)}")
    print(f"    testing samples  : {len(X_test)}")
    print(f"    features         : {X_train.shape[1]}")
    print(f"    classes          : {len(set(y_train.tolist()))} "
          f"(train) / {len(set(y_test.tolist()))} (test)")

    # 5. Train on the TRAINING split only ---------------------------------
    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    print("\nTraining RandomForestClassifier "
          f"({json.dumps(RANDOM_FOREST_PARAMS, sort_keys=True)}) ...")
    model.fit(X_train, y_train)
    print("Training finished.")

    # 6-7. Predict the held-out test split and evaluate -------------------
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", labels=SIGN_CLASSES, zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", labels=SIGN_CLASSES,
        zero_division=0)
    report_text = classification_report(
        y_test, y_pred, labels=SIGN_CLASSES, target_names=SIGN_CLASSES,
        zero_division=0)
    matrix = confusion_matrix(y_test, y_pred, labels=SIGN_CLASSES)

    print("\nEvaluation (held-out test set):")
    print(f"    Accuracy          : {accuracy:.4f}")
    print(f"    Precision (macro) : {p_macro:.4f}")
    print(f"    Recall (macro)    : {r_macro:.4f}")
    print(f"    F1 (macro)        : {f1_macro:.4f}")
    print(f"    Precision (weighted): {p_weighted:.4f}")
    print(f"    Recall (weighted)   : {r_weighted:.4f}")
    print(f"    F1 (weighted)      : {f1_weighted:.4f}")

    # 8. Save model + class information -----------------------------------
    TRAINED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, TRAINED_MODEL_PATH)
    joblib.dump(list(model.classes_), CLASSES_PATH)
    print(f"\nSaved model  : {TRAINED_MODEL_PATH}")
    print(f"Saved classes: {CLASSES_PATH}  ({len(model.classes_)} labels)")

    # 9. Save evaluation results ------------------------------------------
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EVALUATION_DIR / CLASS_REPORT_TXT
    confusion_csv_path = EVALUATION_DIR / CONFUSION_CSV
    confusion_png_path = EVALUATION_DIR / CONFUSION_PNG
    metrics_path = EVALUATION_DIR / METRICS_JSON

    write_classification_report(
        report_path, report_text, accuracy,
        float(p_macro), float(r_macro), float(f1_macro),
        float(p_weighted), float(r_weighted), float(f1_weighted),
        RANDOM_FOREST_PARAMS, len(X_train), len(X_test), stats)
    write_confusion_csv(confusion_csv_path, matrix, SIGN_CLASSES)
    png_written = write_confusion_png(confusion_png_path, matrix, SIGN_CLASSES)

    metrics = {
        "phase": 4,
        "dataset": {
            "path": stats["path"],
            "samples": stats["valid_rows"],
            "features": stats["feature_count"],
            "classes": list(SIGN_CLASSES),
            "class_counts": stats["class_counts"],
        },
        "split": {
            "test_size": TEST_SIZE,
            "stratify": "y",
            "random_state": RANDOM_STATE,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
        "model": {
            "type": "RandomForestClassifier",
            "parameters": {key: value for key, value
                           in RANDOM_FOREST_PARAMS.items()},
        },
        "metrics": {
            "accuracy": accuracy,
            "precision_macro": float(p_macro),
            "recall_macro": float(r_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(p_weighted),
            "recall_weighted": float(r_weighted),
            "f1_weighted": float(f1_weighted),
        },
        "artifacts": {
            "model": str(TRAINED_MODEL_PATH),
            "classes": str(CLASSES_PATH),
            "classification_report": str(report_path),
            "confusion_matrix_csv": str(confusion_csv_path),
            "confusion_matrix_png": str(confusion_png_path)
            if png_written else None,
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n",
                            encoding="utf-8")

    print("\nSaved evaluation files:")
    print(f"    {report_path}")
    print(f"    {confusion_csv_path}")
    if png_written:
        print(f"    {confusion_png_path}")
    else:
        print(f"    (PNG skipped: matplotlib not available)")
    print(f"    {metrics_path}")

    # 10. Reload cross-check ----------------------------------------------
    reload_ok = verify_saved_model(TRAINED_MODEL_PATH, model, X_test, y_test)

    elapsed = time.monotonic() - started
    print("\n" + "=" * 60)
    if reload_ok:
        print(f"PHASE 4 TRAINING COMPLETE ({elapsed:.1f} s).")
        print("Model reload cross-check: PASSED")
        print("Next step (a later phase): real-time prediction - NOT part "
              "of Phase 4.")
        print("=" * 60)
        return 0

    print("TRAINING FINISHED BUT THE RELOAD CROSS-CHECK FAILED.")
    print("Do not use the saved model until this is fixed.")
    print("=" * 60)
    return 1


if __name__ == "__main__":
    sys.exit(main())

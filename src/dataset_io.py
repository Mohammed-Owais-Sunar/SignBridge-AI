"""Safe CSV reading/writing for data/processed/dataset.csv (Phase 3).

Single source of truth for the dataset format - both the collector and the
validator import from here, so they can never disagree:

    dataset.csv = 126 feature columns + 1 label column = 127 columns

        left_0_x, left_0_y, left_0_z, ..., left_20_z,
        right_0_x, ..., right_20_z,
        label

    - Feature values are the NORMALIZED 126-value vector produced by
      src.feature_extractor.build_sample().
    - Labels are human-readable strings ("HELLO", "YES", ...). They are NOT
      encoded to integers in Phase 3 (encoding happens in Phase 4, if needed).

Safety rules implemented here:

- APPEND ONLY: existing samples are never deleted or overwritten, so closing
  the terminal mid-collection keeps everything that was already saved
  (resume support: the next run continues where the previous one stopped).
- The header is written exactly once, when the file is created. A file whose
  header does not match the expected format is REFUSED (DatasetError) instead
  of being silently corrupted.
- Every row is flushed to disk immediately after it is written, so a crash or
  Ctrl+C cannot corrupt already saved samples.
- dataset.csv is only CREATED when the first sample is actually written.
"""

import csv
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

from src.config import DATASET_CSV, LABEL_COLUMN, PROCESSED_DIR, RAW_DIR, \
    SIGN_CLASSES
from src.feature_extractor import MAX_FEATURE_LENGTH, feature_column_names, \
    validate_feature_vector


class DatasetError(Exception):
    """Raised when dataset.csv cannot be used safely (never hidden)."""


def expected_header() -> List[str]:
    """The full 127-column header: 126 feature names + the label column."""
    return feature_column_names() + [LABEL_COLUMN]


def ensure_data_dirs() -> None:
    """Create data/raw/ and data/processed/ if they do not exist yet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class CsvReadResult:
    """Raw, structure-only view of a CSV file (no interpretation)."""

    path: Path
    exists: bool
    header: List[str] = field(default_factory=list)
    rows: List[Tuple[int, List[str]]] = field(default_factory=list)
    # (1-based file line number, cells) for every data row
    error: Optional[str] = None  # read/parse failure, if any


def read_dataset(path: Path = DATASET_CSV) -> CsvReadResult:
    """Read a CSV file without judging its content (the caller decides).

    Returns a CsvReadResult with `error` set when the file could not be read
    at all; a missing file is NOT an error (exists=False).
    """
    path = Path(path)
    if not path.exists():
        return CsvReadResult(path=path, exists=False)

    try:
        with open(path, "r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                header = []  # empty file
            rows = [(reader.line_num, cells) for cells in reader]
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        return CsvReadResult(path=path, exists=True, error=str(error))

    return CsvReadResult(path=path, exists=True, header=header, rows=rows)


def missing_columns(header: Sequence[str]) -> List[str]:
    """Expected columns that are absent from `header`."""
    return [name for name in expected_header() if name not in header]


def unexpected_columns(header: Sequence[str]) -> List[str]:
    """Columns in `header` that do not belong to the dataset format."""
    expected = set(expected_header())
    return [name for name in header if name not in expected]


def parse_row(cells: Sequence[str]) -> Tuple[List[float], str]:
    """Parse one data row into (126 feature floats, label string).

    Raises:
        ValueError: with a readable reason when the row does not have exactly
            127 columns or a feature cell is not a number.
    """
    expected = expected_header()
    if len(cells) != len(expected):
        raise ValueError(f"row has {len(cells)} columns, expected "
                         f"{len(expected)}")

    label = cells[-1].strip()
    features: List[float] = []
    for index, cell in enumerate(cells[:-1]):
        try:
            features.append(float(cell))
        except (TypeError, ValueError):
            raise ValueError(f"column '{expected[index]}' is not a number "
                             f"({cell!r})") from None
    return features, label


@dataclass
class Progress:
    """Everything the collector needs to resume an interrupted session."""

    counts: Counter = field(default_factory=Counter)  # label -> saved rows
    seen: Set[Tuple] = field(default_factory=set)     # exact-duplicate guard
    malformed: List[Tuple[int, str]] = field(default_factory=list)
    unknown_labels: Counter = field(default_factory=Counter)
    valid_rows: int = 0
    total_rows: int = 0


def load_progress(path: Path = DATASET_CSV) -> Progress:
    """Load existing samples so collection can RESUME instead of restart.

    Only rows that are structurally valid (127 columns, numeric finite
    features, known or unknown-but-parseable label) are counted and used for
    duplicate detection. Problem rows are reported in `malformed` - they are
    NEVER deleted or modified here.

    Raises:
        DatasetError: when the file exists but cannot be used (unreadable,
            or a header that does not match the dataset format). The caller
            must stop instead of appending to a wrong file.
    """
    result = read_dataset(path)
    if result.error:
        raise DatasetError(f"could not read {path}: {result.error}")
    if not result.exists:
        return Progress()  # nothing collected yet - start fresh

    progress = Progress(total_rows=len(result.rows))
    if not result.rows and not result.header:
        return progress  # empty file: the writer will add the header later

    missing = missing_columns(result.header)
    extra = unexpected_columns(result.header)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing[:5])}"
                           + ("..." if len(missing) > 5 else ""))
        if extra:
            details.append(f"unexpected: {', '.join(extra[:5])}"
                           + ("..." if len(extra) > 5 else ""))
        raise DatasetError(
            f"{path} does not have the expected header ({'; '.join(details)}). "
            f"Nothing was modified - move or fix the file first.")

    for line_number, cells in result.rows:
        try:
            features, label = parse_row(cells)
        except ValueError as error:
            progress.malformed.append((line_number, str(error)))
            continue
        if not all(math.isfinite(value) for value in features):
            progress.malformed.append(
                (line_number, "row contains NaN or infinite values"))
            continue
        if len(features) != MAX_FEATURE_LENGTH:
            progress.malformed.append(
                (line_number, f"row has {len(features)} features"))
            continue

        if not label:
            progress.malformed.append((line_number, "row has no label"))
            continue
        if label not in SIGN_CLASSES:
            # A label we do not know: keep the row, but it cannot be counted
            # towards any class.
            progress.unknown_labels[label] += 1
        else:
            progress.counts[label] += 1
            progress.valid_rows += 1
        progress.seen.add((label, tuple(features)))

    return progress


class DatasetWriter:
    """Append-only writer for dataset.csv (resume-safe, flush-per-row).

    The file (and its header) is created lazily on the first `write()` call,
    so merely opening the collector does NOT create dataset.csv.

        writer = DatasetWriter()          # validates an existing file only
        writer.write(features_126, "HELLO")
        writer.close()
    """

    def __init__(self, path: Path = DATASET_CSV):
        self.path = Path(path)
        self._handle = None
        self._writer = None

        # Read-only safety check of an existing file (nothing is touched yet)
        if self.path.exists() and self.path.stat().st_size > 0:
            result = read_dataset(self.path)
            if result.error:
                raise DatasetError(f"could not read {self.path}: "
                                   f"{result.error}")
            missing = missing_columns(result.header)
            extra = unexpected_columns(result.header)
            if missing or extra:
                bits = []
                if missing:
                    bits.append(f"missing columns: {', '.join(missing[:5])}")
                if extra:
                    bits.append(f"unexpected columns: "
                                f"{', '.join(extra[:5])}")
                raise DatasetError(
                    f"{self.path} has the wrong header ({'; '.join(bits)}). "
                    f"Refusing to write to it - nothing was modified. "
                    f"Move or fix the file first.")

    def write(self, features: Sequence[float], label: str) -> None:
        """Append one sample (126 normalized features + label) and flush it.

        Raises:
            DatasetError: invalid features/label or an I/O problem. Nothing
                is written when validation fails.
        """
        if label not in SIGN_CLASSES:
            raise DatasetError(f"invalid label {label!r} - expected one of "
                               f"{', '.join(SIGN_CLASSES)}")
        try:
            validate_feature_vector(list(features), MAX_FEATURE_LENGTH)
        except ValueError as error:
            raise DatasetError(f"invalid feature vector: {error}") from None

        if self._handle is None:
            self._open()

        try:
            self._writer.writerow(list(features) + [label])
            self._handle.flush()  # row survives even if the program dies now
        except OSError as error:
            raise DatasetError(f"could not write to {self.path}: {error}")

    def _open(self) -> None:
        """Open the file in append mode, creating it with a header if new."""
        try:
            ensure_data_dirs()
            is_new = (not self.path.exists()
                      or self.path.stat().st_size == 0)
            self._handle = open(self.path, "a", newline="", encoding="utf-8")
        except OSError as error:
            raise DatasetError(f"could not open {self.path} for writing: "
                               f"{error}") from error

        self._writer = csv.writer(self._handle)
        if is_new:
            self._writer.writerow(expected_header())
            self._handle.flush()

    @property
    def created(self) -> bool:
        """True once the file has actually been written to."""
        return self._handle is not None

    def close(self) -> None:
        """Flush and close the file (safe to call more than once)."""
        if self._handle is not None:
            try:
                self._handle.flush()
                self._handle.close()
            finally:
                self._handle = None
                self._writer = None

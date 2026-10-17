"""Strict source loading and content-addressed provenance."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import COLUMNS, EXPECTED_ROWS, FIELDS, TARGET


class DataQualityError(ValueError):
    """The source does not meet the declared input contract."""


def _finite_bound(values, operation):
    result = getattr(pd.to_numeric(values, errors="coerce"), operation)()
    return float(result) if np.isfinite(result) else None


def audit(frame):
    """Describe source defects without silently coercing or dropping them."""
    issues = {}
    for field in FIELDS:
        values = frame[field.name]
        numeric = pd.to_numeric(values, errors="coerce")
        invalid = values.notna() & (
            numeric.isna() | ~np.isfinite(numeric) | (numeric % 1 != 0)
            | (numeric < field.minimum) | (numeric > field.maximum)
        )
        if invalid.any():
            issues[field.name] = int(invalid.sum())
    return {
        "rows": len(frame), "columns": len(frame.columns),
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "missing": frame.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "invalid_values": issues,
        "target_counts": {str(k): int(v) for k, v in frame[TARGET].value_counts(dropna=False).items()},
        "ranges": {field.name: {
            "min": _finite_bound(frame[field.name], "min"),
            "max": _finite_bound(frame[field.name], "max"),
        } for field in FIELDS},
    }


def load_source(path, expected_rows=EXPECTED_ROWS):
    path = Path(path)
    frame = pd.read_csv(path)
    if list(frame.columns) != COLUMNS:
        raise DataQualityError(f"Expected these 22 columns in order: {COLUMNS}; got {list(frame.columns)}")
    if len(frame) != expected_rows:
        raise DataQualityError(f"Expected {expected_rows:,} rows, got {len(frame):,}; check the dataset variant")
    report = audit(frame)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    report["source"] = {"filename": path.name, "sha256": digest.hexdigest()}
    return frame, report


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

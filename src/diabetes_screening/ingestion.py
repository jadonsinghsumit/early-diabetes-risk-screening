"""Strict source loading and content-addressed provenance."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import COLUMNS, EXPECTED_ROWS, FIELDS, TARGET


class DataQualityError(ValueError):
    """The source does not meet the declared input contract."""


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
            "min": float(pd.to_numeric(frame[field.name], errors="coerce").min()),
            "max": float(pd.to_numeric(frame[field.name], errors="coerce").max()),
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
    report["source"] = {"filename": path.name, "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()}
    return frame, report


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

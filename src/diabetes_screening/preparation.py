"""Explicitly derive a binary endpoint from the uploaded three-class table."""
import hashlib
import io
from pathlib import Path
import zipfile

import pandas as pd

from .cleaning import clean_data
from .endpoints import ENDPOINTS
from .ingestion import DataQualityError, write_json
from .schema import COLUMNS, EXPECTED_ROWS, TARGET

THREE_CLASS_FILE = "diabetes_012_health_indicators_BRFSS2015.csv"


def prepare_binary(source, destination, endpoint, expected_rows=EXPECTED_ROWS):
    """Keep original class counts and hashes; never pass off a derived CSV as upstream."""
    source, destination = Path(source), Path(destination)
    sidecar = destination.with_suffix(".provenance.json")
    if destination.exists() or sidecar.exists():
        raise FileExistsError("Prepared CSV or provenance already exists; use a new destination")
    if endpoint not in ENDPOINTS:
        raise ValueError(f"Endpoint must be one of {list(ENDPOINTS)}")
    uploaded = source.read_bytes()
    member = None
    if zipfile.is_zipfile(io.BytesIO(uploaded)):
        with zipfile.ZipFile(io.BytesIO(uploaded)) as archive:
            members = [name for name in archive.namelist() if Path(name).name == THREE_CLASS_FILE]
            if len(members) != 1:
                raise DataQualityError(f"Expected exactly one {THREE_CLASS_FILE}")
            member = members[0]
            csv_bytes = archive.read(member)
    else:
        csv_bytes = uploaded
    frame = pd.read_csv(io.BytesIO(csv_bytes))
    if list(frame.columns) != ["Diabetes_012", *COLUMNS[1:]] or len(frame) != expected_rows:
        raise DataQualityError(f"Expected three-class schema and {expected_rows:,} rows")
    if not frame["Diabetes_012"].isin([0, 1, 2]).all():
        raise DataQualityError("Three-class target contains missing or invalid labels")
    original_counts = frame["Diabetes_012"].value_counts().sort_index()
    frame = frame.rename(columns={"Diabetes_012": TARGET})
    frame[TARGET] = frame[TARGET].map(ENDPOINTS[endpoint])
    excluded_rows = int(frame[TARGET].isna().sum())
    frame = frame.loc[frame[TARGET].notna()].copy()
    cleaned, _ = clean_data(frame)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Unix line endings on every platform, so the derived hash is the same on Windows, macOS and Linux.
    cleaned.to_csv(destination, index=False, lineterminator="\n")
    metadata = {
        "source_filename": source.name, "archive_member": member,
        "source_sha256": hashlib.sha256(uploaded).hexdigest(),
        "source_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "derived_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "endpoint": endpoint, "label_mapping": ENDPOINTS[endpoint],
        "original_target_counts": {str(int(k)): int(v) for k, v in original_counts.items()},
        "derived_target_counts": {str(k): int(v) for k, v in cleaned[TARGET].value_counts().sort_index().items()},
        "source_rows": expected_rows, "excluded_rows": excluded_rows,
        "exclusion_reason": "Existing diabetes excluded from prediabetes cohort" if excluded_rows else "None",
        "rows": len(cleaned), "columns": len(cleaned.columns),
        "derivation": "User-uploaded three-class BRFSS2015 file; explicit endpoint mapping and recorded exclusions",
        "is_official_binary_file": False,
    }
    write_json(sidecar, metadata)
    return metadata

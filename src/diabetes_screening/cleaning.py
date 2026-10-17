"""Conservative cleaning that preserves survey responses and class balance."""
import pandas as pd

from .ingestion import DataQualityError, audit
from .schema import COLUMNS


def clean_data(frame, duplicate_policy="keep"):
    if duplicate_policy not in {"keep", "drop"}:
        raise ValueError("duplicate_policy must be 'keep' or 'drop'")
    report = audit(frame)
    if report["invalid_values"]:
        raise DataQualityError(f"Invalid or implausible codes require review: {report['invalid_values']}")
    if any(report["missing"].values()):
        raise DataQualityError("Missing values require an explicit analysis decision; no automatic imputation")
    if frame.empty:
        raise DataQualityError("The analysis table cannot be empty")
    cleaned = frame[COLUMNS].astype("int16").copy()
    if duplicate_policy == "drop":
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    flags = {
        "HighChol=1 but CholCheck=0 (told cholesterol high without a check in five years)":
            int(((cleaned["HighChol"] == 1) & (cleaned["CholCheck"] == 0)).sum()),
        "BMI below 15": int((cleaned["BMI"] < 15).sum()),
        "BMI 50 or above": int((cleaned["BMI"] >= 50).sum()),
        "BMI 60 or above": int((cleaned["BMI"] >= 60).sum()),
        "MentHlth == 30 (heaping at the top code)": int((cleaned["MentHlth"] == 30).sum()),
        "PhysHlth == 30 (heaping at the top code)": int((cleaned["PhysHlth"] == 30).sum()),
    }
    decisions = {
        "input_rows": len(frame), "output_rows": len(cleaned),
        "duplicate_policy": duplicate_policy, "duplicate_rows": report["duplicate_rows"],
        "missingness_policy": "Fail on missing values; never silently drop or impute respondents",
        "invalid_values_policy": "Fail on noninteger, nonfinite, out-of-codebook values",
        "duplicate_rationale": "Identical profiles may be different respondents; no respondent IDs are available",
        "transformations": "Preserve original codes and BMI/health days; no clipping, scaling, or balancing",
        "review_flags": flags,
        "review_flags_policy": "Counts of valid but unusual or logically odd responses; reported only, no row is altered or removed",
        "distribution_summary": cleaned[["BMI", "MentHlth", "PhysHlth"]].describe(
            percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).to_dict(),
    }
    return cleaned, decisions

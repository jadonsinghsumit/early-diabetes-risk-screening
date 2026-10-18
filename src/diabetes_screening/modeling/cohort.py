"""Load a prepared cohort for modeling, running the same provenance checks as stages 1-4.

AI-use disclosure: drafted with AI assistance (Claude, Anthropic); see docs/splitting.md.
"""
from dataclasses import dataclass

import pandas as pd

from ..cleaning import clean_data
from ..ingestion import load_source
from ..schema import COLUMNS, EXPECTED_ROWS, FIELDS, TARGET
from .provenance import read_provenance, verify_provenance

FEATURES = COLUMNS[1:]
BINARY = [f.name for f in FIELDS if f.kind == "binary" and f.name != TARGET]
ORDINAL = [f.name for f in FIELDS if f.kind == "ordinal"]
HEALTH_DAYS = ["MentHlth", "PhysHlth"]

# Answers that partly measure contact with the health system rather than biology
# (see the ascertainment caution in docs/statistics.md). Step 6 fits models with and without them.
HEALTHCARE_CONTACT = ["CholCheck", "HighBP", "HighChol", "AnyHealthcare", "NoDocbcCost"]
FEATURE_SETS = {
    "full": FEATURES,
    "no-healthcare-contact": [name for name in FEATURES if name not in HEALTHCARE_CONTACT],
}


@dataclass(frozen=True)
class Cohort:
    """Features, outcome and identifiers of one prepared endpoint cohort (row order = prepared CSV)."""
    X: pd.DataFrame
    y: pd.Series
    profile_id: pd.Series
    endpoint: str
    input_sha256: str
    provenance: dict


def load_cohort(path, provenance, expected_source_rows=EXPECTED_ROWS):
    """Read a CSV written by `prepare`, verify it against its sidecar and return a Cohort.

    `profile_id` numbers each distinct combination of the 21 feature values. Respondents with
    identical answers share an id, which lets the split be checked for profile overlap.
    """
    metadata = read_provenance(provenance)
    frame, quality = load_source(path, expected_rows=metadata.get("rows", -1))
    endpoint = verify_provenance(metadata, frame, quality["source"]["sha256"], expected_source_rows)
    cleaned, _ = clean_data(frame)  # refuses missing or out-of-codebook values
    X = cleaned[FEATURES]
    profile_id = X.groupby(FEATURES, sort=True).ngroup().rename("profile_id")
    return Cohort(X=X, y=cleaned[TARGET].rename(TARGET), profile_id=profile_id, endpoint=endpoint,
                  input_sha256=quality["source"]["sha256"], provenance=metadata)


def profile_summary(cohort):
    """Describe repeated answer profiles: are they copies of one person or different people?"""
    groups = pd.DataFrame({"profile": cohort.profile_id, "y": cohort.y}).groupby("profile")["y"]
    size, positives = groups.size(), groups.sum()
    repeated = size > 1
    mixed = repeated & (positives > 0) & (positives < size)
    rows_repeated = int(size[repeated].sum())
    return {
        "rows": int(size.sum()),
        "distinct_profiles": int(len(size)),
        "profiles_seen_more_than_once": int(repeated.sum()),
        "rows_in_repeated_profiles": rows_repeated,
        "share_of_rows_in_repeated_profiles": rows_repeated / int(size.sum()),
        "repeated_profiles_with_both_outcomes": int(mixed.sum()),
        "rows_in_profiles_with_both_outcomes": int(size[mixed].sum()),
        "largest_profile_size": int(size.max()),
        "positive_rate_in_repeated_profiles": float(positives[repeated].sum() / max(rows_repeated, 1)),
        "positive_rate_in_unique_profiles": float(positives[~repeated].sum() / max(int((~repeated).sum()), 1)),
    }

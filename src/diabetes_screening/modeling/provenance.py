"""Check a prepared CSV against the provenance file that `prepare` wrote next to it.

This repeats the checks Sumit's stage 1-4 pipeline runs in `pipeline.run_analysis`, so the
modeling code refuses exactly the same tampered or mismatched inputs. We copied the logic here
instead of editing the stage 1-4 code, which stays as Sumit wrote it.

AI-use disclosure: drafted with AI assistance (Claude, Anthropic); see docs/splitting.md.
"""
import json
from pathlib import Path

from ..endpoints import ENDPOINTS
from ..schema import EXPECTED_ROWS, TARGET


def read_provenance(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_provenance(metadata, frame, input_sha256, expected_source_rows=EXPECTED_ROWS):
    """Raise ValueError unless the hash, endpoint mapping and class counts all agree. Returns the endpoint."""
    if metadata.get("derived_sha256") != input_sha256:
        raise ValueError("Provenance hash does not match the input CSV")
    endpoint = metadata.get("endpoint")
    if endpoint not in ENDPOINTS:
        raise ValueError("Unknown binary endpoint in provenance")
    original = metadata.get("original_target_counts", {})
    mapping = {str(k): v for k, v in ENDPOINTS[endpoint].items()}
    # How many 0s and 1s the endpoint mapping should leave after recoding the three-class source
    retained = {str(label): sum(original.get(code, 0) for code, value in mapping.items() if value == label)
                for label in [0, 1]}
    if (sum(original.values()) != expected_source_rows
            or metadata.get("label_mapping") != mapping
            or sum(retained.values()) != len(frame)):
        raise ValueError("Provenance source counts, cohort size, or endpoint mapping is inconsistent")
    actual = {str(int(k)): int(v) for k, v in frame[TARGET].value_counts().items()}
    if any(actual.get(label, 0) != count for label, count in retained.items()):
        raise ValueError("Input class counts do not match endpoint provenance")
    return endpoint

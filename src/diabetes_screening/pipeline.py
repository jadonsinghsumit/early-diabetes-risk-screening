"""Run the four roadmap stages into a new, self-contained output directory."""
from importlib.metadata import version
from pathlib import Path
import platform
import json

import pandas as pd

from .cleaning import clean_data
from .endpoints import ENDPOINTS, DESCRIPTIONS
from .exploration import explore
from .ingestion import load_source, write_json
from .schema import EXPECTED_ROWS, data_dictionary
from .statistics import hypothesis_table


def run_analysis(source, destination, duplicate_policy="keep", expected_rows=EXPECTED_ROWS, synthetic=False, provenance=None):
    """Validate full-source size; provenance defines any eligible subcohort.

    Override expected_rows only for explicitly labelled synthetic validation.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    source_metadata = (json.loads(Path(provenance).read_text(encoding="utf-8"))
                       if provenance is not None else None)
    cohort_rows = source_metadata.get("rows", expected_rows) if source_metadata else expected_rows
    frame, quality = load_source(source, expected_rows=cohort_rows)
    if source_metadata is not None:
        if source_metadata.get("derived_sha256") != quality["source"]["sha256"]:
            raise ValueError("Provenance hash does not match the input CSV")
        endpoint = source_metadata.get("endpoint")
        if endpoint not in ENDPOINTS:
            raise ValueError("Unknown binary endpoint in provenance")
        original = source_metadata.get("original_target_counts", {})
        mapping = {str(k): v for k, v in ENDPOINTS[endpoint].items()}
        retained = {str(label): sum(original.get(code, 0) for code, value in mapping.items() if value == label)
                    for label in [0, 1]}
        if (sum(original.values()) != expected_rows
                or source_metadata.get("label_mapping") != mapping
                or sum(retained.values()) != cohort_rows):
            raise ValueError("Provenance source counts, cohort size, or endpoint mapping is inconsistent")
        actual = {str(int(k)): int(v) for k, v in frame["Diabetes_binary"].value_counts().items()}
        if any(actual.get(label, 0) != count for label, count in retained.items()):
            raise ValueError("Input class counts do not match endpoint provenance")
        write_json(destination / "source_provenance.json", source_metadata)
    write_json(destination / "quality.json", quality)
    pd.DataFrame(data_dictionary()).to_csv(destination / "data_dictionary.csv", index=False)
    cleaned, decisions = clean_data(frame, duplicate_policy)
    cleaned.to_csv(destination / "cleaned.csv", index=False)
    write_json(destination / "cleaning.json", decisions)
    endpoint = source_metadata["endpoint"] if source_metadata else "combined"
    dictionary = pd.DataFrame(data_dictionary())
    dictionary.loc[dictionary["name"] == "Diabetes_binary", "description"] = DESCRIPTIONS[endpoint]
    dictionary.to_csv(destination / "data_dictionary.csv", index=False)
    explore(cleaned, destination / "eda", endpoint=endpoint)
    hypotheses = hypothesis_table(cleaned)
    hypotheses.to_csv(destination / "hypotheses.csv", index=False)
    tested = int(hypotheses["status"].eq("tested").sum())
    manifest = {
        "source": quality["source"], "expected_rows": cohort_rows, "expected_source_rows": expected_rows,
        "validation_mode": "synthetic/test only" if synthetic or expected_rows != EXPECTED_ROWS else "full-shape source (provenance requires review)",
        "endpoint": endpoint, "source_provenance": source_metadata,
        "duplicate_policy": duplicate_policy, "analysis_rows": len(cleaned),
        "tests_executed": tested, "tests_skipped": len(hypotheses) - tested,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ["pandas", "numpy", "scipy", "statsmodels", "matplotlib", "seaborn"]},
        "limitations": "Self-report, unweighted derived cohort, no causal or clinical claims; see docs/statistics.md",
    }
    write_json(destination / "manifest.json", manifest)
    (destination / "README.md").write_text(
        "# Analysis run\n\n"
        f"Input SHA-256: `{quality['source']['sha256']}`.\n\n"
        f"Endpoint: {endpoint}. Validation mode: {manifest['validation_mode']}. Rows analyzed: {len(cleaned):,}. "
        f"Statistical tests executed: {tested}; skipped: {len(hypotheses) - tested}.\n\n"
        "Review quality.json and cleaning.json before interpreting eda/observations.md and hypotheses.csv. "
        "Blank inferential fields are explained in status, notes, or ci_method. "
        "A manifest is written only after all four stages finish.\n", encoding="utf-8")
    return manifest

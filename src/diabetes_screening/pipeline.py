"""Run the four roadmap stages into a new, self-contained output directory."""
from importlib.metadata import version
from pathlib import Path
import platform

import pandas as pd

from .cleaning import clean_data
from .exploration import explore
from .ingestion import load_source, write_json
from .schema import EXPECTED_ROWS, data_dictionary
from .statistics import hypothesis_table


def run_analysis(source, destination, duplicate_policy="keep", expected_rows=EXPECTED_ROWS, synthetic=False):
    """Use expected_rows only for explicitly labelled synthetic validation."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    frame, quality = load_source(source, expected_rows=expected_rows)
    write_json(destination / "quality.json", quality)
    pd.DataFrame(data_dictionary()).to_csv(destination / "data_dictionary.csv", index=False)
    cleaned, decisions = clean_data(frame, duplicate_policy)
    cleaned.to_csv(destination / "cleaned.csv", index=False)
    write_json(destination / "cleaning.json", decisions)
    explore(cleaned, destination / "eda")
    hypotheses = hypothesis_table(cleaned)
    hypotheses.to_csv(destination / "hypotheses.csv", index=False)
    tested = int(hypotheses["status"].eq("tested").sum())
    manifest = {
        "source": quality["source"], "expected_rows": expected_rows,
        "validation_mode": "synthetic/test only" if synthetic or expected_rows != EXPECTED_ROWS else "full-shape source (provenance requires review)",
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
        f"Validation mode: {manifest['validation_mode']}. Rows analyzed: {len(cleaned):,}. "
        f"Statistical tests executed: {tested}; skipped: {len(hypotheses) - tested}.\n\n"
        "Review quality.json and cleaning.json before interpreting eda/observations.md and hypotheses.csv. "
        "Blank inferential fields are explained in status, notes, or ci_method. "
        "A manifest is written only after all four stages finish.\n", encoding="utf-8")
    return manifest

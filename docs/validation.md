# Verification record

Repository: `jadonsinghsumit/early-diabetes-risk-screening`, branch `sumit-dev-test`. Incremental roadmap commits and the destination's original overview are retained.

## Primary combined endpoint and secondary prediabetes-only analysis

The roadmap's target is diabetes or prediabetes, so the primary cohort maps original code 0 to negative and codes 1 and 2 to positive (`prepare --endpoint combined`). The earlier prediabetes-only focus is retained as a sensitivity analysis. Both are current self-reported status, not future-onset forecasting.

Actual uploaded-data runs completed all four stages for both endpoints:

| | Combined (primary) | Prediabetes-only (secondary) |
| --- | --- | --- |
| Respondents analysed | 253,680 | 218,334 (35,346 existing-diabetes respondents excluded) |
| Positives | 39,977 (15.76%) | 4,631 (2.12%) |
| Identical profiles retained | 23,968 | 23,650 |
| Missing or invalid values | 0 | 0 |
| Hypothesis tests | 25 executed, 0 skipped | 25 executed, 0 skipped |

Source and prepared hashes, original and derived class counts, explicit mapping and exclusions are recorded, and the analysis checks the prepared hash, mapping, cohort size and class counts against source provenance. Review [the combined report](../reports/uploaded-combined/README.md) and [the prediabetes-only report](../reports/prediabetes/README.md) with their `source_provenance.json`, `quality.json`, `cleaning.json`, `manifest.json`, hypothesis tables, EDA charts and observations. Aggregate reports are committed; raw and cleaned row-level tables remain ignored locally. Hashes identify the provided files but do not independently authenticate their upstream derivation.

## Implementation checks

Python 3.12.14 with `requirements.lock`: **39 unit/integration tests passed; 97.82% source statement coverage** against an enforced 80% minimum (latest run on Python 3.13 with current, unlocked library versions; the 34-test, 98.11% figure was the earlier Python 3.12 lock-file run). The suite exercises source contracts, archive selection, missing and invalid values, duplicate policy, known statistical results, tied ranks, multiple testing, skip handling, plots, CLI execution, all endpoint mappings, exclusion/retention of respondents, and provenance tampering.

Dependency checks and package builds also pass. Seaborn/Matplotlib emit upstream deprecation warnings; checks are not suppressed. Remote CI is configured but its result has not been claimed as verified here.

## Remaining limits

Both reports predate the latest code changes (plot labels, per-unit odds ratios, review flags) and must be regenerated with the README commands; their counts do not change.

Live UCI/Kaggle downloads still return HTTP 403. This prevents independent validation of the upstream binary release but does not block the three-class uploaded source analysis. The survey-year documentation discrepancy, self-report, unavailable survey weights/clusters, confounding, and lack of longitudinal outcomes remain limitations.

Stages 5–8 and newer survey harmonization remain outside the current implementation. Later modeling must avoid leakage (including identical profiles), handle class imbalance and use evaluation measures appropriate to screening; results cannot be presented as future-onset prediction without suitable data. The combined positives include 35,346 people with a current diagnosis, so the model partly learns who already has diabetes.

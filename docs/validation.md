# Verification record

Repository: `jadonsinghsumit/early-diabetes-risk-screening`, branch `sumit-dev-test`. Incremental roadmap commits and the destination's original overview are retained.

## Primary prediabetes endpoint

The project lead clarified that early screening should focus on prediabetes. The primary cohort retains original code 0 as negative and code 1 as positive, excluding code 2 (existing diabetes). This is current self-reported prediabetes classification, not future-onset forecasting.

The actual uploaded-data workflow completed all four stages:

- Full source: 253,680 respondents; 35,346 existing-diabetes respondents explicitly excluded.
- Eligible cohort: 218,334 respondents; 213,703 negatives and 4,631 positives (2.12%).
- Missing and invalid values: zero in the eligible cohort.
- Identical profiles: 23,650 retained under the documented default policy.
- Hypothesis tests: all 25 executed; zero skipped. Nine figures generated.
- Source and prepared hashes, original and derived class counts, explicit mapping, and exclusions are recorded. The analysis checks the prepared hash and mapping, cohort size and class counts against source provenance.

Review [the saved primary run](../reports/prediabetes/README.md), its `source_provenance.json`, `quality.json`, `cleaning.json`, `manifest.json`, hypothesis table, EDA charts and observations. Aggregate reports are committed; raw and cleaned row-level tables remain ignored locally. The README gives exact reproduction commands and output review pointers. Hashes identify the provided files but do not independently authenticate their upstream derivation.

## Implementation checks

Python 3.12.14 with `requirements.lock`: **34 unit/integration tests passed; 98.11% source statement coverage** against an enforced 80% minimum. The suite exercises source contracts, archive selection, missing and invalid values, duplicate policy, known statistical results, tied ranks, multiple testing, skip handling, plots, CLI execution, all endpoint mappings, exclusion/retention of respondents, and provenance tampering.

Dependency checks and package builds also pass. Seaborn/Matplotlib emit upstream deprecation warnings; checks are not suppressed. Remote CI is configured but its result has not been claimed as verified here.

## Historical runs and remaining limits

The earlier combined prediabetes/diabetes run (39,977 positives among 253,680 respondents) is preserved under `reports/uploaded-combined` as historical comparison. It is not the primary project endpoint. Earlier synthetic runs tested execution and scale; actual uploaded-data validation now supports the primary cohort workflow.

Live UCI/Kaggle downloads still return HTTP 403. This prevents independent validation of the upstream binary release but does not block the three-class uploaded source analysis. The survey-year documentation discrepancy, self-report, unavailable survey weights/clusters, confounding, and lack of longitudinal outcomes remain limitations.

Stages 5–8 and newer survey harmonization remain outside the current implementation. Later modeling must avoid leakage, account for rare positives and use evaluation measures appropriate to screening; results cannot be presented as future-onset prediction without suitable data.

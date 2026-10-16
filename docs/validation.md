# Verification record

The implementation was migrated to `jadonsinghsumit/early-diabetes-risk-screening`, branch `sumit-dev-test`, retaining incremental stage commits. The destination's pre-existing overview is preserved in `docs/original-project-overview.md`.

Validated on Python 3.12.14 using `requirements.lock`:

- 32 unit/integration tests passed; source statement coverage 98.31%, with an 80% enforced minimum.
- Tests cover archive selection, schema and invalid values, conservative cleaning, known odds ratios, tied-rank effects, BH correction and skipped tests, plots, CLI orchestration, three-class preparation, endpoint direction, and source-hash mismatch rejection.
- Actual uploaded data completed all four stages: 253,680 × 22, zero missing or invalid values, 23,968 repeated profiles retained, 25 statistical tests executed, zero skipped, nine figures generated.
- Combined endpoint: 213,703 negatives and 39,977 positives (15.76%). Original class counts: 0=213,703; 1=4,631; 2=35,346. Conversion is explicit and is not represented as the official binary dataset.
- Earlier full-shape synthetic scale validation also passed; it is superseded by the actual uploaded-data run for this endpoint.

Review [the saved run](../reports/uploaded-combined/README.md), its `source_provenance.json`, `quality.json`, `cleaning.json`, `manifest.json`, hypothesis table and EDA files. Raw/cleaned row-level CSVs remain ignored; only aggregate reports, hashes and charts are committed. Repeat the README preparation/run commands using fresh output names to compare results.

UCI and Kaggle live downloads still return HTTP 403 in this machine. That blocks independent upstream binary-file validation, not analysis of the provided three-class upload. Input hashing establishes file identity, not independent upstream authenticity. The year discrepancy in source documentation and survey-design limitations remain documented.

Seaborn/Matplotlib deprecation warnings remain upstream warnings; tests and checks are not suppressed. Remote CI has not been claimed as passed. Stages 5–8 and newer raw-survey harmonization remain outside scope.

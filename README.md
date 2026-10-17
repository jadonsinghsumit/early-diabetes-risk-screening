# Early Diabetes Risk Screening

This project asks whether self-reported health and lifestyle indicators can identify adults with **diabetes or prediabetes** (the roadmap's `Diabetes_binary` target), using the supplied BRFSS2015-derived survey table. It currently implements roadmap stages **1–4**: ingestion/QC, cleaning, exploration, and hypothesis tests.
Project lead: Sumit Singh Jadaun
Teammates: Priyamvadha and Amiya.

The uploaded `Diabetes_012` table has 253,680 rows. The **primary endpoint** is explicit:

| Original label | Meaning | Analysis decision |
| --- | --- | --- |
| 0 | No diabetes | Negative: 213,703 respondents |
| 1 | Prediabetes | Positive: 4,631 respondents |
| 2 | Existing diabetes | Positive: 35,346 respondents |

The cohort therefore has **253,680 respondents** with **39,977 positive labels (15.76%)**. The positive group mixes people with a current diagnosis (35,346) and people with prediabetes (4,631), so results describe *identifying people with or at risk of diabetes*, not early detection of undiagnosed disease. Survey labels are self-reported; this cross-sectional dataset does not establish who will develop diabetes later. A future-onset prediction claim needs longitudinal outcomes. A secondary **prediabetes-only** analysis (existing diabetes excluded, 218,334 respondents, 4,631 positives, 2.12%) is kept as a sensitivity analysis. This is a research prototype, not a diagnostic tool, and no predictive model has been trained in stages 1–4.

> **Check against the official file.** The roadmap quotes a prevalence of about 14%, which equals class 2 alone (35,346 / 253,680 = 13.93%), and the 218,334 negatives in the official binary release equal classes 0 and 1 together. If `Diabetes_binary.value_counts()` on the official binary CSV shows 35,346 positives, that file treats prediabetes as negative despite its documentation, and the combined endpoint above (derived from the three-class file) is a deliberately different definition. State this in the final report.

## Install

Python 3.10+ is supported (`pyproject.toml`). CI and the exact dependency snapshot (`requirements.lock`) target Python 3.12 on Linux; the code and tests were also run on Python 3.13 with current library releases.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pip check
```

For other supported Python versions, use `python -m pip install -e '.[dev]'` to resolve compatible dependencies. Use the existing cloud checkout directly; a separate worktree is unnecessary.

## Prepare the cohorts and run stages 1–4

Place the supplied three-class ZIP under `data/raw/`, then run the primary (combined) analysis:

```bash
diabetes-screening prepare \
  --input data/raw/diabetes_012_health_indicators_BRFSS2015.csv.zip \
  --output data/derived/combined_BRFSS2015.csv \
  --endpoint combined

diabetes-screening run \
  --input data/derived/combined_BRFSS2015.csv \
  --provenance data/derived/combined_BRFSS2015.provenance.json \
  --output outputs/combined
```

For the prediabetes-only sensitivity analysis use `--endpoint prediabetes` and the paths `data/derived/prediabetes_BRFSS2015.csv` and `outputs/prediabetes`.

Choose **new paths** when repeating preparation or analysis; existing outputs are never overwritten. The source must contain all 253,680 rows and the exact three-class schema. The provenance records source/archive-member hashes, source counts, exclusions, mapping and the derived input hash. Analysis verifies the derived hash, mapping, cohort size and label counts. It accepts a smaller cohort (the prediabetes-only analysis) through this provenance, without using a synthetic row-count bypass. The legacy internal column name `Diabetes_binary` stores the prepared 0/1 outcome; the report's endpoint and dictionary define its meaning.

Raw and cleaned row-level files stay ignored locally. The checked-in aggregate reports are the [primary combined report](reports/uploaded-combined/README.md) and the [prediabetes-only sensitivity report](reports/prediabetes/README.md); they contain aggregate data, charts, provenance and methods. No credentials or background services are required for a local run. A fresh Git clone needs the uploaded source file; Git contains only aggregate reports.

## Review the outputs

| Stage | Outputs | Verification |
| --- | --- | --- |
| 1 | `quality.json`, `data_dictionary.csv`, `source_provenance.json` | Full source 253,680 → analysis cohort (253,680 combined; 218,334 prediabetes-only); source hashes, labels and feature audit |
| 2 | `cleaned.csv`, `cleaning.json` | Eligible rows retained; duplicates audited; no hidden imputation, scaling, clipping or balancing |
| 3 | `eda/*.png`, summary CSVs, `eda/observations.md` | Observed positive fraction (15.76% combined; 2.12% prediabetes-only), distributions and groups (plots use class names), categorical factors, Pearson/Spearman matrices, top correlations and prevalence ratios in the written observations |
| 4 | `hypotheses.csv` | 25 planned tests; H0/H1, status, raw/BH p-values, effects, intervals, sparse-cell notes, and per-unit odds ratios for numeric/ordinal predictors |
| Run | `manifest.json`, run `README.md` | Endpoint (`combined` or `prediabetes`); cohort/source counts, dependency versions and execution/skip counts |

The success manifest is written after all four stages finish. Errors return a nonzero exit status. Missing or invalid included values stop cleaning; no automatic row deletion or imputation occurs. Identical profiles are retained because no respondent identifier establishes accidental duplication. `--duplicates drop` provides a separate sensitivity run and must use a fresh output directory.

Read the [sources](docs/sources.md), [cleaning decisions](docs/cleaning.md), [exploration notes](docs/exploration.md), and [statistical methods](docs/statistics.md). Tests are exploratory, unadjusted for confounding and unavailable survey design; associations do not establish causation or clinical performance. Odds ratios concern this prediabetes endpoint. Mann–Whitney effects have approximate placement intervals; multilevel Cramer's V has no estimated CI and is labelled accordingly. Tiny p-values can underflow to zero.

## Verify the code

```bash
python -m pytest
python -m build
```

**39 tests passed with 97.82% source statement coverage** in the latest validation (Python 3.13, unlocked current library versions). Pytest enforces at least 80% and writes `coverage.xml`. Unit and integration tests include explicit exclusion of existing diabetes, retention of controls and prediabetes cases, provenance checks, known statistical calculations, missing/invalid data, chart generation and CLI behavior. CI repeats checks on Python 3.12; a remote CI result is not claimed here. See [validation details](docs/validation.md).

## Other endpoints and data sources

`prepare --endpoint prediabetes` (existing diabetes excluded) is the secondary sensitivity analysis, and `--endpoint diabetes-only` (prediabetes counted as negative) is a further comparison. The [original project overview](docs/original-project-overview.md) is archived context; current methods and measured counts supersede its assumptions.

The UCI `download` command remains available for the upstream binary CSV, but an already collapsed binary file cannot separate prediabetes from existing diabetes. Use the three-class upload for this project. UCI/Kaggle live requests may be blocked on some machines; this does not block analysis of the uploaded file. Newer raw BRFSS years require separate derivation and codebook validation; NHANES lab-based outcomes are a distinct target and cannot be silently mixed with self-report.

> **Regenerate the checked-in reports after pulling.** `reports/uploaded-combined/` and `reports/prediabetes/` were produced by the earlier code. The plots, `observations.md`, `hypotheses.csv` (new per-unit odds-ratio columns) and `cleaning.json` (new `review_flags`) change with the latest code, so rerun the commands above on fresh output directories and replace the reports with the new aggregate files. The numbers already in the reports do not change.

Stages 5–8 remain future work. Later model evaluation must address class imbalance (15.76% positives), leakage through identical profiles, self-report and healthcare-access bias, and the difference between identifying current diabetes or prediabetes and forecasting future onset.

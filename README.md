# Prediabetes Risk Screening

This project focuses on identifying **self-reported prediabetes among adults without existing diabetes**, using health and lifestyle indicators from the supplied BRFSS2015-derived survey table. It currently implements roadmap stages **1–4**: ingestion/QC, cleaning, exploration, and hypothesis tests. 
Project lead: Sumit Singh Jadaun
Teammates: Priyamvadha and Amiya.

The uploaded `Diabetes_012` table has 253,680 rows. The primary endpoint is explicit:

| Original label | Meaning | Analysis decision |
| --- | --- | --- |
| 0 | No diabetes | Retain as negative: 213,703 respondents |
| 1 | Prediabetes | Retain as positive: 4,631 respondents |
| 2 | Existing diabetes | Exclude: 35,346 respondents |

The resulting cohort has **218,334 respondents**, with **2.12% positive labels**. Existing diabetes is not treated as a healthy negative or merged into the positive group. Survey labels are self-reported; this cross-sectional dataset does not establish who will develop diabetes later. A future-onset prediction claim needs longitudinal outcomes. This is a research prototype, not a diagnostic tool, and no predictive model has been trained in stages 1–4.

## Install

Python 3.12+. The package supports Python 3.12+; the exact dependency snapshot targets Python 3.12 on Linux.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pip check
```

For other supported Python versions, use `python -m pip install -e '.[dev]'` to resolve compatible dependencies. Use the existing cloud checkout directly; a separate worktree is unnecessary.

## Prepare the primary cohort and run stages 1–4

Place the supplied three-class ZIP under `data/raw/`, then run:

```bash
diabetes-screening prepare \
  --input data/raw/diabetes_012_health_indicators_BRFSS2015.csv.zip \
  --output data/derived/prediabetes_BRFSS2015.csv \
  --endpoint prediabetes

diabetes-screening run \
  --input data/derived/prediabetes_BRFSS2015.csv \
  --provenance data/derived/prediabetes_BRFSS2015.provenance.json \
  --output outputs/prediabetes
```

Choose **new paths** when repeating preparation or analysis; existing outputs are never overwritten. The source must contain all 253,680 rows and the exact three-class schema. The provenance records source/archive-member hashes, source counts, exclusions, mapping and the derived input hash. Analysis verifies the derived hash, mapping, cohort size and label counts. It accepts the smaller cohort through this provenance, without using a synthetic row-count bypass. The legacy internal column name `Diabetes_binary` stores the prepared 0/1 outcome; the report's endpoint and dictionary define its meaning.

Raw and cleaned row-level files stay ignored locally. The [checked-in primary report](reports/prediabetes/README.md) contains aggregate data, charts, provenance and methods. No credentials or background services are required for a local run. A fresh Git clone needs the uploaded source file; Git contains only aggregate reports.

## Review the outputs

| Stage | Outputs | Verification |
| --- | --- | --- |
| 1 | `quality.json`, `data_dictionary.csv`, `source_provenance.json` | Full source 253,680 → eligible cohort 218,334; source hashes, labels and feature audit |
| 2 | `cleaned.csv`, `cleaning.json` | Eligible rows retained; duplicates audited; no hidden imputation, scaling, clipping or balancing |
| 3 | `eda/*.png`, summary CSVs, `eda/observations.md` | Observed 2.12% positive fraction, distributions and groups, categorical factors, Pearson/Spearman matrices |
| 4 | `hypotheses.csv` | 25 planned tests; H0/H1, status, raw/BH p-values, effects, intervals and sparse-cell notes |
| Run | `manifest.json`, run `README.md` | Endpoint is `prediabetes`; cohort/source counts, dependency versions and execution/skip counts |

The success manifest is written after all four stages finish. Errors return a nonzero exit status. Missing or invalid included values stop cleaning; no automatic row deletion or imputation occurs. Identical profiles are retained because no respondent identifier establishes accidental duplication. `--duplicates drop` provides a separate sensitivity run and must use a fresh output directory.

Read the [sources](docs/sources.md), [cleaning decisions](docs/cleaning.md), [exploration notes](docs/exploration.md), and [statistical methods](docs/statistics.md). Tests are exploratory, unadjusted for confounding and unavailable survey design; associations do not establish causation or clinical performance. Odds ratios concern this prediabetes endpoint. Mann–Whitney effects have approximate placement intervals; multilevel Cramer's V has no estimated CI and is labelled accordingly. Tiny p-values can underflow to zero.

## Verify the code

```bash
python -m pytest
python -m build
```

**34 tests passed with 98.11% source statement coverage** in the current validation. Pytest enforces at least 80% and writes `coverage.xml`. Unit and integration tests include explicit exclusion of existing diabetes, retention of controls and prediabetes cases, provenance checks, known statistical calculations, missing/invalid data, chart generation and CLI behavior. CI repeats checks on Python 3.12; a remote CI result is not claimed here. See [validation details](docs/validation.md).

## Other endpoints and data sources

`prepare --endpoint combined` and `--endpoint diabetes-only` remain explicit alternatives for comparison, not the primary objective. The [earlier combined-endpoint report](reports/uploaded-combined/README.md) is historical and must not be used as the primary findings. The [original project overview](docs/original-project-overview.md) is archived context; current methods and measured counts supersede its assumptions.

The UCI `download` command remains available for the upstream binary CSV, but an already collapsed binary file cannot isolate prediabetes from existing diabetes. Use the three-class upload for this project objective. UCI/Kaggle live requests remain blocked in this cloud machine; this does not block analysis of the uploaded file. Newer raw BRFSS years require separate derivation and codebook validation; NHANES lab-based outcomes are a distinct target and cannot be silently mixed with self-report.

Stages 5–8 remain future work. Later model evaluation must address rare positives, leakage, self-report bias and the difference between identifying current prediabetes and forecasting future onset.

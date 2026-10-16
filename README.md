# Early Diabetes Risk Screening

A reproducible research workflow for the **first four roadmap stages**: ingestion and quality control, conservative cleaning, exploratory analysis, and association tests. Project lead: Sumit Singh Jadaun; teammates: Priyamvadha and Amiya.

Use the full `diabetes_binary_health_indicators_BRFSS2015.csv`: **253,680 rows × 22 columns**. Positive labels combine self-reported prediabetes and diabetes. This is a research analysis, not a diagnosis tool. No predictive models, train/test splits, threshold selection, or clinical performance claims are included yet.

## Install

Python 3.12 is the tested environment. The package supports Python 3.10+; the exact dependency snapshot targets Python 3.12 on Linux.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pip check
```

For other supported Python versions, use `python -m pip install -e '.[dev]'` so pip can resolve compatible versions. The existing cloud checkout can be used directly; no separate worktree is needed.

## Obtain the data

```bash
diabetes-screening download
```

This downloads the public UCI archive over verified HTTPS and selects only the exact binary filename. It refuses to overwrite an existing file. Network access to `archive.ics.uci.edu` is needed. Alternatively, download the source from [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) and place the full binary CSV under `data/raw/`. Do not use the balanced or three-level file. Raw data is intentionally excluded from Git.

See [source provenance and the survey-year discrepancy](docs/sources.md). Shape checks identify the expected variant but do not authenticate its derivation; review the source and retain the SHA-256 in the run report.

## Run stages 1–4

```bash
diabetes-screening run \
  --input data/raw/diabetes_binary_health_indicators_BRFSS2015.csv \
  --output outputs/baseline
```

Every run needs a **new output directory**, preventing accidental overwrites or stale success markers. A failure returns a nonzero exit code. The success manifest is written only after all four stages finish. No credentials or background services are required to analyze a local CSV.

| Stage | Outputs | What to verify |
| --- | --- | --- |
| 1 | `quality.json`, `data_dictionary.csv` | 253,680 × 22 input, exact columns, dtype/range/label audit, input SHA-256 |
| 2 | `cleaned.csv`, `cleaning.json` | Default row count preserved, duplicates audited, original codes and class balance retained |
| 3 | `eda/*.png`, descriptive CSVs, `eda/observations.md` | Actual positive fraction, group summaries, distributions, categorical factors, Pearson/Spearman matrices |
| 4 | `hypotheses.csv` | 25 planned tests, explicit skips, H0/H1, p-values, BH correction, effects, intervals and limitations |
| Run | `manifest.json`, output `README.md` | Source hash, dependency versions, analysis mode, actual test/skip counts |

Duplicates are retained because matching profiles may represent different respondents. Missing or invalid values stop cleaning rather than triggering hidden imputation or row deletion. To run a separate duplicate sensitivity analysis, choose another output directory and add `--duplicates drop`. Read [cleaning decisions](docs/cleaning.md), [exploration notes](docs/exploration.md), and [statistical methods](docs/statistics.md) before interpreting results.

Binary predictors receive odds ratios and 95% intervals. Multilevel predictors receive omnibus Cramer's V (CI not estimated, explicitly labelled). Selected numeric/ordinal comparisons receive rank-biserial effects and approximate 95% intervals. Sparse chi-square cells are flagged. Tests are unadjusted for confounding and survey design; BH-adjusted p-values do not make the associations causal.

## Verify the implementation

```bash
python -m pytest
python -m build
```

Tests cover source contracts, invalid and missing values, duplicates, safe archive selection, known odds-ratio calculations, tied ranks, correction and skip behavior, chart/table generation, and the CLI end to end. Pytest enforces **at least 80% source statement coverage** and writes `coverage.xml`. CI repeats tests, dependency checks, and packaging on Python 3.12. Network access and real source data are not needed for the synthetic test suite.

For a small explicitly synthetic CSV with the same columns, `--synthetic-rows N` bypasses the production row count and labels all outputs synthetic. **Never use synthetic results as project findings.** Normal runs require the full source shape.

Current verification and the remaining real-data check are recorded in [validation notes](docs/validation.md). Generated figures, raw/cleaned CSVs, coverage reports, and build outputs stay outside Git. Stages 5–8 remain future work; full-cohort exploration must not later be used to tune a held-out evaluation set.

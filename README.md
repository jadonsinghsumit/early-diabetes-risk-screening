# Early Diabetes Risk Screening

This project asks whether self-reported health and lifestyle indicators can identify adults with **diabetes or prediabetes** (the roadmap's `Diabetes_binary` target), using the supplied BRFSS2015-derived survey table. It currently implements roadmap stages **1–4**: ingestion/QC, cleaning, exploration, and hypothesis tests.    

Project lead: Sumit Singh Jadaun   
Teammates: Priyamvadha and Amiya.

> **Update: stages 5 and 6 are now done.** They were added in new files only (`src/diabetes_screening/modeling/`, two new test files, `docs/splitting.md`, `docs/training.md`, `docs/modeling-validation.md` and `reports/modeling/`) and have their own command, `diabetes-screening-model`. The stage 1–4 code, reports and documentation, including the rest of this README, are unchanged. So where the text below says stages 5–8 are future work, read that as stages 7–8. See [Stages 5 and 6: modeling](#stages-5-and-6-modeling) at the end.

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

Stages 5–8 remain future work; see the [stages 5–8 plan](docs/stages-5-8-plan.md). Later model evaluation must address class imbalance (15.76% positives), leakage through identical profiles, self-report and healthcare-access bias, and the difference between identifying current diabetes or prediabetes and forecasting future onset.

## Stages 5 and 6: modeling

*This section was added for stages 5 and 6. Everything above it is unchanged from stages 1–4.*

Stage 5 splits the data and settles how to handle the class imbalance. Stage 6 tunes and compares three models, checks them against a paper risk checklist that clinics use, and looks at which answers drive them. The full write-ups are [splitting](docs/splitting.md) and [training](docs/training.md); the checks we ran are in [modeling validation](docs/modeling-validation.md).

### Setup

The install steps above still work. `requirements.lock` and `pyproject.toml` now also list scikit-learn and XGBoost, so in an existing environment run `python -m pip install -r requirements.lock` and `python -m pip install --no-deps -e .` again. On Linux, XGBoost's wheel also pulls in NVIDIA's NCCL library, which is a large download and isn't used on CPU-only machines.

For the data, the Kaggle download works directly: get `archive.zip` from the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) and pass it to `prepare`, which picks the three-class CSV out of it. We checked that this CSV is byte-for-byte the file used in stages 1–4 (SHA-256 `a971809d…`), and re-running `prepare` reproduces both prepared files exactly.

### Run it

```bash
# Prepare the cohort with the stage 1-4 command, exactly as above
diabetes-screening prepare --input data/raw/archive.zip \
  --output data/derived/combined_BRFSS2015.csv --endpoint combined

# Stage 6 on the committed split (the split every teammate should share)
diabetes-screening-model train \
  --input data/derived/combined_BRFSS2015.csv \
  --provenance data/derived/combined_BRFSS2015.provenance.json \
  --split reports/modeling/combined/split \
  --output outputs/combined/training

# Optional: rebuild stage 5 from scratch (it gives byte-identical split files)
diabetes-screening-model split --input data/derived/combined_BRFSS2015.csv \
  --provenance data/derived/combined_BRFSS2015.provenance.json --output outputs/combined/split
diabetes-screening-model imbalance --input data/derived/combined_BRFSS2015.csv \
  --provenance data/derived/combined_BRFSS2015.provenance.json \
  --split outputs/combined/split --output outputs/combined/imbalance
```

For the prediabetes-only sensitivity analysis, swap `combined` for `prediabetes` everywhere. On a two-core laptop, `split` takes seconds, `imbalance` a few minutes and `train` about 20 to 25 minutes (mostly the random forest search; `--models logistic_regression,xgboost` skips it). As with stages 1–4, every command checks its input against the provenance file and refuses to write into an existing folder.

### What's in `reports/modeling/`

| Folder (per endpoint) | What it holds |
| --- | --- |
| `split/` | `split.json` (seed, input hash, counts per side and fold), `split_assignment.csv.gz` (which rows are test and which fold each training row is in), `profiles.json` (the repeated-answers audit) |
| `imbalance/` | Weighted vs unweighted models on the five folds, the repeated-profile check, a chart and a README |
| `training/` | Model comparison (`comparison.csv`, `paired_differences.csv`), every candidate tried (`tuning.csv`), the chosen model (`selection.json`), adjusted odds ratios, permutation importance, the reduced feature-set check, charts, a README and the fitted logistic regression and XGBoost pipelines (`models/`) |

None of these files contain survey answers. The fitted random forest (about 16 MB) and the out-of-fold predictions (one row per respondent, needed in stage 7) aren't committed; `train` writes both to your local `outputs/` folder.

### What we found

**The split (stage 5).** One stratified 80/20 split with seed 42, and five fixed folds inside the training part. For the combined endpoint that's 202,944 training rows (31,982 positive) and 50,736 locked test rows (7,995 positive). The test rows can only be read after stage 7 saves a threshold for this exact split, and so far no stage has used a single one of them.

**Two worries from the plan, both tested (stage 5).** About 15% of respondents share their exact answers with someone else. They're different people (over 1,800 shared profiles include both outcomes), and keeping them together in the folds changed PR-AUC by 0.001 at most, so a plain split is fine. Class weighting didn't help any model rank people better (PR-AUC within 0.001). It only pushed predicted risks up, from the true 15.8% to about 39%, which has the same effect as lowering the cutoff. So the models are unweighted, and stage 7 will choose the cutoff on purpose.

**The models (stage 6, combined endpoint, mean of five validation folds).**

| Model | PR-AUC | ROC-AUC | Share of people flagged to catch 80% of cases |
| --- | --- | --- | --- |
| No model (everyone gets the average risk) | 0.158 | 0.500 | 100% |
| Paper diabetes risk checklist (NIDDK/ADA) | 0.325 | 0.756 | 48% at its own 5-point cutoff |
| Logistic regression | 0.439 | 0.819 | 39% |
| Random forest | 0.456 | 0.822 | 39% |
| **XGBoost (chosen)** | **0.461** | **0.826** | **38%** |

XGBoost beat the random forest in all five folds (corrected t-test p = 0.005), so under the rule we wrote down before the run, it goes forward to stage 7. Compared with the paper checklist, it finds the same 80% of cases while raising roughly 95 fewer false alarms per 1,000 people screened. The step from the checklist to any model is large; the step from logistic regression to XGBoost is small (+0.022 PR-AUC).

**What drives the predictions.** In every model, the answers that matter most are BMI, general health, age, high blood pressure and high cholesterol. Dropping the five answers that partly reflect contact with doctors costs about 0.03 PR-AUC, but the models still beat the checklist easily. Adjusted odds ratios from statsmodels show that several single-answer associations from stage 4 shrink, vanish or even reverse once the other answers are held fixed. Physical-health days reverse, and smoking and vegetables lose their association, which is a useful warning about reading unadjusted comparisons. All variance inflation factors are below 2.

**Prediabetes only (sensitivity analysis).** The same ranking holds (XGBoost PR-AUC 0.060 against 0.021 for no model and 0.040 for the checklist), but the signal is weak: catching 80% of cases means flagging about 44% of people, and only about 4% of those flagged have prediabetes. Many people with prediabetes don't know they have it, so the labels themselves are noisy.

These are validation numbers on training rows. The held-out test set, opened once in stage 7, will give the final figures.

### Where we departed from the stages 5–8 plan

The plan in `docs/stages-5-8-plan.md` is left as written; the details of each change are in the two write-ups.

- We follow the README and use the combined endpoint as the main one. Section 0 of the plan predates that switch.
- We measured whether people with identical answers inflate the scores, found they don't, and used a plain stratified split instead of a grouped one.
- We left the classes unweighted (including XGBoost's `scale_pos_weight`), because weighting only moved the cutoff.
- Our checklist baseline is the published diabetes risk test rather than a hand-made rule.
- We added a separate `diabetes-screening-model` command rather than editing the stage 1–4 command.

### Tests

`python -m pytest` now runs 57 tests: the 39 stage 1–4 tests, unchanged, plus 18 new ones. All pass on Python 3.12 with `requirements.lock` (the CI setup), with 98.46% statement coverage. The new tests cover split stratification and reproducibility, the test-set lock, scalers fitted on training folds only, the checklist scoring, the screening metrics, the corrected t-test, the selection rule, the odds-ratio code and both new commands from start to finish.

### Use of AI tools

The stage 5 and 6 code, tests and write-ups were drafted with help from an AI assistant (Claude, by Anthropic), as noted in each new file. The team members responsible for these stages should review and run them, be ready to explain each step, and record their own review in the notes at the end of [splitting](docs/splitting.md) and [training](docs/training.md), as the course's generative-AI policy requires.

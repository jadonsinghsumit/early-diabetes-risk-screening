# Early Diabetes Risk Screening

Can self-reported health answers pick out adults who have diabetes or prediabetes, while keeping missed cases low? This project answers that with the CDC's 2015 BRFSS survey (253,680 adults, 21 answers each), in eight reproducible stages: data checks, cleaning, exploration, association tests, a leakage-safe split, model tuning, a locked screening threshold, and this handoff.

**Answer, on a held-out test set of 50,736 people:** a tuned XGBoost model, at a cutoff fixed in advance on training data, finds **79.4%** of people with diabetes or prediabetes (95% CI 78.5–80.3%) while flagging 37.6% of everyone. For the same number of cases found, it raises about **96 fewer false alarms per 1,000 people** than the paper risk checklist clinics use. It finds diagnosed diabetes (81%) much more readily than prediabetes (65%), and it misses more cases among adults under 45 and people with higher incomes.

This is a research prototype for risk stratification, **not a diagnostic or clinical decision tool**. The outcome is self-reported, the data are cross-sectional and unweighted, and the associations it reports are not causes.

**Project lead:** Sumit Singh Jadaun. **Team:** Priyamvadha and Amiya.

## Results at a glance

Locked test set, main endpoint (prediabetes or diabetes vs no diabetes, 15.8% positive). Intervals are 95% bootstrap intervals.

| Model | Recall | Precision | Share flagged | PR-AUC | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| No model (everyone gets the average risk) | – | 0.158 | – | 0.158 | 0.500 |
| Paper checklist (NIDDK/ADA, 5+ points) | 0.800 | 0.267 | 0.473 | 0.325 | 0.756 |
| Logistic regression (reference) | 0.794 | 0.324 | 0.386 | 0.442 | 0.819 |
| **XGBoost (selected, cutoff 0.153)** | **0.794** (0.785–0.803) | **0.333** (0.326–0.339) | **0.376** | **0.465** (0.454–0.476) | **0.826** (0.821–0.830) |

Per 1,000 people screened (about 158 with the condition), XGBoost flags 376, finds 125 cases, misses 33 and raises 251 false alarms; the checklist flags 473 to find 126. The predicted risks are well calibrated (calibration slope 0.99), so a stated risk of 30% means about 30% of such people have the condition.

The **prediabetes-only** sensitivity analysis (existing diabetes excluded, 2.1% positive) is much harder: at the same 80% target the model flags 44% of people and only 4% of those flagged report prediabetes (PR-AUC 0.062, ROC-AUC 0.759). Full results and the false-negative discussion are in [evaluation](docs/evaluation.md).

## The roadmap, stage by stage

| Stage | What it does | Write-up | Outputs | Notebook |
| --- | --- | --- | --- | --- |
| 1. Ingestion and quality control | Load the file, check shape, codes and hashes | [sources](docs/sources.md) | [reports/uploaded-combined](reports/uploaded-combined/README.md) | [01](notebooks/01_ingestion_quality.ipynb) |
| 2. Cleaning | Stop on bad values, keep every respondent, flag odd answers | [cleaning](docs/cleaning.md) | `cleaning.json`, `data_dictionary.csv` | [02](notebooks/02_cleaning.ipynb) |
| 3. Exploration | Class balance, distributions, correlations, observations | [exploration](docs/exploration.md) | `eda/` | [03](notebooks/03_exploration.ipynb) |
| 4. Hypotheses and statistics | 25 planned tests with effect sizes, intervals and BH correction | [statistics](docs/statistics.md) | `hypotheses.csv` | [04](notebooks/04_hypotheses.ipynb) |
| 5. Splitting and imbalance | Stratified 80/20 split, locked test set, weighting and leakage checks | [splitting](docs/splitting.md) | [reports/modeling/*/split, imbalance](reports/modeling/README.md) | [05](notebooks/05_splitting_imbalance.ipynb) |
| 6. Training and tuning | Three tuned models, paper-checklist baseline, odds ratios, importance | [training](docs/training.md) | `reports/modeling/*/training` | [06](notebooks/06_training.ipynb) |
| 7. Evaluation and threshold | Cutoff locked on training data, one test-set evaluation, subgroups, misses | [evaluation](docs/evaluation.md) | `reports/modeling/*/threshold`, `evaluation` | [07](notebooks/07_evaluation.ipynb) |
| 8. Reporting and handoff | This README, notebooks, verification record, technical report, presentation | [validation (1–4)](docs/validation.md), [validation (5–8)](docs/modeling-validation.md) | – | – |

## Data and endpoint

**Source.** The three-class file `diabetes_012_health_indicators_BRFSS2015.csv` from the [Kaggle Diabetes Health Indicators dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) (`archive.zip`, SHA-256 `ecbd6572…`; the CSV inside is `a971809d…`). It codes diabetes status as 0 (none, 213,703), 1 (prediabetes, 4,631) or 2 (diabetes, 35,346).

**Main endpoint: prediabetes or diabetes vs neither** (`prepare --endpoint combined`): 253,680 people, 39,977 positive (15.76%). This is the roadmap's `Diabetes_binary` target. The roadmap quotes "about 14%", which is diabetes alone (13.93%): Kaggle's separate binary file turns out to code prediabetes as negative ([details](docs/modeling-validation.md)). We count prediabetes as positive on purpose, because it is the state a screening tool should catch.

**Sensitivity analysis: prediabetes vs no diabetes** (`--endpoint prediabetes`): existing diabetes excluded, 218,334 people, 4,631 positive (2.12%).

**Survey year.** The roadmap and Kaggle describe the file as BRFSS 2015, while UCI's landing page mentions 2014; we keep that discrepancy visible rather than resolve it ([sources](docs/sources.md)). The table has no survey weights, strata or clusters, so rates describe this table, not the US population.

Raw and prepared row-level files are never committed (`data/` and `outputs/` are git-ignored). Everything under `reports/` is aggregate: counts, metrics, settings and charts.

## Setup

Python 3.10+ is supported. CI uses Python 3.12 on Linux with the exact versions in `requirements.lock`; the stage 7 results were produced on Windows with Python 3.13 and the same locked versions.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
python -m pip check
```

To open the notebooks, also run `python -m pip install -e ".[notebooks]"`. On Linux, XGBoost's wheel pulls in NVIDIA's NCCL library, a large download that CPU-only machines don't use.

Download `archive.zip` from the Kaggle page above into `data/raw/` (Kaggle needs a free account in the browser; the public API endpoint `https://www.kaggle.com/api/v1/datasets/download/alexteboul/diabetes-health-indicators-dataset` also works without one).

## Reproduce everything

Every command checks its input against the provenance file written by `prepare` and refuses to write into an existing folder, so pick new output paths when repeating a step. Times are for a two-core laptop.

```bash
# Stages 1-4
diabetes-screening prepare --input data/raw/archive.zip \
  --output data/derived/combined_BRFSS2015.csv --endpoint combined
diabetes-screening run --input data/derived/combined_BRFSS2015.csv \
  --provenance data/derived/combined_BRFSS2015.provenance.json --output outputs/stage1-4/combined

# Shorthand for the modeling commands
IN="--input data/derived/combined_BRFSS2015.csv --provenance data/derived/combined_BRFSS2015.provenance.json"

# Stage 5 (seconds, then a few minutes); reproduces the committed split byte for byte
diabetes-screening-model split $IN --output outputs/combined/split
diabetes-screening-model imbalance $IN --split reports/modeling/combined/split --output outputs/combined/imbalance

# Stage 6 (20-25 minutes, mostly the random forest search)
diabetes-screening-model train $IN --split reports/modeling/combined/split --output outputs/combined/training

# Stage 7: lock the cutoff on training rows, then score the test set once
diabetes-screening-model threshold $IN --split reports/modeling/combined/split \
  --training reports/modeling/combined/training --output outputs/combined/threshold
diabetes-screening-model test $IN --split reports/modeling/combined/split \
  --training reports/modeling/combined/training --threshold outputs/combined/threshold \
  --source data/raw/archive.zip --output outputs/combined/evaluation
```

For the sensitivity analysis, use `--endpoint prediabetes` in `prepare` and swap `combined` for `prediabetes` everywhere else. Stages 5–7 use the committed split in `reports/modeling/<endpoint>/split`, so every teammate works on the same rows; `test` only runs with a `threshold.json` that matches that split and the committed model file.

What reproduces exactly (checked on Windows against reports produced on Linux; see [validation](docs/modeling-validation.md)): the prepared files, every stage 1–4 table, the split files, `threshold.json` and every stage 7 table. Stage 6 retraining matches to within 0.0003 PR-AUC and 0.001 ROC-AUC and picks the same model; it isn't byte-identical because XGBoost's floating-point arithmetic differs slightly between operating systems.

## Methods in brief

- **Stages 1–2.** Explicit endpoint mapping with recorded hashes and counts; missing or out-of-codebook values stop the run instead of being imputed; the 23,968 rows that repeat another respondent's answers are kept, because there is no respondent ID to show they are copies.
- **Stages 3–4.** Descriptive comparisons, Pearson and Spearman correlations, 25 planned chi-square and Mann–Whitney tests with odds ratios, Cramér's V or rank-biserial effects, 95% intervals and Benjamini–Hochberg correction. All unadjusted and non-causal.
- **Stage 5.** One stratified 80/20 split (seed 42) with five fixed folds inside the training part; scaling happens inside the model pipelines, so it is learned from training folds only. Class weighting didn't improve ranking (it only moved the cutoff), and keeping identical answer profiles together changed PR-AUC by at most 0.001, so the models are unweighted on a plain split.
- **Stage 6.** Logistic regression, random forest and XGBoost tuned by randomized search on PR-AUC, compared with no model and the NIDDK/ADA paper checklist, with a selection rule written down before the run (XGBoost won in all five folds). Adjusted odds ratios from statsmodels show several stage 4 associations shrinking or reversing once other answers are held fixed. BMI, general health, age, high blood pressure and high cholesterol drive every model.
- **Stage 7.** Out-of-fold predictions on the training rows set the cutoff: the highest risk that still flags 80% of cases. It was written to `threshold.json` before the test set was read. Calibration was checked against a rule fixed in advance (no recalibration needed). The test set was then scored once, with 1,000-resample bootstrap intervals, workload per 1,000, calibration, subgroups by age, sex and income, and a breakdown of missed cases.

## Limitations

- **Current, self-reported status, not future risk.** The labels say who reported diabetes or prediabetes in 2015. Nothing here predicts who will develop it, and undiagnosed people sit among the negatives.
- **Diagnosed diabetes dominates the positives** (88%), so part of what the model learns is the footprint of an existing diagnosis. Prediabetes, the case a screening tool exists for, is caught less often (65%).
- **Healthcare-contact answers** (`CholCheck`, `HighBP`, `HighChol`, `AnyHealthcare`, `NoDocbcCost`) partly measure seeing a doctor. Without them the models lose about 0.03 PR-AUC but still beat the checklist.
- **One cutoff, unequal trade-offs.** Sensitivity is 44% for adults under 45 and 68% for incomes of $50k+, while over-65s face a 51% false-alarm rate.
- **Unweighted, single-year survey extract.** No population prevalence claims; performance on other years or on lab-confirmed outcomes (such as NHANES) is untested.
- **Associations are not causes**, including the adjusted odds ratios.

## Verify the code

```bash
python -m pytest     # 63 tests, coverage report, fails below 80%
python -m build
```

The suite covers source and provenance checks, cleaning rules, known statistical results, split stratification and reproducibility, the test-set lock, scalers fitted on training folds only, the checklist scoring, the corrected t-test, the calibration and bootstrap code, the threshold-then-test order, loading models pickled on another operating system, and every command end to end. CI runs it on Python 3.12 for every pull request. Verification records: [stages 1–4](docs/validation.md) and [stages 5–8](docs/modeling-validation.md).

## Sources

1. Kaggle, Diabetes Health Indicators Dataset: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
2. UCI Machine Learning Repository, CDC Diabetes Health Indicators: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
3. CDC, MMWR 2017 (66:43): https://www.cdc.gov/mmwr/volumes/66/wr/mm6643a2.htm
4. CDC / NCHS, NHANES: https://www.cdc.gov/nchs/nhanes/
5. CDC / NCHS, Vital and Health Statistics Series 11 No. 237 (historical report; check its date before citing it as current): https://www.cdc.gov/nchs/data/series/sr_11/sr11_237.pdf
6. NIDDK, Diabetes Risk Test (the checklist baseline): https://www.niddk.nih.gov/-/media/Files/Health-Information/Health-Professionals/Diabetes/am-i-at-risk/Diabetes-Risk-Test-PDF_508_1.pdf

The [original project overview](docs/original-project-overview.md) and the [stages 5–8 plan](docs/stages-5-8-plan.md) are kept as written for the record; where the work departed from the plan, each write-up says so and why.

## Use of AI tools

Stages 5–8 (code, tests, write-ups and notebooks) were drafted with help from an AI assistant (Claude, by Anthropic), as noted in each new file. The team members responsible for each stage should review and run their parts, be ready to explain every step, and record their own review in the notes at the end of [splitting](docs/splitting.md), [training](docs/training.md) and [evaluation](docs/evaluation.md), as the course's generative-AI policy requires.

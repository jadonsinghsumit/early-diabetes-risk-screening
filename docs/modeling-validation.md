# Checks we ran for stages 5 to 8

This page records how we checked the stage 5–8 work, in the same spirit as the stage 1–4 [validation record](validation.md).

## The data

- **Source.** The Kaggle download `archive.zip` ([Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), archive SHA-256 `ecbd6572…`). Its three-class CSV has SHA-256 `a971809d…`, the same as the source recorded in both stage 1–4 provenance files. For stages 7 and 8 the archive was fetched again from Kaggle's public download endpoint and had the same hash.
- **Preparation reproduces exactly.** Running the stage 1–4 `prepare` command on that zip gives prepared files whose hashes match the committed reports byte for byte: `7e0e0421…` (combined) and `33018d98…` (prediabetes-only). This now holds on Windows too, after the line-ending fix described under stage 8.
- **The official binary file.** A row-by-row comparison shows Kaggle's `diabetes_binary` file has 35,346 positives, all of them existing diabetes. All 4,631 prediabetes respondents are coded 0, and the features are identical to the three-class file. This answers the open question in the README: that file is the diabetes-only definition.

## Stage 5

| Check | Result |
| --- | --- |
| Split counts, combined | Train 202,944 (31,982 positive); test 50,736 (7,995 positive); folds of 40,588–40,589 rows with 6,396–6,397 positive |
| Split counts, prediabetes-only | Train 174,667 (3,705 positive); test 43,667 (926 positive); 741 positive in every fold |
| Reproducible | Separate runs produced byte-identical `split_assignment.csv.gz` files (hashes `e8a57949…` and `76326823…`) and identical fold metrics. Rebuilt on Windows in stage 8: `split_assignment.csv.gz`, `split.json` and `profiles.json` identical for both endpoints |
| Test rows untouched | Every stage 5 and 6 manifest records `"test_rows_used": 0`; `locked_test_data()` refuses without a matching `threshold.json` |
| Repeated profiles | Grouped and ordinary folds differ by 0.001 PR-AUC or less, for all three models and both endpoints |

## Stage 6

| Check | Result |
| --- | --- |
| Search is reproducible | Refitting each winning setting fold by fold gives exactly the search's own score (see "refit check" in each training README). An interrupted earlier run produced the same tuned logistic regression score (0.4391) |
| Selection rule fixed in advance | The rule and the corrected t-test were written into the code before the full run; the test `test_selection_rule_prefers_simple_or_calibrated_models_when_gaps_are_noise` exercises all three branches on made-up scores |
| Corrected t-test | Checked against a hand calculation (`test_corrected_t_test_matches_a_hand_calculation`) |
| Checklist scoring | Checked against the published point table on hand-made respondents (`test_ada_checklist_points_follow_the_paper_test`) |
| Odds-ratio code | Recovers a planted effect in simulated data, and the statsmodels fit converged on the real training rows for both endpoints |
| Explanations in the write-up | The confounding patterns we describe (for example, smokers being older and more often men) were checked against the training rows before we wrote them down |
| Retraining on another operating system | Rerunning `train` on Windows (Python 3.13.9, the same locked library versions) gave mean validation scores within 0.0003 PR-AUC, 0.001 ROC-AUC and 0.00001 Brier score of the committed run, for every model and both endpoints, and the selection rule chose XGBoost again for both. XGBoost's floating-point arithmetic differs slightly between operating systems, so stage 6 reproduces to these tolerances rather than byte for byte |

**One problem we found and fixed.** In a first full run, the permutation-importance chart showed almost every unimportant answer with a small negative score, as if shuffling vegetables or fruit improved the model. The cause was a scikit-learn detail: with `max_samples` below 1, the unshuffled baseline is scored on the whole fold but the shuffled copies on a random subsample, and PR-AUC on half a fold isn't comparable with PR-AUC on the whole fold. We now score both on the whole held-out fold, and the committed results come from the corrected code.

## Stage 7

| Check | Result |
| --- | --- |
| Cutoff chosen on training rows only | `threshold` never calls the function that releases test rows; both threshold manifests record `"test_rows_used": 0` |
| Out-of-fold predictions reproduce stage 6 | Combined: XGBoost 0.4609 against stage 6's 0.4610, logistic regression 0.4391 against 0.4391. Prediabetes-only: 0.0599 against 0.0599 and 0.0550 against 0.0550. The command warns if a model drifts by more than 0.0005 |
| `threshold.json` is deterministic | Running `threshold` a second time gave byte-identical `threshold.json`, `operating_points.csv` and `calibration_validation.csv` for both endpoints |
| The lock holds | `test` refuses without `threshold.json`, with one made for another split or input, and when a model file's SHA-256 differs from the one recorded (`test_cli_locks_the_threshold_before_the_test_set_is_scored_once`) |
| Calibration rule applied as written | Out-of-fold slopes 0.992 (combined) and 0.981 (prediabetes), intercepts within 0.001 of zero, so no recalibration; Platt scaling itself is tested on planted miscalibration |
| Bootstrap and subgroup intervals | Bootstrap intervals are reproducible with a fixed seed, contain the point estimate and collapse to a point for a perfect separator; Wilson intervals and subgroup counts are checked on hand-made data |
| Original classes line up | `--source` checks the three-class file's SHA-256 against the provenance and that its recoded labels reproduce the cohort's outcome row by row before splitting misses into prediabetes and diabetes |
| `test` is deterministic | A second `test` run (used to redraw two charts after a labelling fix) wrote byte-identical `test_results.csv`, `calibration_test.csv`, `reliability_test.csv`, `subgroups.csv`, `missed_cases.csv`, `flagged_by_original_class.csv`, `manifest.json` and `README.md` for both endpoints |

## Stage 8

**Two portability bugs, found by running the project on Windows and fixed.**

1. *Line endings.* pandas writes the operating system's line ending by default, so `prepare` on Windows wrote CRLF files whose hash didn't match the committed split, and every later command refused them. `prepare` and `save_split` now always write `\n`, and a test checks the prepared file contains no carriage returns. With the fix, the Windows-prepared files match the Linux hashes exactly.
2. *XGBoost pickles across operating systems.* XGBoost pickles a model with its random-number generator's state written as text, and that text layout differs between Linux and Windows builds, so Windows refused the committed Linux models ("input stream corrupted"). The trees themselves are stored in XGBoost's portable format inside the same file. `modeling/portable.py` loads that part when the full snapshot can't be read, without changing the committed file or its hash; `training_workflow.load_pipeline` and stage 7 use it. A test makes a real snapshot unreadable (raising the error Windows gives for a Linux pickle) and checks the fallback predicts identically.

**Stage 1–4 reports regenerated.** Both reports were rerun with the current code. Every aggregate table (class balance, group summaries, categorical factors, correlations, data dictionary) and every count is identical. What changed is what the README predicted: the new `review_flags` in `cleaning.json`, three per-unit odds-ratio columns and updated notes in `hypotheses.csv`, the expanded observations and the relabelled plots. The provenance now names the Kaggle `archive.zip` as the source file; it holds the same three-class CSV as the team's original upload. The hand-written report READMEs were kept.

**Notebooks.** All seven notebooks in `notebooks/` run top to bottom without errors or warnings from a fresh kernel, reading only committed reports.

## The code

- 63 tests pass on Python 3.13.9 (Windows) with the versions in `requirements.lock`: the 39 stage 1–4 tests, the 18 stage 5–6 tests and 6 new stage 7 tests. Statement coverage is 98.10% (CI requires 80%). The stage 7 tests also pass with deprecation and future warnings treated as errors; the 35 warnings left in a full run come from the stage 1–4 plotting code, as before. `pip check` is clean.
- The `modeling` package and the new tests pass `flake8 --max-line-length 120`.
- Changes to stage 1–4 files are deliberate and small: one argument in `preparation.py` (the line-ending fix) with one assertion added to `tests/test_preparation.py`; the regenerated reports; one sentence each in `reports/prediabetes/README.md` and `docs/validation.md` that said stages 5–8 were future work; and the rewritten root README. All other stage 1–4 code is unchanged.

## Known limits

The final, unbiased numbers are the stage 7 test-set results. The validation numbers in stages 5 and 6 are slightly optimistic, because tuning, model selection and the cutoff all used the same folds; the test set agrees with them closely (PR-AUC 0.465 against 0.461). Remote CI results are not claimed here until the pull request runs.

# Checks we ran for stages 5 and 6

This page records how we checked the stage 5 and 6 work, in the same spirit as the stage 1–4 [validation record](validation.md), which is left as it was.

## The data

- **Source.** The Kaggle download `archive.zip` ([Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset), archive SHA-256 `ecbd6572…`). Its three-class CSV has SHA-256 `a971809d…`, the same as the source recorded in both stage 1–4 provenance files.
- **Preparation reproduces exactly.** Running the stage 1–4 `prepare` command on that zip gave prepared files whose hashes match the committed reports byte for byte: `7e0e0421…` (combined) and `33018d98…` (prediabetes-only).
- **The official binary file.** A row-by-row comparison shows Kaggle's `diabetes_binary` file has 35,346 positives, all of them existing diabetes. All 4,631 prediabetes respondents are coded 0, and the features are identical to the three-class file. This answers the open question in the README: that file is the diabetes-only definition.

## Stage 5

| Check | Result |
| --- | --- |
| Split counts, combined | Train 202,944 (31,982 positive); test 50,736 (7,995 positive); folds of 40,588–40,589 rows with 6,396–6,397 positive |
| Split counts, prediabetes-only | Train 174,667 (3,705 positive); test 43,667 (926 positive); 741 positive in every fold |
| Reproducible | Separate runs produced byte-identical `split_assignment.csv.gz` files (hashes `e8a57949…` and `76326823…`) and identical fold metrics |
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

**One problem we found and fixed.** In a first full run, the permutation-importance chart showed almost every unimportant answer with a small negative score, as if shuffling vegetables or fruit improved the model. The cause was a scikit-learn detail: with `max_samples` below 1, the unshuffled baseline is scored on the whole fold but the shuffled copies on a random subsample, and PR-AUC on half a fold isn't comparable with PR-AUC on the whole fold. We now score both on the whole held-out fold, and the committed results come from the corrected code.

## The code

- 57 tests pass on Python 3.12.3 with `requirements.lock`, the same setup CI uses: the 39 stage 1–4 tests, unchanged, plus 18 new ones. Statement coverage is 98.46% (CI requires 80%). The new tests also pass with deprecation and future warnings treated as errors; the 35 warnings left in a full run come from the stage 1–4 plotting code, as before. `pip check` is clean and the package builds.
- New modules pass `flake8 --max-line-length 120`.
- The stage 1–4 files are untouched. Comparing the repository with the original zip, the only differences in existing files are additions: a banner and a new section in the README, two dependencies and one command entry in `pyproject.toml`, and five new pinned packages in `requirements.lock`. Every other stage 1–4 file, including all of `reports/uploaded-combined/`, `reports/prediabetes/` and `docs/`, is byte-identical to the original.

## Known limits

The numbers in stages 5 and 6 are validation-fold numbers on training rows. They're slightly optimistic, because tuning and model selection used the same folds. The held-out test set, opened once in stage 7 after the threshold is fixed, gives the final, unbiased figures.

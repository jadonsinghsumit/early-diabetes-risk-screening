# Plan for roadmap stages 5–8

Status: stages 1–4 are implemented and tested (see [validation](validation.md)). This plan is for the teammates picking up stages 5–8. Sections 0 and 1 contain decisions to make **before** writing model code.

## 0. Facts that shape every later stage

| Fact | Consequence |
| --- | --- |
| Primary cohort: 218,334 respondents, **4,631 positives (2.12%)** after excluding existing diabetes | Accuracy is meaningless (always predicting "no" scores 97.9%). Use PR-AUC, recall, precision, F2 and calibration. |
| An 80/20 stratified split leaves about 43,667 test rows with **about 926 positives**; each of 5 training folds holds about 741 positives | Metrics on the test set carry real uncertainty: report bootstrap confidence intervals, and do not tune on the test set. |
| **23,650 identical profiles** are kept in the data | Identical rows can land in both train and test and flatter the scores. Split and cross-validate by profile group (see stage 5). |
| Outcome is *self-reported prediabetes*, which needs a test and a conversation with a doctor | Healthcare-contact variables (`CholCheck`, `HighBP`, `HighChol`, `AnyHealthcare`, `NoDocbcCost`) partly measure access to care. Test a model with and without them. |
| The data are a survey extract with no weights, strata or clusters | No population-prevalence claims; unweighted metrics describe this table only. |
| Cross-sectional data | Wording must be "identify current self-reported prediabetes", never "predict who will develop diabetes". |

## 1. Decisions to settle first (one short team meeting)

1. **Endpoint.** The roadmap text assumes `Diabetes_binary` (prediabetes **or** diabetes, about 14–16% positive). The project lead has set the primary endpoint to prediabetes versus no diabetes. Keep that, but run stage 5–7 code through a single `--endpoint` switch so the combined endpoint can be reported as a comparison with no extra work (`prepare --endpoint combined` already exists).
2. **Screening target.** Fix, in writing, the minimum sensitivity the model must reach (for example "at least 80% of prediabetes cases flagged") and who judges the false-alarm cost. Stage 7 picks the threshold from this statement, not from the test set.
3. **Feature policy.** Decide whether the headline model may use the healthcare-contact variables (see section 0). Recommended: report both.
4. **Interfaces.** Agree on the shared functions below so three people's code fits together.

## 2. Shared module contract (`src/diabetes_screening/modeling/`)

| Function | Purpose |
| --- | --- |
| `load_cohort(path, provenance)` | Reads the prepared CSV, checks the provenance hash and returns `X`, `y` and a `profile_id` (hash of the 21 feature values) per row. |
| `make_split(X, y, groups, seed=42)` | Stratified, group-aware 80/20 split. Saves **row indices and the seed** to `outputs/.../split.json`, not data. The test indices are used once, at the end. |
| `build_preprocessor(kind)` | `ColumnTransformer` for `"linear"` (robust-scale BMI and health days, keep binary columns, ordinal columns numeric) or `"tree"` (no scaling). |
| `cv_splitter(seed)` | `StratifiedGroupKFold(n_splits=5, shuffle=True)` on the training set only. |
| `evaluate(y_true, proba, threshold=None)` | Returns every stage 7 metric in one dictionary, so all models are scored identically. |

All preprocessing lives inside a scikit-learn `Pipeline`, so scalers are fitted on training folds only.

## 3. Stage 5: splitting and imbalance (why: an honest evaluation)

- Implement `make_split` and `cv_splitter` as above, with the profile-group rule. Add a test that no profile appears on both sides of the split.
- Strategies to compare, all inside cross-validation: (a) no weighting (baseline), (b) `class_weight="balanced"` (logistic regression, random forest) or `scale_pos_weight` (XGBoost). Treat resampling (SMOTE or undersampling) as optional and only inside the training folds through `imblearn.pipeline.Pipeline`; justify it if used, because reweighting usually does as well here.
- Remember that weighting distorts probabilities: stage 7 must recalibrate before reading probabilities as risks.
- **Done when:** a saved split file, a leakage test, and a short table comparing weighted and unweighted baselines on validation folds.

## 4. Stage 6: training and tuning (why: interpretable baseline first, then test nonlinear gains)

1. **Baselines.** A dummy classifier (prevalence) and a simple rule (for example high blood pressure and age 9 or above), so every model is compared with something simple.
2. **Models.** Logistic regression (L2, tune `C`), random forest (tune `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`), XGBoost (tune `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `scale_pos_weight`). Add `xgboost` to `pyproject.toml` and refresh `requirements.lock`; CI must stay green.
3. **Tuning.** `RandomizedSearchCV` over `cv_splitter`, **primary scoring `average_precision`** (PR-AUC), secondary F2 and recall at a fixed precision. Fix random seeds; log every search to `outputs/.../tuning.csv`.
4. **Inference.** Use `statsmodels.Logit` separately for coefficients, adjusted odds ratios and confidence intervals, and check multicollinearity (variance inflation factors; `Age`, `Income`, `Education` and `GenHlth` are related). Compare the adjusted odds ratios with the unadjusted ones from stage 4 and explain every sign change, as in a Simpson's-paradox check.
5. **Interpretation.** Permutation importance on validation data for all three models; optional SHAP for XGBoost. Run the model with and without the healthcare-contact variables.
- **Done when:** a comparison table (mean and spread across folds, per model), the chosen hyperparameters, fitted pipelines saved with `joblib`, and a feature-interpretation page.

## 5. Stage 7: evaluation and thresholds (why: choose the operating point, then report honestly)

1. **Threshold.** Get out-of-fold predicted probabilities on the **training** set, recalibrate if the model was weighted (isotonic or Platt, fitted in-fold), and choose the threshold that meets the sensitivity target from decision 2 (or maximizes F2). Write it to `threshold.json` and commit it **before** anyone scores the test set.
2. **Final test.** Score the locked test set once. Report recall, precision, specificity, PR-AUC, ROC-AUC, F2 and the confusion matrix, each with a bootstrap 95% interval (1,000 resamples).
3. **Workload.** Convert to plain language per 1,000 people screened: how many flagged, how many true cases found, how many missed, and the positive predictive value. With 2.12% prevalence even a good model has a low PPV (illustratively, 80% sensitivity and 70% specificity give about 5%), so say so plainly.
4. **Calibration.** Reliability curve, Brier score and calibration slope and intercept.
5. **Subgroups.** Recall and false-positive rate by age band, sex and income band, to find groups the model serves badly.
6. **Figures.** PR and ROC curves, calibration curve, and a false-negative discussion (who is missed and why).
- **Done when:** `threshold.json` is committed before the test evaluation, the test results table is produced by one script, and the false-negative discussion is written.

## 6. Stage 8: reporting and handoff (why: a reproducible team submission)

- **Code.** Keep logic in the package; notebooks (`notebooks/01_…07_`) only call functions and tell the story. Add a `diabetes-screening model` command that runs stages 5–7 from the prepared cohort.
- **Reproducibility.** Update `requirements.lock`, record seeds, versions and input hashes in a manifest (the stage 1–4 manifest is the template), and re-run from a fresh clone on a clean machine.
- **README.** Setup, data source and provenance (including the BRFSS 2015 versus UCI "2014" discrepancy noted in [sources](sources.md)), methods, results, limitations, and the statement that this is a research prototype, not a diagnostic tool.
- **Report.** Technical PDF with high-resolution figures (300 DPI, labelled axes) and the sources from the roadmap. Check the date and relevance of the NHANES PDF before citing it as current evidence.
- **Video.** 8–10 minute walkthrough: question, data and the endpoint choice, EDA highlights, models, the locked threshold, honest limits. Rehearse once with the final numbers.
- **Done when:** a fresh clone reproduces every table and figure, and the PDF, video and README tell the same story with the same numbers.

## 7. Suggested split of work and order

| Order | Work | Suggested owner | Depends on |
| --- | --- | --- | --- |
| 1 | Decisions in section 1; the `modeling/` contract | Everyone | none |
| 2 | Stage 5 split, leakage test, weighting comparison | Sumit | 1 |
| 3 | Stage 6 logistic regression plus statsmodels inference | Priyamvadha | 2 |
| 3 | Stage 6 random forest and XGBoost tuning | Amiya | 2 |
| 4 | Stage 7 threshold, calibration, test evaluation | Sumit (reviewed by one teammate) | 3 |
| 5 | Stage 8 notebooks, README, PDF, video | Everyone, Sumit integrates | 4 |

The owners are a suggestion; change them freely. Merge to `main` only through pull requests that pass CI. The test set stays untouched until step 4.

## 8. Risks to watch

- Tuning on the test set, even indirectly (for example, looking at test results and then changing the features).
- Leakage through identical profiles or scalers fitted before the split.
- Reading weighted-model probabilities as true risks.
- Calling the model a predictor of future diabetes, or a diagnostic tool.
- Presenting healthcare-contact variables as causes.
- A very small positive count in subgroups: report counts next to every subgroup metric.

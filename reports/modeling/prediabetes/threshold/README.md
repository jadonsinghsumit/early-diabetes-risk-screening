# Stage 7a: the locked cutoff (Prediabetes vs no diabetes)

Everything on this page comes from the 174,667 training rows (3,705 positive) and the five saved folds of split `7632682314c1`. The 43,667 test rows were not loaded.

**Locked cutoff: 0.0172** on the predicted risk from XGBoost (`xgboost.joblib`, SHA-256 `2efdd41db100…`). Anyone at or above it is flagged.

Rule: The highest cutoff on the selected model's out-of-fold predicted risks (training rows only) that still flags at least 80% of the training cases. The highest such cutoff flags the fewest people, so it is the cheapest way to meet the target. Fixed before any test row is read.

On the out-of-fold predictions this cutoff catches 80.0% of cases and flags 43.5% of people; 3.9% of the people it flags have the condition. Per 1,000 people screened that is about 435 flagged, 17 cases found, 4 missed and 418 false alarms.

## Other cutoffs (for information only)

What each alternative would cost on the same out-of-fold predictions. Only the locked row is used.

| rule | threshold | recall | precision | specificity | flagged_share | flagged_per_1000 | cases_found_per_1000 | cases_missed_per_1000 | false_alarms_per_1000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catch 70% of cases | 0.023 | 0.700 | 0.045 | 0.678 | 0.330 | 330 | 15 | 6 | 315 |
| catch 75% of cases | 0.020 | 0.750 | 0.042 | 0.628 | 0.380 | 380 | 16 | 5 | 365 |
| **catch 80% of cases (locked)** | 0.017 | 0.800 | 0.039 | 0.573 | 0.435 | 435 | 17 | 4 | 418 |
| catch 85% of cases | 0.014 | 0.850 | 0.037 | 0.516 | 0.491 | 491 | 18 | 3 | 473 |
| catch 90% of cases | 0.011 | 0.900 | 0.034 | 0.439 | 0.569 | 569 | 19 | 2 | 549 |
| best F2 | 0.042 | 0.419 | 0.062 | 0.862 | 0.144 | 144 | 9 | 12 | 135 |

## Calibration on the out-of-fold predictions

Rule, fixed before this run: Recalibrate (Platt scaling: a logistic regression of the outcome on the model's out-of-fold log-odds) only if the calibration slope is outside 0.9-1.1 or the calibration-in-the-large intercept is further than 0.1 from 0. Recalibration is monotone, so it never changes who is flagged, only the risk a person is told.

| model | observed_rate | mean_predicted_risk | brier | calibration_intercept | calibration_slope |
| --- | --- | --- | --- | --- | --- |
| XGBoost | 0.021 | 0.021 | 0.020 | -0.001 | 0.981 |
| Logistic regression | 0.021 | 0.021 | 0.020 | -0.002 | 1.024 |

Recalibration: not needed: the selected model's risks already meet the rule, so they are used as they are.

![Out-of-fold calibration](calibration_validation.png)

## Check against stage 6

Refitting the saved pipelines fold by fold should reproduce the stage 6 refit check.

| model | oof_pr_auc_now | stage6_refit_check | reproduces |
| --- | --- | --- | --- |
| XGBoost | 0.0599 | 0.0599 | True |
| Logistic regression | 0.0550 | 0.0550 | True |

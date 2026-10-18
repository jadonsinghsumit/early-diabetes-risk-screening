# Stage 6: tuned models (Prediabetes or diabetes vs no diabetes)

Everything on this page comes from the 202,944 training rows (31,982 positive) and the five saved folds of split `e8a579495ac8`. The 50,736 test rows were not loaded.

## How the models compare

Mean ± SD over the five validation folds. The `_at_80_recall` columns answer a screening question: if we lower the cutoff until 80% of real cases get flagged, how many of the flagged people actually have the condition (precision), how many people without it are correctly left alone (specificity), and what share of everyone ends up flagged? Ignore the checklist's Brier score: points divided by 9 isn't a risk.

| model | pr_auc | roc_auc | brier | precision_at_80_recall | specificity_at_80_recall | flagged_share_at_80_recall |
| --- | --- | --- | --- | --- | --- | --- |
| No model (prevalence) | 0.158 ± 0.000 | 0.500 ± 0.000 | 0.133 ± 0.000 | 0.158 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| ADA-style checklist | 0.325 ± 0.004 | 0.756 ± 0.004 | 0.228 ± 0.001 | 0.256 ± 0.022 | 0.544 ± 0.091 | 0.514 ± 0.085 |
| Logistic regression | 0.439 ± 0.007 | 0.819 ± 0.004 | 0.108 ± 0.001 | 0.324 ± 0.006 | 0.688 ± 0.009 | 0.389 ± 0.007 |
| Random forest | 0.456 ± 0.007 | 0.822 ± 0.004 | 0.106 ± 0.001 | 0.326 ± 0.005 | 0.690 ± 0.007 | 0.387 ± 0.005 |
| XGBoost | 0.461 ± 0.006 | 0.826 ± 0.003 | 0.106 ± 0.001 | 0.332 ± 0.004 | 0.698 ± 0.005 | 0.380 ± 0.005 |

The paper checklist used the way clinics use it (5 or more points means increased risk):

| model | recall | precision | specificity | flagged_share |
| --- | --- | --- | --- | --- |
| ADA-style checklist | 0.802 ± 0.005 | 0.265 ± 0.002 | 0.584 ± 0.002 | 0.477 ± 0.001 |

**Chosen for stage 7: XGBoost.** Reason: highest mean validation PR-AUC (0.4610), ahead of Random forest by 0.0052 in 5 of 5 folds (corrected t-test p = 0.005).

The rule, written down before the full run: 1) Start from the model with the best mean validation PR-AUC. 2) If logistic regression is not significantly worse than it, take logistic regression, because it is the easiest to explain. 3) Otherwise, among the models that are not significantly worse than the best one, take the one with the lowest mean Brier score, because stage 7 reads its probabilities as risks. 'Significantly worse' means p < 0.05 in Nadeau and Bengio's corrected resampled t-test on the five fold-by-fold PR-AUC differences.

## Fold-by-fold gaps in PR-AUC

All models were scored on the same five folds, so we can compare them fold by fold. `p_value` comes from Nadeau and Bengio's corrected resampled t-test, which allows for the folds sharing most of their training data. With only five folds it's a cautious test.

| model | compared_with | mean_pr_auc_gain | sd_pr_auc_gain | folds_won | folds | corrected_t | p_value |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic regression | ADA-style checklist | 0.114 | 0.004 | 5 | 5 | 38.038 | 0.000 |
| Random forest | ADA-style checklist | 0.131 | 0.003 | 5 | 5 | 62.846 | 0.000 |
| XGBoost | ADA-style checklist | 0.136 | 0.003 | 5 | 5 | 59.613 | 0.000 |
| Random forest | Logistic regression | 0.017 | 0.002 | 5 | 5 | 10.741 | 0.000 |
| XGBoost | Logistic regression | 0.022 | 0.002 | 5 | 5 | 15.282 | 0.000 |
| XGBoost | Random forest | 0.005 | 0.001 | 5 | 5 | 5.638 | 0.005 |

## Without the healthcare-contact answers

Same tuned settings, refitted without CholCheck, HighBP, HighChol, AnyHealthcare and NoDocbcCost. These answers partly reflect whether someone sees a doctor, so it's worth knowing how much the models lean on them.

| model | pr_auc_all_21 | pr_auc_without_contact | mean_drop | sd_drop | folds_where_full_is_better |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.439 | 0.409 | 0.030 | 0.002 | 5 |
| Random forest | 0.456 | 0.427 | 0.029 | 0.003 | 5 |
| XGBoost | 0.461 | 0.431 | 0.030 | 0.003 | 5 |

## Tuned settings

The refit check re-scores the winning settings with a fresh fit on each fold. It should match the search's own score, and it does when the run is reproducible.

- **Logistic regression**: `{"model__C": 3.47027, "preprocess__kind": "linear-onehot"}` (search PR-AUC 0.4391, refit check 0.4391)
- **Random forest**: `{"model__n_estimators": 200, "model__min_samples_leaf": 10, "model__max_features": 0.33, "model__max_depth": 12}` (search PR-AUC 0.4558, refit check 0.4558)
- **XGBoost**: `{"model__colsample_bytree": 0.63999, "model__learning_rate": 0.05758, "model__max_depth": 3, "model__min_child_weight": 25, "model__n_estimators": 800, "model__reg_lambda": 6.8131, "model__subsample": 0.70618}` (search PR-AUC 0.4610, refit check 0.4610)

## Adjusted odds ratios (statsmodels, training rows only)

Unadjusted means one answer at a time; adjusted means all 21 answers in one ordinary logistic regression. `change_after_adjustment` flags associations that weaken, vanish or reverse once the other answers are held fixed (our Simpson's-paradox check). VIF above about 5 would mean an answer overlaps heavily with the others.

| feature | unit | unadjusted_or | adjusted_or | adjusted_ci_low | adjusted_ci_high | change_after_adjustment | vif |
| --- | --- | --- | --- | --- | --- | --- | --- |
| HighBP | yes vs no | 4.777 | 2.028 | 1.967 | 2.091 | weaker | 1.332 |
| HighChol | yes vs no | 3.231 | 1.810 | 1.760 | 1.862 | weaker | 1.172 |
| CholCheck | yes vs no | 5.888 | 3.329 | 2.907 | 3.812 | weaker | 1.034 |
| BMI | per 1 kg/m² | 1.083 | 1.064 | 1.062 | 1.066 | weaker | 1.142 |
| Smoker | yes vs no | 1.404 | 0.983 | 0.956 | 1.011 | no longer clear | 1.092 |
| Stroke | yes vs no | 2.886 | 1.076 | 1.020 | 1.136 | weaker | 1.081 |
| HeartDiseaseorAttack | yes vs no | 3.485 | 1.218 | 1.173 | 1.266 | weaker | 1.171 |
| PhysActivity | yes vs no | 0.495 | 0.951 | 0.922 | 0.980 | weaker | 1.158 |
| Fruits | yes vs no | 0.792 | 0.954 | 0.927 | 0.982 | weaker | 1.112 |
| Veggies | yes vs no | 0.687 | 0.975 | 0.943 | 1.008 | no longer clear | 1.111 |
| HvyAlcoholConsump | yes vs no | 0.413 | 0.518 | 0.479 | 0.559 | weaker | 1.024 |
| AnyHealthcare | yes vs no | 1.238 | 1.061 | 0.990 | 1.137 | no longer clear | 1.114 |
| NoDocbcCost | yes vs no | 1.397 | 1.070 | 1.019 | 1.122 | weaker | 1.143 |
| GenHlth | per step toward 'poor' | 2.169 | 1.679 | 1.651 | 1.708 | weaker | 1.793 |
| MentHlth | per 1 day | 1.024 | 0.998 | 0.997 | 1.000 | no longer clear | 1.238 |
| PhysHlth | per 1 day | 1.045 | 0.993 | 0.992 | 0.995 | reversed | 1.624 |
| DiffWalk | yes vs no | 3.682 | 1.104 | 1.065 | 1.144 | weaker | 1.532 |
| Sex | male vs female | 1.177 | 1.273 | 1.238 | 1.309 | stronger | 1.075 |
| Age | per 5-year age band | 1.206 | 1.137 | 1.130 | 1.143 | weaker | 1.349 |
| Education | per 1 code step | 0.714 | 0.966 | 0.952 | 0.980 | weaker | 1.325 |
| Income | per 1 code step | 0.813 | 0.946 | 0.939 | 0.953 | weaker | 1.501 |

## The five answers each model relies on most (permutation importance)

How much validation PR-AUC drops when one answer is shuffled, averaged over the five folds.

| model | feature | pr_auc_drop_mean | pr_auc_drop_sd |
| --- | --- | --- | --- |
| Logistic regression | GenHlth | 0.085 | 0.004 |
| XGBoost | BMI | 0.082 | 0.004 |
| Random forest | BMI | 0.080 | 0.003 |
| XGBoost | GenHlth | 0.073 | 0.004 |
| Random forest | GenHlth | 0.069 | 0.002 |
| Logistic regression | BMI | 0.064 | 0.004 |
| Logistic regression | Age | 0.040 | 0.002 |
| XGBoost | Age | 0.038 | 0.002 |
| Random forest | HighBP | 0.036 | 0.001 |
| Random forest | HighChol | 0.031 | 0.002 |
| Logistic regression | HighBP | 0.031 | 0.001 |
| XGBoost | HighChol | 0.029 | 0.002 |
| XGBoost | HighBP | 0.028 | 0.001 |
| Random forest | Age | 0.027 | 0.001 |
| Logistic regression | HighChol | 0.023 | 0.001 |

![Permutation importance](permutation_importance.png)

![Model comparison](comparison.png)

![Odds ratios](odds_ratios.png)

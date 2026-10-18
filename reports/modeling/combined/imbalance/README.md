# Stage 5: does class weighting help? (Prediabetes or diabetes vs no diabetes)

These results use the 202,944 training rows only (31,982 positive, 15.76%). The 50,736 test rows were never loaded. Split `e8a579495ac8` (stratified, seed 42).

Each number is the mean ± SD over the five validation folds. Recall and precision here use the default 0.5 cutoff, which is only a placeholder; stage 7 picks the real one. For reference, a model that ranks people at random gets a PR-AUC equal to the share of positives.

| model | weighting | pr_auc | roc_auc | brier | mean_predicted_risk | recall | precision |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | unweighted | 0.431 ± 0.007 | 0.817 ± 0.004 | 0.109 ± 0.001 | 0.158 ± 0.001 | 0.183 ± 0.006 | 0.552 ± 0.012 |
| logistic_regression | weighted | 0.430 ± 0.007 | 0.818 ± 0.004 | 0.179 ± 0.001 | 0.395 ± 0.001 | 0.762 ± 0.006 | 0.339 ± 0.003 |
| random_forest | unweighted | 0.456 ± 0.007 | 0.822 ± 0.004 | 0.106 ± 0.001 | 0.158 ± 0.001 | 0.144 ± 0.002 | 0.611 ± 0.010 |
| random_forest | weighted | 0.455 ± 0.007 | 0.823 ± 0.004 | 0.172 ± 0.001 | 0.386 ± 0.001 | 0.771 ± 0.008 | 0.339 ± 0.003 |
| xgboost | unweighted | 0.458 ± 0.006 | 0.825 ± 0.003 | 0.106 ± 0.001 | 0.157 ± 0.001 | 0.203 ± 0.003 | 0.585 ± 0.013 |
| xgboost | weighted | 0.459 ± 0.006 | 0.825 ± 0.003 | 0.176 ± 0.001 | 0.385 ± 0.001 | 0.790 ± 0.007 | 0.335 ± 0.003 |

## Does weighting change who ranks as high risk, or just move the cutoff?

Averaged over folds. A Spearman correlation close to 1 means the weighted and unweighted models put people in almost the same order. A small SD in the log-odds shift means weighting simply slides everyone's score up by about the same amount, which is the same as using a lower cutoff.

| model | spearman_weighted_vs_unweighted | mean_logit_shift | sd_logit_shift |
| --- | --- | --- | --- |
| logistic_regression | 0.999 | 1.625 | 0.124 |
| random_forest | 0.997 | 1.598 | 0.145 |
| xgboost | 0.997 | 1.643 | 0.153 |

## Do repeated answer profiles inflate the scores? (unweighted models)

Same training rows, two ways of forming the folds: ordinary stratified folds, and folds that keep everyone with identical answers together. If repeated profiles were leaking labels, the second version would score clearly lower.

| model | cv_folds | pr_auc | roc_auc | brier | pr_auc_drop_vs_plain |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | plain stratified | 0.431 ± 0.007 | 0.817 ± 0.004 | 0.109 ± 0.001 | 0.000 |
| logistic_regression | profile-grouped | 0.431 ± 0.001 | 0.817 ± 0.003 | 0.109 ± 0.000 | 0.000 |
| random_forest | plain stratified | 0.456 ± 0.007 | 0.822 ± 0.004 | 0.106 ± 0.001 | 0.000 |
| random_forest | profile-grouped | 0.455 ± 0.003 | 0.823 ± 0.004 | 0.107 ± 0.001 | 0.001 |
| xgboost | plain stratified | 0.458 ± 0.006 | 0.825 ± 0.003 | 0.106 ± 0.001 | 0.000 |
| xgboost | profile-grouped | 0.458 ± 0.004 | 0.825 ± 0.004 | 0.106 ± 0.001 | 0.000 |

![Weighting comparison](weighting_comparison.png)

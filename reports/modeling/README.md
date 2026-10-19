# Stage 5, 6 and 7 results

One folder per endpoint: `combined/` is the main analysis (prediabetes or diabetes vs no diabetes), and `prediabetes/` is the sensitivity analysis (prediabetes vs no diabetes, existing diabetes excluded). Each has five parts, in the order they were made:

- `split/`: the shared train/test split and cross-validation folds (stage 5). Load it with `diabetes_screening.modeling.load_split` instead of making a new one, so everyone works on the same rows.
- `imbalance/`: the class-weighting comparison and the repeated-answers check (stage 5). Start with its `README.md`.
- `training/`: the tuned models, the comparison with the paper checklist, the chosen model, odds ratios and permutation importance (stage 6). Start with its `README.md`.
- `threshold/`: the locked cutoff (`threshold.json`), the alternatives it was chosen from, and the out-of-fold calibration check (stage 7, training rows only).
- `evaluation/`: the one-time scoring of the locked test set at that cutoff: metrics with bootstrap intervals, workload per 1,000 people, calibration, subgroups and the missed cases (stage 7). Start with its `README.md`.

These files hold counts, metrics, settings and charts only, no survey answers. `training/models/` has the fitted logistic regression and XGBoost pipelines; load them with `diabetes_screening.modeling.training_workflow.load_pipeline`, which also works for XGBoost models pickled on another operating system. The fitted random forest (about 16 MB) and the out-of-fold predictions (one row per respondent) are left out; `diabetes-screening-model train` writes both to your local `outputs/` folder. Everything here was produced by the commands in the main README; the write-ups are [splitting](../../docs/splitting.md), [training](../../docs/training.md) and [evaluation](../../docs/evaluation.md).

"""The two stage 5 commands: make the split, then collect the evidence on class imbalance.

AI-use disclosure: drafted with AI assistance (Claude, Anthropic); see docs/splitting.md.
"""
from importlib.metadata import version
from pathlib import Path
import platform

from ..endpoints import LABELS
from ..ingestion import write_json
from ..schema import EXPECTED_ROWS
from .cohort import load_cohort, profile_summary
from .imbalance import MODELS, compare_weighting, grouped_cv_check, model_factory, plot_weighting, summarize
from .split import load_split, make_split, save_split, training_data

PACKAGES = ["numpy", "pandas", "scikit-learn", "xgboost", "scipy", "matplotlib"]


def endpoint_label(endpoint):
    negative, positive = LABELS[endpoint]
    return f"{positive} vs {negative.lower()}"


def run_split(source, provenance, destination, seed=42, test_size=0.2, n_splits=5, group_profiles=False,
              expected_source_rows=EXPECTED_ROWS):
    """Create and save the split for one prepared cohort; also record the repeated-profile audit."""
    cohort = load_cohort(source, provenance, expected_source_rows)
    plan = make_split(cohort, seed=seed, test_size=test_size, n_splits=n_splits, group_profiles=group_profiles)
    summary = save_split(plan, cohort, destination)
    write_json(Path(destination) / "profiles.json", profile_summary(cohort))
    return summary


def _markdown(frame, digits=3):
    def cell(value):
        return f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + " --- |" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def _mean_sd(mean, sd):
    if mean != mean:  # NaN: e.g. precision when the 0.5 cutoff flags nobody in any fold
        return "n/a (none flagged)"
    return f"{mean:.3f}" if sd != sd else f"{mean:.3f} ± {sd:.3f}"


def _with_spread(summary, keys, metrics):
    table = summary[list(keys)].copy()
    for metric in metrics:
        table[metric] = [_mean_sd(m, s) for m, s in zip(summary[f"{metric}_mean"], summary[f"{metric}_sd"])]
    return table


def run_imbalance(source, provenance, split_directory, destination, models=MODELS, grouped_check=True,
                  expected_source_rows=EXPECTED_ROWS, log=print):
    """Compare weighted and unweighted models on the saved CV folds; the test rows are never loaded."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    cohort = load_cohort(source, provenance, expected_source_rows)
    plan = load_split(split_directory, cohort)
    X, y, folds = training_data(cohort, plan)
    log(f"Training rows: {len(y):,} ({int(y.sum()):,} positive); test rows untouched: {len(plan.test_index):,}")
    per_fold, agreement, timing = compare_weighting(X, y, folds, models, plan.seed, log=log)
    summary = summarize(per_fold)
    per_fold.to_csv(destination / "weighting_folds.csv", index=False)
    summary.to_csv(destination / "weighting_summary.csv", index=False)
    agreement.to_csv(destination / "weighting_rank_agreement.csv", index=False)
    timing.to_csv(destination / "timing.csv", index=False)
    plot_weighting(summary, destination / "weighting_comparison.png", endpoint_label(cohort.endpoint))
    check = None
    if grouped_check:
        profiles = cohort.profile_id.iloc[plan.train_index].to_numpy()
        check = grouped_cv_check(X, y, folds, profiles, models, plan.seed, plain_per_fold=per_fold, log=log)
        check.to_csv(destination / "grouped_cv_check.csv", index=False)
    settings = {name: {step: repr(est) for step, est in model_factory(name, True, plan.seed)(y).steps}
                for name in models}
    manifest = {
        "stage": 5, "endpoint": cohort.endpoint, "input_sha256": cohort.input_sha256,
        "split_sha256": plan.sha256, "split_strategy": plan.strategy, "seed": plan.seed,
        "training_rows_used": int(len(y)), "test_rows_used": 0, "models": list(models),
        "model_settings_weighted_variant": settings,
        "weighting": {"logistic_regression": "class_weight='balanced'", "random_forest": "class_weight='balanced'",
                      "xgboost": "scale_pos_weight = negatives/positives of each training fold"},
        "python": platform.python_version(), "packages": {name: version(name) for name in PACKAGES},
    }
    write_json(destination / "manifest.json", manifest)
    agreement_summary = agreement.groupby("model", sort=False)[
        ["spearman_weighted_vs_unweighted", "mean_logit_shift", "sd_logit_shift"]].mean().reset_index()
    text = [
        f"# Stage 5: does class weighting help? ({endpoint_label(cohort.endpoint)})", "",
        f"These results use the {len(y):,} training rows only ({int(y.sum()):,} positive, {y.mean():.2%}). "
        f"The {len(plan.test_index):,} test rows were never loaded. Split `{plan.sha256[:12]}` "
        f"({plan.strategy}, seed {plan.seed}).", "",
        "Each number is the mean ± SD over the five validation folds. Recall and precision here use the default "
        "0.5 cutoff, which is only a placeholder; stage 7 picks the real one. For reference, a model that ranks "
        "people at random gets a PR-AUC equal to the share of positives.", "",
        _markdown(_with_spread(summary, ["model", "weighting"],
                               ["pr_auc", "roc_auc", "brier", "mean_predicted_risk", "recall", "precision"])), "",
        "## Does weighting change who ranks as high risk, or just move the cutoff?", "",
        "Averaged over folds. A Spearman correlation close to 1 means the weighted and unweighted models put "
        "people in almost the same order. A small SD in the log-odds shift means weighting simply slides "
        "everyone's score up by about the same amount, which is the same as using a lower cutoff.", "",
        _markdown(agreement_summary), "",
    ]
    if check is not None:
        text += ["## Do repeated answer profiles inflate the scores? (unweighted models)", "",
                 "Same training rows, two ways of forming the folds: ordinary stratified folds, and folds that "
                 "keep everyone with identical answers together. If repeated profiles were leaking labels, the "
                 "second version would score clearly lower.", "",
                 _markdown(_with_spread(check, ["model", "cv_folds"], ["pr_auc", "roc_auc", "brier"])
                           .assign(pr_auc_drop_vs_plain=check["pr_auc_drop_vs_plain"].round(3) + 0.0)), ""]
    text += ["![Weighting comparison](weighting_comparison.png)", ""]
    (destination / "README.md").write_text("\n".join(text), encoding="utf-8")
    return manifest

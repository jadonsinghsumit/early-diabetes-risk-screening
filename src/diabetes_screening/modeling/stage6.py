"""Command-level runner for roadmap stage 6: tune, compare, interpret, and save the fitted pipelines.

AI-use disclosure: drafted with AI assistance (Claude, Anthropic); see docs/training.md.
"""
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import joblib
import pandas as pd
from sklearn.base import clone

from ..ingestion import write_json
from ..schema import EXPECTED_ROWS
from .cohort import load_cohort
from .imbalance import MODELS
from .inference import odds_ratio_table, plot_odds_ratios
from .metrics import SCREENING_RECALL
from .split import load_split, training_data
from .stage5 import PACKAGES, _markdown, _mean_sd, endpoint_label
from .training import (DEFAULT_ITERATIONS, LABELS, SEARCH_SPACES, SELECTION_RULE, _plain, baseline_results,
                       choose_model, feature_set_results, fold_results, paired_differences, plot_comparison,
                       plot_importance, summarize_folds, tune, tuning_table)


def _readable(summary, metrics):
    table = pd.DataFrame({"model": [LABELS.get(m, m.replace("_", " ")) for m in summary.model]})
    for metric in metrics:
        table[metric] = [_mean_sd(m, s) for m, s in zip(summary[f"{metric}_mean"], summary[f"{metric}_sd"])]
    return table


def run_training(source, provenance, split_directory, destination, models=MODELS, n_iter=None, importance=True,
                 save_models=True, expected_source_rows=EXPECTED_ROWS, log=print):
    """Stage 6 end to end on the training rows of a saved split. The test rows are never loaded."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    cohort = load_cohort(source, provenance, expected_source_rows)
    plan = load_split(split_directory, cohort)
    X, y, folds = training_data(cohort, plan)
    label = endpoint_label(cohort.endpoint)
    log(f"Training rows: {len(y):,} ({int(y.sum()):,} positive). Test rows locked: {len(plan.test_index):,}")

    per_fold = [baseline_results(X, y, folds)]
    tuning, best, estimators, oof, importances, timings = [], {}, {}, {}, [], []
    for name in models:
        started = time.perf_counter()
        log(f"  tuning {name} ({n_iter or DEFAULT_ITERATIONS[name]} candidates x {plan.n_splits} folds)")
        search = tune(name, X, y, plan.cv(), n_iter, plan.seed)
        tuning.append(tuning_table(name, search))
        estimators[name] = search.best_estimator_
        frame, proba, imp = fold_results(lambda y_fit, e=search.best_estimator_: clone(e), X, y, folds, name,
                                         importance, plan.seed, log)
        per_fold.append(frame)
        oof[name], importances = proba, importances + [imp]
        best[name] = {"params": {k: _plain(v) for k, v in search.best_params_.items()},
                      "search_pr_auc_mean": float(search.best_score_),
                      "refit_check_pr_auc_mean": float(frame.pr_auc.mean()),
                      "candidates_tried": len(search.cv_results_["params"])}
        if save_models:
            (destination / "models").mkdir(exist_ok=True)
            joblib.dump(search.best_estimator_, destination / "models" / f"{name}.joblib", compress=3)
        seconds = time.perf_counter() - started
        timings.append({"model": name, "seconds": round(seconds, 1)})
        log(f"  {name}: best PR-AUC {search.best_score_:.4f} ({seconds:.0f}s)")

    per_fold = pd.concat(per_fold, ignore_index=True)
    checklist = summarize_folds(per_fold[per_fold.model == "ada_checklist_5_points"],
                                metrics=["recall", "precision", "specificity", "flagged_share"])
    summary = summarize_folds(per_fold[per_fold.model != "ada_checklist_5_points"])
    paired = paired_differences(per_fold, ["ada_checklist", *models])
    selection = choose_model(per_fold, list(models))
    log("  refitting without healthcare-contact answers")
    reduced = feature_set_results(estimators, X, y, folds)
    feature_sets = _feature_set_table(per_fold, reduced, list(models))
    log("  statsmodels inference")
    odds, fit_summary = odds_ratio_table(X, y)

    per_fold.to_csv(destination / "comparison_folds.csv", index=False)
    summary.to_csv(destination / "comparison.csv", index=False)
    paired.to_csv(destination / "paired_differences.csv", index=False)
    pd.concat(tuning, ignore_index=True).to_csv(destination / "tuning.csv", index=False)
    feature_sets.to_csv(destination / "feature_sets.csv", index=False)
    odds.to_csv(destination / "odds_ratios.csv", index=False)
    write_json(destination / "best_params.json", best)
    checklist.to_csv(destination / "checklist_at_5_points.csv", index=False)
    write_json(destination / "selection.json", {**selection, "rule": SELECTION_RULE})
    plot_comparison(summary, destination / "comparison.png", label)
    plot_odds_ratios(odds, destination / "odds_ratios.png", label)
    importance_table = None
    if importance:
        importance_frame = pd.concat(importances, ignore_index=True)
        importance_table = (importance_frame.groupby(["model", "feature"])["pr_auc_drop"].agg(["mean", "std"])
                            .reset_index().rename(columns={"mean": "pr_auc_drop_mean", "std": "pr_auc_drop_sd"}))
        importance_table.to_csv(destination / "permutation_importance.csv", index=False)
        plot_importance(importance_table, destination / "permutation_importance.png", label)
    # Out-of-fold probabilities for stage 7 (row = position in the prepared CSV). Row-level, so it stays
    # in the local outputs folder and is never committed.
    pd.DataFrame({"row": plan.train_index, "fold": folds, "y": y.to_numpy(), **oof}).to_csv(
        destination / "oof_predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})

    manifest = {
        "stage": 6, "endpoint": cohort.endpoint, "input_sha256": cohort.input_sha256,
        "split_sha256": plan.sha256, "seed": plan.seed, "training_rows_used": int(len(y)), "test_rows_used": 0,
        "models": list(models), "search_spaces": {m: {k: repr(v) for k, v in SEARCH_SPACES[m].items()} for m in models},
        "candidates_per_model": {m: best[m]["candidates_tried"] for m in models},
        "screening_recall_target": SCREENING_RECALL, "class_weighting": "none (stage 5 decision)",
        "timing_seconds": timings, "statsmodels_fit": fit_summary,
        "python": platform.python_version(), "packages": {n: version(n) for n in PACKAGES + ["statsmodels"]},
    }
    write_json(destination / "manifest.json", manifest)
    _write_readme(destination, label, cohort, plan, y, summary, paired, selection, feature_sets, odds,
                  importance_table, best, checklist)
    return manifest


def _feature_set_table(per_fold, reduced, models):
    rows = []
    for name in models:
        full = per_fold[per_fold.model == name].set_index("fold").pr_auc
        less = reduced[reduced.model == name].set_index("fold").pr_auc
        gap = full - less
        rows.append({"model": name, "pr_auc_all_21": full.mean(), "pr_auc_without_contact": less.mean(),
                     "mean_drop": gap.mean(), "sd_drop": gap.std(), "folds_where_full_is_better": int((gap > 0).sum())})
    return pd.DataFrame(rows)


def _write_readme(destination, label, cohort, plan, y, summary, paired, selection, feature_sets, odds,
                  importance_table, best, checklist):
    metrics = ["pr_auc", "roc_auc", "brier", "precision_at_80_recall", "specificity_at_80_recall",
               "flagged_share_at_80_recall"]
    chosen = LABELS.get(selection["chosen"], selection["chosen"])
    text = [
        f"# Stage 6: tuned models ({label})", "",
        f"Everything on this page comes from the {len(y):,} training rows ({int(y.sum()):,} positive) and the five "
        f"saved folds of split `{plan.sha256[:12]}`. The {len(plan.test_index):,} test rows were not loaded.", "",
        "## How the models compare", "",
        "Mean ± SD over the five validation folds. The `_at_80_recall` columns answer a screening question: if we "
        "lower the cutoff until 80% of real cases get flagged, how many of the flagged people actually have the "
        "condition (precision), how many people without it are correctly left alone (specificity), and what share "
        "of everyone ends up flagged? Ignore the checklist's Brier score: points divided by 9 isn't a risk.", "",
        _markdown(_readable(summary, metrics)), "",
        "The paper checklist used the way clinics use it (5 or more points means increased risk):", "",
        _markdown(_readable(checklist.assign(model="ada_checklist"),
                            ["recall", "precision", "specificity", "flagged_share"])), "",
        f"**Chosen for stage 7: {chosen}.** Reason: {selection['reason']}.", "",
        f"The rule, written down before the full run: {SELECTION_RULE}", "",
        "## Fold-by-fold gaps in PR-AUC", "",
        "All models were scored on the same five folds, so we can compare them fold by fold. `p_value` comes from "
        "Nadeau and Bengio's corrected resampled t-test, which allows for the folds sharing most of their training "
        "data. With only five folds it's a cautious test.", "",
        _markdown(paired.assign(model=paired.model.map(LABELS), compared_with=paired.compared_with.map(LABELS))
                  .round(4)), "",
        "## Without the healthcare-contact answers", "",
        "Same tuned settings, refitted without CholCheck, HighBP, HighChol, AnyHealthcare and NoDocbcCost. "
        "These answers partly reflect whether someone sees a doctor, so it's worth knowing how much the models "
        "lean on them.", "",
        _markdown(feature_sets.assign(model=feature_sets.model.map(LABELS))), "",
        "## Tuned settings", "",
        "The refit check re-scores the winning settings with a fresh fit on each fold. It should match the "
        "search's own score, and it does when the run is reproducible.", "",
    ]
    for name, info in best.items():
        text.append(f"- **{LABELS[name]}**: `{json.dumps(info['params'])}` "
                    f"(search PR-AUC {info['search_pr_auc_mean']:.4f}, "
                    f"refit check {info['refit_check_pr_auc_mean']:.4f})")
    text += ["", "## Adjusted odds ratios (statsmodels, training rows only)", "",
             "Unadjusted means one answer at a time; adjusted means all 21 answers in one ordinary logistic "
             "regression. `change_after_adjustment` flags associations that weaken, vanish or reverse once the "
             "other answers are held fixed (our Simpson's-paradox check). VIF above about 5 would mean an answer "
             "overlaps heavily with the others.", "",
             _markdown(odds[["feature", "unit", "unadjusted_or", "adjusted_or", "adjusted_ci_low", "adjusted_ci_high",
                             "change_after_adjustment", "vif"]].round(3)), ""]
    if importance_table is not None:
        top = (importance_table.sort_values("pr_auc_drop_mean", ascending=False).groupby("model").head(5)
               .assign(model=lambda t: t.model.map(LABELS)))
        text += ["## The five answers each model relies on most (permutation importance)", "",
                 "How much validation PR-AUC drops when one answer is shuffled, averaged over the five folds.", "",
                 _markdown(top[["model", "feature", "pr_auc_drop_mean", "pr_auc_drop_sd"]].round(4)), "",
                 "![Permutation importance](permutation_importance.png)", ""]
    text += ["![Model comparison](comparison.png)", "", "![Odds ratios](odds_ratios.png)", ""]
    (destination / "README.md").write_text("\n".join(text), encoding="utf-8")


def load_pipeline(path):
    """Load a fitted pipeline saved by `train` (scikit-learn/XGBoost versions should match the manifest)."""
    return joblib.load(path)

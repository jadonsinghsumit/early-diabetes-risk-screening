"""Stage 6: tune three model families on the saved folds and compare them fairly.

Ground rules (all of them follow from stage 5):
- Every model is tuned and scored on the same five validation folds (`plan.cv()`), using training rows only.
- PR-AUC picks the winner of each search. We also track what matters for screening: if we have
  to catch 80% of cases, how many false alarms does each model cost us?
- No class weighting (stage 5 showed it only moves the cutoff), so predicted probabilities stay
  readable as risks for stage 7.

AI-use disclosure: drafted with AI assistance (Claude, Anthropic); see docs/training.md.
"""
import time

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from scipy.stats import t as student_t
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

from .baselines import ADA_CUTOFF, BASELINES
from .cohort import FEATURE_SETS
from .imbalance import model_factory
from .metrics import at_recall, evaluate, max_f2

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)
from matplotlib.ticker import MaxNLocator  # noqa: E402

# What each search is allowed to try. The ranges stay inside what's sensible for 200k rows and
# 21 coarse answers; the reasoning is in docs/training.md.
SEARCH_SPACES = {
    "logistic_regression": {
        "preprocess__kind": ["linear", "linear-onehot"],  # ordinal codes as numbers, or one column per level
        "model__C": loguniform(1e-3, 1e2),  # inverse L2 strength
    },
    "random_forest": {
        "model__n_estimators": [200, 300],
        "model__max_depth": [None, 12, 20],
        "model__min_samples_leaf": [10, 20, 50, 100],
        "model__max_features": ["sqrt", 0.33],
    },
    "xgboost": {
        "model__n_estimators": [200, 400, 600, 800],
        "model__learning_rate": loguniform(0.02, 0.2),
        "model__max_depth": [3, 4, 5, 6],
        "model__min_child_weight": [1, 5, 10, 25],
        "model__subsample": uniform(0.7, 0.3),
        "model__colsample_bytree": uniform(0.6, 0.4),
        "model__reg_lambda": loguniform(0.5, 20),
    },
}
DEFAULT_ITERATIONS = {"logistic_regression": 12, "random_forest": 8, "xgboost": 16}
SIMPLICITY = ["logistic_regression", "random_forest", "xgboost"]  # simplest first, for tie-breaking
LABELS = {"no_model": "No model (prevalence)", "ada_checklist": "ADA-style checklist",
          "logistic_regression": "Logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost"}


def _precision_at_target(y, proba):
    return at_recall(y, proba)["precision"]


def _specificity_at_target(y, proba):
    return at_recall(y, proba)["specificity"]


SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "brier": "neg_brier_score",
    "precision_at_80_recall": make_scorer(_precision_at_target, response_method="predict_proba"),
    "specificity_at_80_recall": make_scorer(_specificity_at_target, response_method="predict_proba"),
    "max_f2": make_scorer(max_f2, response_method="predict_proba"),
}
COMPARISON_METRICS = ["pr_auc", "roc_auc", "brier", "mean_predicted_risk", "precision_at_80_recall",
                      "specificity_at_80_recall", "flagged_share_at_80_recall", "max_f2"]


def tune(name, X, y, cv, n_iter=None, seed=42):
    """Randomized search over SEARCH_SPACES[name] on the fixed folds; refits the winner on all training rows."""
    search = RandomizedSearchCV(
        model_factory(name, weighted=False, seed=seed)(y), SEARCH_SPACES[name],
        n_iter=n_iter or DEFAULT_ITERATIONS[name], scoring=SCORING, refit="pr_auc", cv=cv,
        random_state=seed, n_jobs=1, error_score="raise")
    search.fit(X, y)
    return search


def tuning_table(name, search):
    """Every candidate the search tried: its settings, mean and SD per metric, and its PR-AUC rank."""
    results = pd.DataFrame(search.cv_results_)
    table = pd.DataFrame({"model": name, "rank_pr_auc": results["rank_test_pr_auc"],
                          "params": results["params"].map(lambda p: {k: _plain(v) for k, v in p.items()}),
                          "mean_fit_seconds": results["mean_fit_time"].round(2)})
    for metric in SCORING:
        sign = -1 if metric == "brier" else 1  # scikit-learn stores the Brier score negated
        table[f"{metric}_mean"] = sign * results[f"mean_test_{metric}"]
        table[f"{metric}_sd"] = results[f"std_test_{metric}"]
    return table.sort_values("rank_pr_auc").reset_index(drop=True)


def _plain(value):
    """Make numpy scalars JSON-friendly and keep floats short."""
    if isinstance(value, (np.floating, float)):
        return round(float(value), 5)
    if isinstance(value, np.integer):
        return int(value)
    return value


def fold_results(build, X, y, folds, label, importance=False, seed=42, log=print):
    """Fit `build(y_fit)` on four folds, score the fifth, five times over.

    Returns per-fold metrics, out-of-fold probabilities and (optionally) permutation importance
    measured on each held-out fold: how much PR-AUC drops when one answer is shuffled.
    """
    X, y, folds = X.reset_index(drop=True), np.asarray(y), np.asarray(folds)
    proba, rows, importances = np.full(len(y), np.nan), [], []
    for fold in np.unique(folds):
        fit, held_out = folds != fold, folds == fold
        model = build(y[fit]).fit(X[fit], y[fit])
        proba[held_out] = model.predict_proba(X[held_out])[:, 1]
        rows.append({"model": label, "fold": int(fold), **evaluate(y[held_out], proba[held_out])})
        if importance:
            started = time.perf_counter()
            # Score the shuffled and unshuffled versions on the same rows (the whole held-out fold).
            # scikit-learn scores the baseline on all rows but shuffled copies on the subsample when
            # max_samples < 1, and PR-AUC on half a fold isn't comparable with PR-AUC on the full fold.
            result = permutation_importance(model, X[held_out], y[held_out], scoring="average_precision",
                                            n_repeats=3, random_state=seed, n_jobs=1)
            importances += [{"model": label, "fold": int(fold), "feature": feature, "pr_auc_drop": drop}
                            for feature, drop in zip(X.columns, result.importances_mean)]
            log(f"    permutation importance, fold {fold}: {time.perf_counter() - started:.0f}s")
    return pd.DataFrame(rows), proba, pd.DataFrame(importances)


def baseline_results(X, y, folds):
    """Per-fold scores for the two yardsticks, plus the checklist at its own 5-point cutoff."""
    frames = []
    for name, build in BASELINES.items():
        frame, proba, _ = fold_results(build, X, y, folds, name)
        frames.append(frame)
        if name == "ada_checklist":
            at_cutoff = [{"model": "ada_checklist_5_points", "fold": int(k),
                          **evaluate(np.asarray(y)[np.asarray(folds) == k], proba[np.asarray(folds) == k],
                                     threshold=ADA_CUTOFF)} for k in np.unique(folds)]
            frames.append(pd.DataFrame(at_cutoff))
    return pd.concat(frames, ignore_index=True)


def summarize_folds(per_fold, metrics=COMPARISON_METRICS, by="model"):
    grouped = per_fold.groupby(by, sort=False)[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{'sd' if stat == 'std' else stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def corrected_t_test(gaps):
    """Nadeau and Bengio's corrected resampled t-test for a k-fold cross-validation difference.

    The five fold scores aren't independent (every pair of folds shares most of its training rows),
    so a plain paired t-test is overconfident. The correction inflates the variance by
    (1/k + n_validation/n_training), which is 1/5 + 1/4 for five folds. Returns (t, two-sided p).
    """
    gaps = np.asarray(gaps, dtype=float)
    k, mean, variance = len(gaps), gaps.mean(), gaps.var(ddof=1)
    if variance == 0:
        return (np.inf if mean else 0.0), (0.0 if mean else 1.0)
    statistic = mean / np.sqrt((1 / k + 1 / (k - 1)) * variance)
    return float(statistic), float(2 * student_t.sf(abs(statistic), df=k - 1))


def paired_differences(per_fold, models, metric="pr_auc"):
    """Fold-by-fold gaps between every pair of models. Same folds, so the comparison is paired."""
    wide = per_fold.pivot(index="fold", columns="model", values=metric)
    rows = []
    for i, first in enumerate(models):
        for second in models[i + 1:]:
            gap = wide[second] - wide[first]
            statistic, p_value = corrected_t_test(gap)
            rows.append({"model": second, "compared_with": first, f"mean_{metric}_gain": gap.mean(),
                         f"sd_{metric}_gain": gap.std(), "folds_won": int((gap > 0).sum()), "folds": len(gap),
                         "corrected_t": statistic, "p_value": p_value})
    return pd.DataFrame(rows)


SIGNIFICANCE = 0.05
SELECTION_RULE = (
    "1) Start from the model with the best mean validation PR-AUC. 2) If logistic regression is not "
    "significantly worse than it, take logistic regression, because it is the easiest to explain. 3) Otherwise, "
    "among the models that are not significantly worse than the best one, take the one with the lowest mean Brier "
    "score, because stage 7 reads its probabilities as risks. 'Significantly worse' means p < 0.05 in Nadeau and "
    "Bengio's corrected resampled t-test on the five fold-by-fold PR-AUC differences.")


def choose_model(per_fold, models, metric="pr_auc"):
    """Apply SELECTION_RULE (written down before the full tuning run) and explain the outcome in words."""
    rows = per_fold[per_fold.model.isin(models)]
    means, brier = rows.groupby("model")[metric].mean(), rows.groupby("model")["brier"].mean()
    best = means.idxmax()
    wide = rows.pivot(index="fold", columns="model", values=metric)

    def versus(other):
        gap = wide[best] - wide[other]
        return gap.mean(), corrected_t_test(gap)[1], int((gap > 0).sum())

    simplest = "logistic_regression"
    if best != simplest and simplest in means.index:
        mean, p_value, _ = versus(simplest)
        if p_value >= SIGNIFICANCE:
            return {"chosen": simplest, "best_mean_pr_auc": best, "tied_within_noise": [best, simplest],
                    "reason": f"{LABELS[best]} beats it by only {mean:.4f} PR-AUC, which is not significant "
                              f"(corrected t-test p = {p_value:.2f}), so the simpler model wins"}
    tied = [m for m in means.index if m == best or versus(m)[1] >= SIGNIFICANCE]
    chosen = min(tied, key=lambda m: brier[m])
    others = [m for m in means.index if m != best]
    if chosen != best:
        mean, p_value, _ = versus(chosen)
        reason = (f"{LABELS[best]} has the highest mean PR-AUC, but only by {mean:.4f} (p = {p_value:.2f}), and "
                  f"{LABELS[chosen]} has better-calibrated probabilities (Brier {brier[chosen]:.4f} vs "
                  f"{brier[best]:.4f})")
    elif len(tied) > 1:
        reason = (f"highest mean validation PR-AUC ({means[best]:.4f}) and the lowest Brier score among the "
                  f"models it can't be told apart from ({', '.join(LABELS[m] for m in tied if m != best)})")
    else:
        reason = f"highest mean validation PR-AUC ({means[best]:.4f})"
        if others:
            runner_up = means[others].idxmax()
            mean, p_value, won = versus(runner_up)
            reason += (f", ahead of {LABELS[runner_up]} by {mean:.4f} in {won} of {len(wide)} folds "
                       f"(corrected t-test p = {p_value:.3f})")
    return {"chosen": chosen, "best_mean_pr_auc": best, "tied_within_noise": tied, "reason": reason}


def feature_set_results(best_estimators, X, y, folds, features="no-healthcare-contact"):
    """Refit each tuned model without the healthcare-contact answers (same settings, same folds)."""
    frames = []
    for name, estimator in best_estimators.items():
        reduced = clone(estimator).set_params(preprocess__features=tuple(FEATURE_SETS[features]))
        frame, _, _ = fold_results(lambda y_fit, e=reduced: clone(e), X, y, folds, name)
        frames.append(frame.assign(feature_set=features))
    return pd.concat(frames, ignore_index=True)


INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
MODEL_COLORS = {"no_model": "#b4b2a9", "ada_checklist": "#8a8880", "logistic_regression": "#2a78d6",
                "random_forest": "#1baf7a", "xgboost": "#eb6834"}  # validated categorical slots 1, 3 and 2


def _style(axis):
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors=MUTED, labelsize=9)


def plot_comparison(summary, destination, endpoint_label):
    """Two panels: ranking quality (PR-AUC) and false-alarm cost at 80% recall (precision)."""
    order = [m for m in ["no_model", "ada_checklist", "logistic_regression", "random_forest", "xgboost"]
             if m in set(summary.model)]
    rows = summary.set_index("model").loc[order]
    panels = [("pr_auc", "PR-AUC (higher = better ranking)"),
              ("precision_at_80_recall", "Precision when catching 80% of cases")]
    figure, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    y_positions = np.arange(len(order))[::-1]
    for axis, (metric, title) in zip(axes, panels):
        axis.barh(y_positions, rows[f"{metric}_mean"], xerr=rows[f"{metric}_sd"], height=0.6,
                  color=[MODEL_COLORS[m] for m in order], error_kw={"elinewidth": 1, "ecolor": INK})
        for y_pos, value, spread in zip(y_positions, rows[f"{metric}_mean"], rows[f"{metric}_sd"].fillna(0)):
            axis.annotate(f"{value:.3f}", (value + spread, y_pos), xytext=(6, 0), textcoords="offset points",
                          va="center", fontsize=9, color=INK)
        axis.set_title(title, fontsize=10.5, color=INK, loc="left")
        axis.set_xlim(0, max(rows[f"{metric}_mean"]) * 1.25)
        _style(axis)
    axes[0].set_yticks(y_positions, [LABELS[m] for m in order])
    figure.suptitle(f"Model comparison on validation folds ({endpoint_label}); mean of 5 folds, whiskers = SD",
                    fontsize=11, color=INK, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_importance(table, destination, endpoint_label):
    """Permutation importance per model, features in the same order in every panel.

    `table` has one row per model and feature with `pr_auc_drop_mean` and `pr_auc_drop_sd`, the same
    table that's saved as permutation_importance.csv, so the chart can be redrawn from that file.
    """
    order = table.groupby("feature")["pr_auc_drop_mean"].mean().sort_values().index.tolist()
    models = [m for m in SIMPLICITY if m in set(table.model)]
    figure, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 6.2), sharey=True, squeeze=False)
    for axis, name in zip(axes[0], models):
        rows = table[table.model == name].set_index("feature").loc[order]
        axis.barh(np.arange(len(order)), rows["pr_auc_drop_mean"], xerr=rows["pr_auc_drop_sd"], height=0.65,
                  color=MODEL_COLORS[name], error_kw={"elinewidth": 0.8, "ecolor": INK})
        axis.axvline(0, color=MUTED, linewidth=0.8)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=4))  # few ticks, so labels never run together
        axis.set_title(LABELS[name], fontsize=10.5, color=INK, loc="left")
        axis.set_xlabel("Drop in PR-AUC when the answer is shuffled", fontsize=8.5, color=MUTED)
        _style(axis)
    axes[0][0].set_yticks(np.arange(len(order)), order)
    figure.suptitle(f"Which answers the tuned models rely on ({endpoint_label}); mean of 5 validation folds",
                    fontsize=11, color=INK, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)

"""Stage 5 evidence: does class weighting help, and does profile overlap inflate scores?

Everything here runs on the training rows only. Each fold fits a fresh pipeline whose scaler and
class weights are learned from that fold's training part, then scores the held-out fold.
"""
import time

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .metrics import evaluate
from .preprocess import ScreeningPreprocessor
from .split import assign_folds

matplotlib.use("Agg")  # headless: charts are written to files
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)

MODELS = ("logistic_regression", "random_forest", "xgboost")
SUMMARY_METRICS = ["pr_auc", "roc_auc", "brier", "mean_predicted_risk", "prevalence", "recall", "precision",
                   "specificity", "f2", "flagged_share"]


def model_factory(name, weighted, seed=42, features=None):
    """Return build(y_fit) -> unfitted Pipeline with fixed, untuned settings (stage 6 tunes them).

    Weighting: class_weight="balanced" for logistic regression and random forest; for XGBoost,
    scale_pos_weight = negatives / positives counted in the fold's own training labels.
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {MODELS}")

    def build(y_fit):
        y_fit = np.asarray(y_fit)
        if name == "logistic_regression":
            kind = "linear"
            estimator = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced" if weighted else None)
        elif name == "random_forest":
            kind = "tree"
            estimator = RandomForestClassifier(
                n_estimators=200, min_samples_leaf=20, max_features="sqrt", n_jobs=-1, random_state=seed,
                class_weight="balanced" if weighted else None)
        else:
            kind = "tree"
            ratio = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1)) if weighted else 1.0
            estimator = XGBClassifier(
                n_estimators=300, learning_rate=0.1, max_depth=4, tree_method="hist", eval_metric="logloss",
                scale_pos_weight=ratio, random_state=seed, n_jobs=-1)
        return Pipeline([("preprocess", ScreeningPreprocessor(kind, features)), ("model", estimator)])

    return build


def out_of_fold(build, X, y, folds, keep_models=False):
    """Fit one fresh pipeline per fold on the other folds; return held-out probabilities.

    Every row receives a prediction from a model that never saw it. Stage 7 reuses this to choose
    the threshold on training data instead of the test set.
    """
    X, y, folds = X.reset_index(drop=True), np.asarray(y), np.asarray(folds)
    proba, models = np.full(len(y), np.nan), []
    for fold in np.unique(folds):
        fit, held_out = folds != fold, folds == fold
        model = build(y[fit])
        model.fit(X[fit], y[fit])
        proba[held_out] = model.predict_proba(X[held_out])[:, 1]
        if keep_models:
            models.append(model)
    return proba, models


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _per_fold(y, proba, folds, **labels):
    return [{**labels, "fold": int(k), **evaluate(y[folds == k], proba[folds == k])} for k in np.unique(folds)]


def compare_weighting(X, y, folds, models=MODELS, seed=42, log=print):
    """Cross-validated metrics for every model with and without class weighting.

    Returns (per_fold, agreement, timing). `agreement` tests the claim that weighting mostly moves
    probabilities rather than re-ranking people: per fold, the Spearman correlation between the
    weighted and unweighted scores, and the mean and spread of the shift in log-odds. A spread near
    zero means weighting is equivalent to choosing a different threshold.
    """
    y, folds = np.asarray(y), np.asarray(folds)
    per_fold, agreement, timing = [], [], []
    for name in models:
        scores = {}
        for weighted in (False, True):
            started = time.perf_counter()
            scores[weighted], _ = out_of_fold(model_factory(name, weighted, seed), X, y, folds)
            seconds = time.perf_counter() - started
            weighting = "weighted" if weighted else "unweighted"
            timing.append({"model": name, "weighting": weighting, "seconds_all_folds": round(seconds, 1)})
            per_fold += _per_fold(y, scores[weighted], folds, model=name, weighting=weighting)
            log(f"  {name} ({weighting}): {seconds:.0f}s")
        for k in np.unique(folds):
            held = folds == k
            shift = _logit(scores[True][held]) - _logit(scores[False][held])
            agreement.append({"model": name, "fold": int(k),
                              "spearman_weighted_vs_unweighted": float(spearmanr(scores[True][held],
                                                                                 scores[False][held])[0]),
                              "mean_logit_shift": float(shift.mean()), "sd_logit_shift": float(shift.std())})
    return pd.DataFrame(per_fold), pd.DataFrame(agreement), pd.DataFrame(timing)


def summarize(per_fold, by=("model", "weighting"), metrics=SUMMARY_METRICS):
    """Mean and standard deviation across folds, one row per group, columns `<metric>_mean/_sd`."""
    grouped = per_fold.groupby(list(by), sort=False)[metrics].agg(["mean", "std"])
    grouped.columns = [f"{metric}_{'sd' if stat == 'std' else stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def grouped_cv_check(X, y, plain_folds, profile_ids, models=MODELS, seed=42, plain_per_fold=None, log=print):
    """Compare plain stratified folds with folds that keep identical answer profiles together.

    If scores drop sharply with grouped folds, models are scoring well partly by recognising answer
    profiles they have already seen. Only unweighted models are compared; weighting is a separate question.
    """
    y = np.asarray(y)
    grouped_folds = assign_folds(y, len(np.unique(plain_folds)), seed, groups=np.asarray(profile_ids))
    rows = []
    for name in models:
        if plain_per_fold is not None:
            plain = plain_per_fold[(plain_per_fold.model == name) & (plain_per_fold.weighting == "unweighted")]
            rows += plain.assign(cv_folds="plain stratified").to_dict("records")
        else:
            proba, _ = out_of_fold(model_factory(name, False, seed), X, y, plain_folds)
            rows += _per_fold(y, proba, np.asarray(plain_folds), model=name, weighting="unweighted",
                              cv_folds="plain stratified")
        started = time.perf_counter()
        proba, _ = out_of_fold(model_factory(name, False, seed), X, y, grouped_folds)
        rows += _per_fold(y, proba, grouped_folds, model=name, weighting="unweighted", cv_folds="profile-grouped")
        log(f"  {name} (profile-grouped folds): {time.perf_counter() - started:.0f}s")
    table = summarize(pd.DataFrame(rows), by=("model", "cv_folds"), metrics=["pr_auc", "roc_auc", "brier"])
    plain = table[table.cv_folds == "plain stratified"].set_index("model")
    grouped = table[table.cv_folds == "profile-grouped"].set_index("model")
    table["pr_auc_drop_vs_plain"] = table.apply(
        lambda row: plain.loc[row.model, "pr_auc_mean"] - grouped.loc[row.model, "pr_auc_mean"]
        if row.cv_folds == "profile-grouped" else 0.0, axis=1)
    return table


LABELS = {"logistic_regression": "Logistic\nregression", "random_forest": "Random\nforest",
          "xgboost": "XGBoost"}
COLORS = {"unweighted": "#2a78d6", "weighted": "#eb6834"}  # categorical slots 1 and 2 (validated palette)
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def plot_weighting(summary, destination, endpoint_label):
    """Three panels: ranking quality (PR-AUC), recall at the default 0.5 cutoff, and mean predicted risk."""
    panels = [("pr_auc", "PR-AUC (ranking quality)", "Dashed: random ranking (= prevalence)"),
              ("recall", "Recall at the default 0.5 cutoff", "Share of true cases flagged"),
              ("mean_predicted_risk", "Average predicted risk", "Dashed: actual prevalence")]
    models = list(dict.fromkeys(summary["model"]))
    x = np.arange(len(models))
    width = 0.36
    prevalence = float(summary["prevalence_mean"].iloc[0])
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, (metric, title, note) in zip(axes, panels):
        for offset, weighting in ((-width / 2, "unweighted"), (width / 2, "weighted")):
            rows = summary.set_index(["model", "weighting"]).loc[[(m, weighting) for m in models]]
            bars = axis.bar(x + offset, rows[f"{metric}_mean"], width - 0.04, yerr=rows[f"{metric}_sd"],
                            color=COLORS[weighting], hatch="//" if weighting == "weighted" else None,
                            edgecolor="white", linewidth=0, error_kw={"elinewidth": 1, "ecolor": INK},
                            label="Class-weighted" if weighting == "weighted" else "No weighting")
            for bar, value in zip(bars, rows[f"{metric}_mean"]):
                label = f"{value:.3f}" if 0 < value < 0.1 else f"{value:.2f}"  # keep small PR-AUCs readable
                axis.annotate(label, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                              xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                              fontsize=8, color=INK)
        if metric in {"pr_auc", "mean_predicted_risk"}:
            axis.axhline(prevalence, color=MUTED, linestyle="--", linewidth=1)
        axis.set_xticks(x, [LABELS.get(m, m) for m in models])
        axis.set_ylim(0, 1.08 if metric == "recall" else max(0.05, axis.get_ylim()[1] * 1.12))
        axis.set_title(title, fontsize=10.5, color=INK, loc="left")
        axis.set_xlabel(note, fontsize=8.5, color=MUTED)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.tick_params(colors=MUTED, labelsize=8.5)
    figure.suptitle(f"Class weighting on validation folds ({endpoint_label}); bars = mean of 5 folds, "
                    "whiskers = SD", fontsize=11, color=INK, x=0.01, y=0.995, ha="left")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, fontsize=9, ncol=2, loc="upper left",
                  bbox_to_anchor=(0.005, 0.945))
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)

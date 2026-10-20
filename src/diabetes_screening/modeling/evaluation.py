"""Stage 7 building blocks: calibration, operating points, bootstrap intervals, subgroups and misses.

Everything here is a plain function of labels, predicted risks and a cutoff, so the same code scores
the out-of-fold predictions (when the cutoff is chosen) and the locked test set (when it is checked).
"""
import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from statsmodels.stats.proportion import proportion_confint

from .metrics import SCREENING_RECALL, at_recall, evaluate
from .training import GRID, INK, LABELS, MODEL_COLORS, MUTED

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)
from matplotlib.patches import Rectangle  # noqa: E402

TARGET_RECALLS = (0.70, 0.75, 0.80, 0.85, 0.90)
THRESHOLD_RULE = ("The highest cutoff on the selected model's out-of-fold predicted risks (training rows only) that "
                  "still flags at least {recall:.0%} of the training cases. The highest such cutoff flags the fewest "
                  "people, so it is the cheapest way to meet the target. Fixed before any test row is read.")
SLOPE_RANGE, MAX_INTERCEPT = (0.9, 1.1), 0.1
RECALIBRATION_RULE = (
    "Recalibrate (Platt scaling: a logistic regression of the outcome on the model's out-of-fold log-odds) only if "
    "the calibration slope is outside 0.9-1.1 or the calibration-in-the-large intercept is further than 0.1 from 0. "
    "Recalibration is monotone, so it never changes who is flagged, only the risk a person is told.")
BOOTSTRAP_METRICS = ["recall", "precision", "specificity", "f2", "flagged_share", "pr_auc", "roc_auc", "brier"]

# Bands wide enough that every test-set subgroup holds hundreds of cases (codes from schema.py).
SUBGROUPS = {
    "Age": ("Age", [("18-44", range(1, 6)), ("45-64", range(6, 10)), ("65+", range(10, 14))]),
    "Sex": ("Sex", [("Female", [0]), ("Male", [1])]),
    "Income": ("Income", [("Under $25k", range(1, 5)), ("$25k-$50k", range(5, 7)), ("$50k+", range(7, 9))]),
}
# Readable summaries of the answers that drive the models, for the "who gets missed" table.
PROFILE = [
    ("BMI (mean)", lambda X: X["BMI"]),
    ("Obese, BMI 30+ (share)", lambda X: X["BMI"] >= 30),
    ("General health fair or poor (share)", lambda X: X["GenHlth"] >= 4),
    ("Age under 45 (share)", lambda X: X["Age"] <= 5),
    ("Age 65+ (share)", lambda X: X["Age"] >= 10),
    ("High blood pressure (share)", lambda X: X["HighBP"] == 1),
    ("High cholesterol (share)", lambda X: X["HighChol"] == 1),
    ("No physical activity (share)", lambda X: X["PhysActivity"] == 0),
    ("Difficulty walking (share)", lambda X: X["DiffWalk"] == 1),
    ("No cholesterol check in 5 years (share)", lambda X: X["CholCheck"] == 0),
    ("No health coverage (share)", lambda X: X["AnyHealthcare"] == 0),
]
ORIGINAL_CLASSES = {0: "No diabetes", 1: "Prediabetes", 2: "Diabetes"}


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_summary(y_true, proba):
    """How far predicted risks sit from what happened.

    Calibration slope: a logistic regression of the outcome on the model's log-odds. 1 is ideal; below 1
    means the risks are too extreme, above 1 too timid. Calibration-in-the-large: the same regression with
    the slope held at 1 (log-odds as an offset); 0 means the risks are right on average. The slope fit is
    also Platt scaling, so its two coefficients are what recalibration would apply. If everyone gets the
    same risk there is no slope to estimate, and both are None.
    """
    y = np.asarray(y_true).astype(int)
    log_odds = _logit(proba)
    binomial = sm.families.Binomial()
    in_the_large = sm.GLM(y, np.ones((len(y), 1)), family=binomial, offset=log_odds).fit()
    slope, platt = None, None
    if np.ptp(log_odds) > 0:
        fit = sm.GLM(y, sm.add_constant(log_odds, has_constant="add"), family=binomial).fit()
        slope, platt = float(fit.params[1]), {"intercept": float(fit.params[0]), "slope": float(fit.params[1])}
    return {"rows": int(len(y)), "observed_rate": float(y.mean()), "mean_predicted_risk": float(np.mean(proba)),
            "brier": float(np.mean((np.asarray(proba, dtype=float) - y) ** 2)),
            "calibration_intercept": float(in_the_large.params[0]), "calibration_slope": slope, "platt": platt}


def needs_recalibration(summary):
    low, high = SLOPE_RANGE
    slope = summary["calibration_slope"]
    return (slope is not None and not low <= slope <= high) or abs(summary["calibration_intercept"]) > MAX_INTERCEPT


def recalibrate(proba, platt):
    """Apply Platt scaling fitted on out-of-fold predictions; `platt=None` leaves the risks alone."""
    if not platt:
        return np.asarray(proba, dtype=float)
    return 1 / (1 + np.exp(-(platt["intercept"] + platt["slope"] * _logit(proba))))


def reliability_table(y_true, proba, bins=10):
    """Mean predicted risk against the observed rate, in tenths of the predicted-risk distribution."""
    frame = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(proba, dtype=float)})
    # Ranking first gives equal-sized bins even when many people share a predicted risk.
    frame["bin"] = pd.qcut(frame.p.rank(method="first"), min(bins, len(frame)), labels=False)
    return (frame.groupby("bin")
            .agg(rows=("y", "size"), positives=("y", "sum"), mean_predicted=("p", "mean"),
                 observed_rate=("y", "mean"), lowest_risk=("p", "min"), highest_risk=("p", "max"))
            .reset_index())


def f2_cutoff(y_true, proba):
    """The cutoff with the best F2 score (recall counts four times as much as precision)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    with np.errstate(divide="ignore", invalid="ignore"):
        f2 = np.where(precision + recall > 0, 5 * precision * recall / (4 * precision + recall), 0)
    return float(thresholds[int(np.nanargmax(f2[:-1]))])  # the curve's last point has no cutoff


def per_thousand(result):
    """Confusion counts rescaled to 1,000 people screened."""
    scale = 1000 / (result["tp"] + result["fp"] + result["tn"] + result["fn"])
    return {"flagged_per_1000": (result["tp"] + result["fp"]) * scale,
            "cases_per_1000": (result["tp"] + result["fn"]) * scale,
            "cases_found_per_1000": result["tp"] * scale, "cases_missed_per_1000": result["fn"] * scale,
            "false_alarms_per_1000": result["fp"] * scale}


AT_CUTOFF = ["threshold", "recall", "precision", "specificity", "f2", "flagged_share", "tp", "fp", "tn", "fn"]


def at_cutoff(y_true, proba, cutoff):
    """Everything that depends on the cutoff, plus the per-1,000 workload."""
    result = evaluate(y_true, proba, threshold=cutoff)
    return {**{key: result[key] for key in AT_CUTOFF}, **per_thousand(result)}


def operating_points(y_true, proba, locked_recall=SCREENING_RECALL, recalls=TARGET_RECALLS):
    """The cutoffs we could have chosen, with what each one costs. Only `locked_recall` is used."""
    rows = []
    for target in sorted(set(recalls) | {locked_recall}):
        cutoff = at_recall(y_true, proba, target)["threshold"]
        rows.append({"rule": f"catch {target:.0%} of cases", "locked": bool(np.isclose(target, locked_recall)),
                     **at_cutoff(y_true, proba, cutoff)})
    rows.append({"rule": "best F2", "locked": False, **at_cutoff(y_true, proba, f2_cutoff(y_true, proba))})
    return pd.DataFrame(rows)


def bootstrap_intervals(y_true, proba, threshold, n_resamples=1000, seed=42, level=0.95):
    """Percentile intervals from resampling respondents with replacement. The cutoff stays fixed,
    because it was locked in advance; only the test rows are uncertain."""
    y, p = np.asarray(y_true).astype(int), np.asarray(proba, dtype=float)
    flagged = p >= threshold
    rng = np.random.default_rng(seed)
    draws = np.full((n_resamples, len(BOOTSTRAP_METRICS)), np.nan)
    for i in range(n_resamples):
        rows = rng.integers(0, len(y), len(y))
        yb, pb, fb = y[rows], p[rows], flagged[rows]
        positives = int(yb.sum())
        negatives = len(yb) - positives
        tp, fp = int((fb & (yb == 1)).sum()), int((fb & (yb == 0)).sum())
        fn = positives - tp
        both_classes = 0 < positives < len(yb)
        draws[i] = [tp / positives if positives else np.nan, tp / (tp + fp) if tp + fp else np.nan,
                    1 - fp / negatives if negatives else np.nan, 5 * tp / (5 * tp + 4 * fn + fp) if tp else 0.0,
                    fb.mean(), average_precision_score(yb, pb) if both_classes else np.nan,
                    roc_auc_score(yb, pb) if both_classes else np.nan, np.mean((pb - yb) ** 2)]
    tail = (1 - level) / 2 * 100
    low, high = np.nanpercentile(draws, [tail, 100 - tail], axis=0)
    return {metric: (float(a), float(b)) for metric, a, b in zip(BOOTSTRAP_METRICS, low, high)}


def _wilson(successes, total):
    return proportion_confint(successes, total, method="wilson") if total else (np.nan, np.nan)


def subgroup_table(X, y_true, proba, threshold):
    """Recall and false-alarm rate per subgroup at the locked cutoff, with counts and Wilson 95% intervals."""
    y, flagged = np.asarray(y_true).astype(int), np.asarray(proba) >= threshold
    rows = []
    for group, (column, levels) in SUBGROUPS.items():
        values = X[column].to_numpy()
        for level, codes in levels:
            member = np.isin(values, list(codes))
            cases, others = member & (y == 1), member & (y == 0)
            tp, fp = int((flagged & cases).sum()), int((flagged & others).sum())
            n_cases, n_others = int(cases.sum()), int(others.sum())
            recall_ci, fpr_ci = _wilson(tp, n_cases), _wilson(fp, n_others)
            rows.append({"group": group, "level": level, "rows": int(member.sum()), "positives": n_cases,
                         "prevalence": n_cases / max(int(member.sum()), 1),
                         "flagged_share": float(flagged[member].mean()) if member.any() else np.nan,
                         "recall": tp / n_cases if n_cases else np.nan,
                         "recall_ci_low": recall_ci[0], "recall_ci_high": recall_ci[1],
                         "false_positive_rate": fp / n_others if n_others else np.nan,
                         "false_positive_rate_ci_low": fpr_ci[0], "false_positive_rate_ci_high": fpr_ci[1],
                         "precision": tp / (tp + fp) if tp + fp else np.nan})
    return pd.DataFrame(rows)


def missed_case_profile(X, y_true, proba, threshold):
    """Average answers of the cases the model caught, the cases it missed, and everyone without the condition."""
    y, p = np.asarray(y_true).astype(int), np.asarray(proba, dtype=float)
    flagged = p >= threshold
    groups = {"caught_cases": (y == 1) & flagged, "missed_cases": (y == 1) & ~flagged, "non_cases": y == 0}
    X = X.reset_index(drop=True)

    def means(values):
        return {name: float(values[mask].mean()) if mask.any() else np.nan for name, mask in groups.items()}

    rows = [{"answer": "People", **{name: int(mask.sum()) for name, mask in groups.items()}},
            {"answer": "Predicted risk (mean)", **means(p)}]
    rows += [{"answer": label, **means(summary(X).astype(float).to_numpy())} for label, summary in PROFILE]
    return pd.DataFrame(rows)


def flagged_by_original_class(original, proba, threshold):
    """Share flagged within each original survey class (no diabetes, prediabetes, diabetes).

    For the two positive classes this is recall; for 'no diabetes' it is the false-alarm rate.
    """
    original, flagged = np.asarray(original), np.asarray(proba) >= threshold
    rows = []
    for code, label in ORIGINAL_CLASSES.items():
        member = original == code
        if member.any():
            count, total = int(flagged[member].sum()), int(member.sum())
            low, high = _wilson(count, total)
            rows.append({"original_class": code, "label": label, "rows": total, "flagged": count,
                         "not_flagged": total - count, "flagged_share": count / total,
                         "flagged_share_ci_low": low, "flagged_share_ci_high": high})
    return pd.DataFrame(rows)


DRAW_ORDER = ["ada_checklist", "logistic_regression", "random_forest", "xgboost"]  # selected model ends on top
SURFACE = "#f1f0ec"


def _axes_style(axis, grid="both"):
    axis.grid(axis=grid, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.tick_params(colors=MUTED, labelsize=9)


def _dot(color):
    return {"marker": "o", "markersize": 9, "color": color, "markeredgecolor": "white", "markeredgewidth": 2,
            "linestyle": "none"}


def plot_curves(y_true, scores, points, destination, title):
    """Precision-recall and ROC curves, with a dot at each model's locked cutoff."""
    y = np.asarray(y_true).astype(int)
    figure, (pr_axis, roc_axis) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    top = 0.0
    for name in [m for m in DRAW_ORDER if m in scores]:
        color = MODEL_COLORS[name]
        precision, recall, _ = precision_recall_curve(y, scores[name])
        top = max(top, precision[recall >= 0.05].max())  # ignore the noisy spike at the first few flags
        fpr, tpr, _ = roc_curve(y, scores[name])
        pr_axis.plot(recall, precision, color=color, linewidth=2, label=LABELS[name])
        roc_axis.plot(fpr, tpr, color=color, linewidth=2, label=LABELS[name])
        pr_axis.plot(points[name]["recall"], points[name]["precision"], **_dot(color))
        roc_axis.plot(1 - points[name]["specificity"], points[name]["recall"], **_dot(color))
    prevalence = y.mean()
    pr_axis.axhline(prevalence, color=MUTED, linestyle="--", linewidth=1)
    pr_axis.annotate(f"No model ({prevalence:.1%})", (0.02, prevalence), xytext=(0, 4), textcoords="offset points",
                     ha="left", fontsize=8.5, color=MUTED)  # left end: the curves are high there, so no overlap
    roc_axis.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
    roc_axis.annotate("Random ranking", (0.62, 0.58), fontsize=8.5, color=MUTED, rotation=37)
    for axis, name, xlabel, ylabel in [
            (pr_axis, "Precision-recall", "Recall: share of cases flagged", "Precision: share of flags that are cases"),
            (roc_axis, "ROC", "False-alarm rate: share of non-cases flagged", "Recall: share of cases flagged")]:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.set_title(name, fontsize=10.5, color=INK, loc="left")
        axis.set_xlabel(xlabel, fontsize=9, color=MUTED)
        axis.set_ylabel(ylabel, fontsize=9, color=MUTED)
        _axes_style(axis)
    # Zoom precision to the data: with a rare outcome it never gets near 1, and a 0-1 axis flattens the curves.
    pr_axis.set_ylim(0, min(1.02, top * 1.25))
    pr_axis.legend(frameon=False, fontsize=9, loc="upper right")
    figure.suptitle(f"{title}; dots = each model's locked cutoff", fontsize=11, color=INK, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_calibration(series, destination, title):
    """Observed rate against mean predicted risk; `series` is a list of (label, color, reliability_table)."""
    top = max(max(t.mean_predicted.max(), t.observed_rate.max()) for _, _, t in series) * 1.08
    figure, axis = plt.subplots(figsize=(5.8, 5.4))
    axis.plot([0, top], [0, top], color=MUTED, linestyle="--", linewidth=1, label="Perfect calibration")
    for label, color, table in series:
        axis.plot(table.mean_predicted, table.observed_rate, linewidth=2, color=color, label=label,
                  marker="o", markersize=8, markeredgecolor="white", markeredgewidth=2)
    axis.set_xlim(0, top)
    axis.set_ylim(0, top)
    axis.set_xlabel("Mean predicted risk (each dot = a tenth of people, by predicted risk)", fontsize=9, color=MUTED)
    axis.set_ylabel("Observed share with the condition", fontsize=9, color=MUTED)
    axis.legend(frameon=False, fontsize=9, loc="upper left")
    _axes_style(axis)
    axis.set_title(title, fontsize=10.5, color=INK, loc="left")
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_subgroups(table, overall, destination, title, color):
    """Recall and false-alarm rate per subgroup (Wilson 95% intervals); the dashed line is everyone."""
    rows = table.reset_index(drop=True)
    y_positions = np.arange(len(rows))[::-1]
    # Escape "$": matplotlib reads text between two dollar signs as a formula ("$25k-$50k").
    labels = [f"{r.group} {r.level}  ({r.rows:,} people, {r.positives:,} cases)".replace("$", r"\$")
              for r in rows.itertuples()]
    figure, axes = plt.subplots(1, 2, figsize=(12, 0.45 * len(rows) + 1.8), sharey=True)
    panels = [("recall", "Recall: share of cases flagged"),
              ("false_positive_rate", "False-alarm rate: share of non-cases flagged")]
    for axis, (metric, name) in zip(axes, panels):
        value = rows[metric]
        # Wilson bounds at 0% or 100% can land a rounding error inside the estimate; clip that away.
        errors = np.clip([value - rows[f"{metric}_ci_low"], rows[f"{metric}_ci_high"] - value], 0, None)
        axis.errorbar(value, y_positions, xerr=errors, ecolor=INK, elinewidth=1, capsize=0, **_dot(color))
        axis.axvline(overall[metric], color=MUTED, linestyle="--", linewidth=1)
        axis.annotate(f"All: {overall[metric]:.1%}", (overall[metric], y_positions.max() + 0.45), xytext=(4, 0),
                      textcoords="offset points", fontsize=8.5, color=MUTED, va="center")
        axis.set_xlim(0, 1)
        axis.set_ylim(-0.6, y_positions.max() + 0.8)
        axis.set_title(name, fontsize=10.5, color=INK, loc="left")
        _axes_style(axis, grid="x")
    axes[0].set_yticks(y_positions, labels)
    figure.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_confusion(result, destination, title):
    """The 2 x 2 table at the locked cutoff, as counts and per 1,000 people screened."""
    total = result["tp"] + result["fp"] + result["tn"] + result["fn"]
    cells = [[("Cases found", result["tp"]), ("Cases missed", result["fn"])],
             [("False alarms", result["fp"]), ("Correctly not flagged", result["tn"])]]
    figure, axis = plt.subplots(figsize=(6.6, 3.8))
    for r, row in enumerate(cells):
        for c, (name, count) in enumerate(row):
            axis.add_patch(Rectangle((c, 1 - r), 1, 1, facecolor=SURFACE, edgecolor="white", linewidth=4))
            axis.text(c + 0.5, 1.62 - r, f"{count:,}", ha="center", va="center", fontsize=16, color=INK)
            axis.text(c + 0.5, 1.32 - r, f"{name}\n{1000 * count / total:.0f} per 1,000 screened", ha="center",
                      va="center", fontsize=9, color=MUTED)
    axis.set_xlim(0, 2)
    axis.set_ylim(0, 2)
    axis.set_xticks([0.5, 1.5], ["Flagged", "Not flagged"])
    axis.xaxis.tick_top()
    axis.set_yticks([1.5, 0.5], ["Has the condition", "Does not"])
    axis.tick_params(length=0, labelsize=9.5, colors=INK)
    for side in axis.spines.values():
        side.set_visible(False)
    axis.set_title(title, fontsize=10.5, color=INK, loc="left", pad=28)
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)

"""One scoring function for every model, fold and stage, so the numbers always line up.

AI-use disclosure: drafted with AI assistance (Claude, Anthropic); see docs/splitting.md.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, fbeta_score,
                             log_loss, precision_recall_curve, roc_auc_score)

SCREENING_RECALL = 0.80  # the plan's example target: catch at least 8 in 10 cases (stage 7 confirms it)


def at_recall(y_true, proba, min_recall=SCREENING_RECALL):
    """The screening operating point: the highest cutoff that still catches `min_recall` of cases.

    The highest such cutoff flags the fewest people, so it's the cheapest way to hit the target.
    Returns the cutoff and what it costs: precision, specificity and the share of people flagged.
    """
    y_true, proba = np.asarray(y_true).astype(int), np.asarray(proba, dtype=float)
    positives = int(y_true.sum())
    order = np.argsort(-proba, kind="mergesort")  # highest risk first
    caught = np.cumsum(y_true[order])  # cases caught if we flag the top 1, 2, 3, ... people
    needed = np.ceil(min_recall * positives - 1e-9)
    cutoff = proba[order][int(np.argmax(caught >= needed))] if positives else np.inf
    flagged = proba >= cutoff  # ties at the cutoff are all flagged, as a real cutoff would do
    tp = int((flagged & (y_true == 1)).sum())
    fp = int((flagged & (y_true == 0)).sum())
    negatives = len(y_true) - positives
    return {"threshold": float(cutoff), "recall": tp / positives if positives else float("nan"),
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "specificity": 1 - fp / negatives if negatives else float("nan"),
            "flagged_share": float(flagged.mean())}


def max_f2(y_true, proba):
    """Best F2 over all cutoffs. F2 weights recall four times as heavily as precision."""
    precision, recall, _ = precision_recall_curve(y_true, proba)
    with np.errstate(divide="ignore", invalid="ignore"):
        f2 = np.where(precision + recall > 0, 5 * precision * recall / (4 * precision + recall), 0)
    return float(np.nanmax(f2))


def evaluate(y_true, proba, threshold=0.5, min_recall=SCREENING_RECALL):
    """Ranking and probability metrics, the 80%-recall operating point, and counts at `threshold`.

    PR-AUC (average precision) is our main ranking metric for an imbalanced screening problem; a
    model that ranks people at random scores about the prevalence. Brier score and mean predicted
    risk tell us whether the probabilities can be read as risks. Everything at `threshold` depends
    on the cutoff, which stage 7 chooses; the `*_at_80_recall` numbers compare models at the same
    screening target without choosing a cutoff for them.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    flagged = proba >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, flagged, labels=[0, 1]).ravel()
    target = at_recall(y_true, proba, min_recall)

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else float("nan")

    return {
        "rows": int(len(y_true)), "positives": int(y_true.sum()), "prevalence": float(y_true.mean()),
        "mean_predicted_risk": float(proba.mean()),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "max_f2": max_f2(y_true, proba),
        "precision_at_80_recall": target["precision"], "specificity_at_80_recall": target["specificity"],
        "flagged_share_at_80_recall": target["flagged_share"], "threshold_at_80_recall": target["threshold"],
        "threshold": float(threshold), "flagged_share": float(flagged.mean()),
        "recall": ratio(tp, tp + fn), "precision": ratio(tp, tp + fp), "specificity": ratio(tn, tn + fp),
        "f2": float(fbeta_score(y_true, flagged, beta=2, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }

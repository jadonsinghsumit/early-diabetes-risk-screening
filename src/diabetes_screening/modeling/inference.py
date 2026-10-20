"""Stage 6 inference with statsmodels: adjusted odds ratios, intervals and collinearity.

The scikit-learn models are tuned to predict well. This module asks a different question: holding
the other answers fixed, how is each answer associated with the outcome? We fit an ordinary
(unpenalized) logistic regression on the training rows only and compare each adjusted odds ratio with
the one-answer-at-a-time (unadjusted) version. A flip in direction is a Simpson's-paradox-style warning
that the simple association was driven by something else, such as age.
"""
import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ..schema import FIELDS
from .cohort import FEATURES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

KIND = {field.name: field.kind for field in FIELDS}
UNIT = {"binary": "yes vs no", "numeric": "per 1 unit", "ordinal": "per 1 code step"}
UNIT_OVERRIDES = {"Sex": "male vs female", "BMI": "per 1 kg/m²", "Age": "per 5-year age band",
                  "MentHlth": "per 1 day", "PhysHlth": "per 1 day", "GenHlth": "per step toward 'poor'"}


def _logit(y, design):
    return sm.Logit(np.asarray(y, dtype=float), design).fit(disp=0, maxiter=200)


def odds_ratio_table(X, y, features=FEATURES):
    """Adjusted (all answers together) and unadjusted (one at a time) odds ratios with 95% Wald CIs."""
    X = X[list(features)].astype(float)
    full = _logit(y, sm.add_constant(X, has_constant="add"))
    adjusted_ci = full.conf_int()
    rows = []
    for name in features:
        single = _logit(y, sm.add_constant(X[[name]], has_constant="add"))
        single_ci = single.conf_int().loc[name]
        rows.append({
            "feature": name, "kind": KIND[name], "unit": UNIT_OVERRIDES.get(name, UNIT[KIND[name]]),
            "unadjusted_or": np.exp(single.params[name]),
            "unadjusted_ci_low": np.exp(single_ci[0]), "unadjusted_ci_high": np.exp(single_ci[1]),
            "adjusted_or": np.exp(full.params[name]),
            "adjusted_ci_low": np.exp(adjusted_ci.loc[name, 0]), "adjusted_ci_high": np.exp(adjusted_ci.loc[name, 1]),
            "adjusted_p_value": full.pvalues[name],
        })
    table = pd.DataFrame(rows)
    table["change_after_adjustment"] = [_change(row) for row in table.itertuples()]
    table["vif"] = variance_inflation(X)
    fit_summary = {"rows": int(full.nobs), "positives": int(np.sum(y)), "pseudo_r2_mcfadden": float(full.prsquared),
                   "converged": bool(full.mle_retvals["converged"]), "features": len(features)}
    return table, fit_summary


def _change(row):
    """Plain-language label for how an association changed once the other answers were held fixed."""
    unadjusted_clear = row.unadjusted_ci_low > 1 or row.unadjusted_ci_high < 1
    adjusted_clear = row.adjusted_ci_low > 1 or row.adjusted_ci_high < 1
    if unadjusted_clear and adjusted_clear and (row.unadjusted_or - 1) * (row.adjusted_or - 1) < 0:
        return "reversed"
    if unadjusted_clear and not adjusted_clear:
        return "no longer clear"
    if adjusted_clear and not unadjusted_clear:
        return "appears only after adjustment"
    if not adjusted_clear:
        return "unclear either way"
    shift = abs(np.log(row.adjusted_or)) / max(abs(np.log(row.unadjusted_or)), 1e-12)
    return "weaker" if shift < 0.8 else "stronger" if shift > 1.25 else "similar"


def variance_inflation(X):
    """Variance inflation factor per column; above about 5 means the answer overlaps heavily with others."""
    design = sm.add_constant(X, has_constant="add").to_numpy()
    return [variance_inflation_factor(design, i + 1) for i in range(X.shape[1])]


def plot_odds_ratios(table, destination, endpoint_label):
    """Forest plot: unadjusted (hollow) and adjusted (filled) odds ratios on a log scale."""
    table = table.sort_values("adjusted_or").reset_index(drop=True)
    y = np.arange(len(table))
    figure, axis = plt.subplots(figsize=(8.5, 7.2))
    axis.axvline(1, color="#6b6a63", linewidth=1, linestyle="--")
    for offset, prefix, style in ((0.18, "unadjusted", {"mfc": "white"}), (-0.18, "adjusted", {})):
        axis.errorbar(table[f"{prefix}_or"], y + offset,
                      xerr=[table[f"{prefix}_or"] - table[f"{prefix}_ci_low"],
                            table[f"{prefix}_ci_high"] - table[f"{prefix}_or"]],
                      fmt="o", ms=5, color="#2a78d6" if prefix == "adjusted" else "#6b6a63", elinewidth=1,
                      label=f"{prefix.capitalize()} (one answer at a time)" if prefix == "unadjusted"
                      else "Adjusted (all answers together)", **style)
    axis.set_xscale("log")
    low = min(table.unadjusted_ci_low.min(), table.adjusted_ci_low.min())
    high = max(table.unadjusted_ci_high.max(), table.adjusted_ci_high.max())
    ticks = [v for v in (0.25, 0.33, 0.5, 0.67, 1, 1.5, 2, 3, 4, 6, 8) if low / 1.2 <= v <= high * 1.2]
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_yticks(y, [f"{f}  ({u})" for f, u in zip(table.feature, table.unit)], fontsize=8.5)
    axis.set_xlabel("Odds ratio, 95% CI (log scale; right of 1 = higher odds of the outcome)", fontsize=9)
    axis.set_title(f"Associations before and after adjustment ({endpoint_label}), training rows only",
                   fontsize=10.5, loc="left")
    axis.grid(axis="x", color="#e4e3dd", linewidth=0.8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(frameon=False, fontsize=8.5, loc="lower right")
    figure.tight_layout()
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)

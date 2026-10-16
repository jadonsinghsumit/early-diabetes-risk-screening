"""Unadjusted association tests; no fitted predictive models or causal claims."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import Table2x2
from statsmodels.stats.multitest import multipletests

from .schema import FIELDS, TARGET


def odds_ratio(table):
    """Exposure=1 vs 0; columns are target=1 then target=0."""
    cells = table.reindex(index=[1, 0], columns=[1, 0], fill_value=0).to_numpy(dtype=float, copy=True)
    corrected = bool((cells == 0).any())
    if corrected:
        cells += 0.5
    estimate = Table2x2(cells, shift_zeros=False)
    lower, upper = estimate.oddsratio_confint(alpha=0.05, method="normal")
    return float(estimate.oddsratio), float(lower), float(upper), corrected


def rank_effect(positive, negative):
    """Rank-biserial effect and large-sample placement-based 95% interval.

    Positive effects mean higher values among target-positive respondents.
    Ties contribute half a win. Separate empirical placement variances
    estimate uncertainty under the observed alternative in O(n log n).
    """
    positive, negative = np.asarray(positive), np.asarray(negative)
    ordered_positive, ordered_negative = np.sort(positive), np.sort(negative)
    wins_positive = (np.searchsorted(ordered_negative, positive, side="left")
                     + np.searchsorted(ordered_negative, positive, side="right")) / (2 * len(negative))
    wins_negative = 1 - (np.searchsorted(ordered_positive, negative, side="left")
                         + np.searchsorted(ordered_positive, negative, side="right")) / (2 * len(positive))
    effect = 2 * wins_positive.mean() - 1
    variance = (wins_positive.var(ddof=1) / len(positive)
                + wins_negative.var(ddof=1) / len(negative))
    margin = stats.norm.ppf(0.975) * 2 * np.sqrt(variance)
    return float(effect), float(max(-1, effect - margin)), float(min(1, effect + margin))


def _base(feature, test, null, alternative):
    return {"feature": feature, "test": test, "H0": null, "H1": alternative,
            "status": "tested", "p_value": np.nan, "statistic": np.nan,
            "effect_name": "", "effect_size": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
            "ci_method": "", "cramers_v": np.nan, "minimum_expected_count": np.nan,
            "zero_cell_correction": False, "notes": ""}


def categorical_test(frame, feature):
    row = _base(feature, "chi_square", "Predictor and target are independent",
                "Predictor and target are associated")
    table = pd.crosstab(frame[feature], frame[TARGET]).reindex(columns=[0, 1], fill_value=0)
    if table.shape[0] < 2 or (table.sum(axis=0) == 0).any():
        row.update(status="skipped", notes="Requires variation in predictor and both target groups")
        return row
    statistic, p_value, _, expected = stats.chi2_contingency(table, correction=False)
    cramers_v = np.sqrt(statistic / (table.to_numpy().sum() * min(table.shape[0] - 1, 1)))
    row.update(statistic=float(statistic), p_value=float(p_value), cramers_v=float(cramers_v),
               effect_name="Cramer's V", effect_size=float(cramers_v),
               minimum_expected_count=float(expected.min()),
               ci_method="Not estimated for omnibus Cramer's V")
    if expected.min() < 5:
        row["notes"] = "Sparse expected cells: chi-square approximation may be unreliable; exploratory only"
    if set(table.index) == {0, 1}:
        effect, lower, upper, corrected = odds_ratio(table)
        row.update(effect_name="odds ratio (code 1 vs 0)", effect_size=effect,
                   ci_lower=lower, ci_upper=upper, ci_method="95% log-Wald, unadjusted",
                   zero_cell_correction=corrected)
        if corrected:
            row["notes"] += "; added 0.5 to all four cells for the odds ratio only"
    return row


def distribution_test(frame, feature):
    row = _base(feature, "mann_whitney_u", "Target groups have equal predictor distributions",
                "Predictor distributions differ between target groups")
    positive = frame.loc[frame[TARGET] == 1, feature].to_numpy()
    negative = frame.loc[frame[TARGET] == 0, feature].to_numpy()
    if min(len(positive), len(negative)) < 2:
        row.update(status="skipped", notes="Requires at least two observations in each target group")
        return row
    result = stats.mannwhitneyu(positive, negative, alternative="two-sided", method="asymptotic")
    effect, lower, upper = rank_effect(positive, negative)
    row.update(statistic=float(result.statistic), p_value=float(result.pvalue),
               effect_name="rank-biserial correlation (positive higher)", effect_size=effect,
               ci_lower=lower, ci_upper=upper, ci_method="95% normal placement interval, clipped to [-1, 1]",
               notes="Tie-corrected asymptotic U test; distribution test, not solely a test of medians")
    return row


def hypothesis_table(frame):
    rows = []
    for field in FIELDS:
        if field.name == TARGET:
            continue
        if field.kind in {"binary", "ordinal"}:
            rows.append(categorical_test(frame, field.name))
        if field.kind in {"numeric", "ordinal"}:
            rows.append(distribution_test(frame, field.name))
    table = pd.DataFrame(rows)
    tested = table["status"].eq("tested") & table["p_value"].notna()
    table["p_adjusted_bh"] = np.nan
    if tested.any():
        table.loc[tested, "p_adjusted_bh"] = multipletests(table.loc[tested, "p_value"], method="fdr_bh")[1]
    table["interpretation"] = "Not tested; inspect notes"
    for index in table.index[tested]:
        table.loc[index, "interpretation"] = (
            "Evidence of unadjusted association after BH correction; not causation"
            if table.loc[index, "p_adjusted_bh"] < 0.05 else
            "Insufficient evidence after BH correction; does not establish no association")
    return table

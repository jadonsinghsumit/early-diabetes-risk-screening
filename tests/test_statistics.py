import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu
from diabetes_screening.schema import TARGET
from diabetes_screening.statistics import (categorical_test, distribution_test, hypothesis_table,
                                            logistic_odds_ratio, odds_ratio, rank_effect)


def test_odds_ratio_direction_and_known_interval():
    # Exposure doubles odds: (40/20) / (20/20) = 2.
    table = pd.DataFrame([[20, 20], [20, 40]], index=[0, 1], columns=[0, 1])
    estimate, lower, upper, corrected = odds_ratio(table)
    margin = 1.95996398454 * np.sqrt(1/40 + 1/20 + 1/20 + 1/20)
    assert estimate == 2
    assert lower == pytest.approx(np.exp(np.log(2) - margin))
    assert upper == pytest.approx(np.exp(np.log(2) + margin))
    assert not corrected


def test_zero_cells_are_corrected_explicitly():
    table = pd.DataFrame([[10, 0], [4, 8]], index=[0, 1], columns=[0, 1])
    estimate, lower, upper, corrected = odds_ratio(table)
    assert corrected and 0 < lower < estimate < upper


def test_rank_effect_matches_u_and_handles_ties():
    positive, negative = [2, 2, 3, 4], [1, 2, 2, 3]
    estimate, lower, upper = rank_effect(positive, negative)
    assert estimate == pytest.approx(2 * mannwhitneyu(positive, negative).statistic / 16 - 1)
    assert -1 <= lower <= estimate <= upper <= 1
    reverse, _, _ = rank_effect(negative, positive)
    assert reverse == -estimate
    assert rank_effect([1, 1], [1, 1]) == (0, 0, 0)


def test_hypotheses_adjust_all_planned_tests(sample):
    table = hypothesis_table(sample)
    assert len(table) == 25
    assert table["status"].eq("tested").all()
    assert table["p_adjusted_bh"].between(0, 1).all()
    assert (table["p_adjusted_bh"] >= table["p_value"] - 1e-15).all()
    assert table["H0"].str.len().gt(0).all()
    assert table["H1"].str.len().gt(0).all()


def test_degenerate_and_sparse_groups(sample):
    sample["HighBP"] = 1
    assert categorical_test(sample, "HighBP")["status"] == "skipped"
    sparse = pd.DataFrame({TARGET: [0, 0, 1, 1], "HighBP": [0, 0, 0, 1]})
    row = categorical_test(sparse, "HighBP")
    assert "Sparse" in row["notes"] and row["zero_cell_correction"]
    sample[TARGET] = 0
    assert categorical_test(sample, "Sex")["status"] == "skipped"
    assert distribution_test(sample, "BMI")["status"] == "skipped"


def test_all_tests_skipped_is_an_explicit_outcome(sample):
    sample[TARGET] = 0
    table = hypothesis_table(sample)
    assert table["status"].eq("skipped").all()
    assert table["p_adjusted_bh"].isna().all()


def test_per_unit_odds_ratio_matches_two_by_two_table_for_binary_predictor(sample):
    table = pd.crosstab(sample["HighBP"], sample[TARGET]).reindex(columns=[0, 1], fill_value=0)
    expected, lower, upper, _ = odds_ratio(table)
    estimate, low, high, note = logistic_odds_ratio(sample, "HighBP")
    assert note == ""
    assert (estimate, low, high) == pytest.approx((expected, lower, upper), rel=1e-4)


def test_per_unit_odds_ratio_direction_and_failure_modes(sample):
    shifted = sample.copy()
    shifted["BMI"] = 20 + 3 * shifted[TARGET] + np.random.default_rng(5).integers(0, 8, 160)   # overlapping, higher when positive
    estimate, lower, upper, _ = logistic_odds_ratio(shifted, "BMI")
    assert 1 < lower < estimate < upper
    shifted["BMI"] = 30
    estimate, lower, upper, note = logistic_odds_ratio(shifted, "BMI")
    assert np.isnan([estimate, lower, upper]).all() and "not reported" in note
    separated = pd.DataFrame({TARGET: [0, 0, 0, 1, 1, 1], "BMI": [1, 2, 3, 10, 11, 12]})
    estimate, _, _, note = logistic_odds_ratio(separated, "BMI")
    assert np.isnan(estimate) and "did not converge" in note


def test_numeric_and_ordinal_rows_carry_per_unit_odds_ratios(sample):
    table = hypothesis_table(sample)
    numeric_rows = table["test"].eq("mann_whitney_u")
    assert table.loc[numeric_rows, "odds_ratio_per_unit"].notna().all()
    assert table.loc[~numeric_rows, "odds_ratio_per_unit"].isna().all()

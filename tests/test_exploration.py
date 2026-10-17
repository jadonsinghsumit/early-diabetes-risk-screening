import pandas as pd
from diabetes_screening.exploration import explore


def test_exploration_exports_actual_balance_and_comparisons(sample, tmp_path):
    balance = explore(sample, tmp_path)
    assert balance.loc[1, "fraction"] == 0.25
    assert balance["count"].sum() == len(sample)
    assert len(list(tmp_path.glob("*.png"))) == 9
    for figure in tmp_path.glob("*.png"):
        assert figure.stat().st_size > 1000
    correlation = pd.read_csv(tmp_path / "correlations_spearman.csv", index_col=0)
    assert correlation.loc["BMI", "BMI"] == 1
    assert "25.00%" in (tmp_path / "observations.md").read_text()
    factors = pd.read_csv(tmp_path / "categorical_factors.csv")
    assert factors.groupby("feature")["count"].sum().eq(160).all()


def test_observations_name_the_strongest_signals_and_plots_use_class_names(sample, tmp_path):
    explore(sample, tmp_path, endpoint="prediabetes")
    text = (tmp_path / "observations.md").read_text()
    assert "Strongest Spearman correlations" in text
    assert "prevalence ratios" in text and "not odds ratios" in text
    assert "Binary endpoint: prediabetes" in text

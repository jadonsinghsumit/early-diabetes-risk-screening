"""Stage 6 tests: baselines, the screening metrics, model selection, inference and the `train` command."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from diabetes_screening.modeling.cli import main
from diabetes_screening.modeling.baselines import AdaRiskScore, ada_points
from diabetes_screening.modeling.cohort import FEATURE_SETS, FEATURES
from diabetes_screening.modeling.inference import odds_ratio_table
from diabetes_screening.modeling.metrics import at_recall, max_f2
from diabetes_screening.modeling.preprocess import ScreeningPreprocessor
from diabetes_screening.modeling.training import choose_model, corrected_t_test, paired_differences
from diabetes_screening.preparation import prepare_binary
from diabetes_screening.schema import TARGET


def answers(**overrides):
    """One respondent with low-risk answers, changed where asked."""
    row = {name: 0 for name in FEATURES}
    row.update({"BMI": 22, "GenHlth": 1, "Age": 1, "Education": 6, "Income": 8, "PhysActivity": 1})
    row.update(overrides)
    return row


def test_ada_checklist_points_follow_the_paper_test():
    X = pd.DataFrame([
        answers(),                                                    # young, active, normal weight: 0
        answers(Age=5, BMI=27),                                       # 40-44 (1) + overweight (1): 2
        answers(Age=9, Sex=1, HighBP=1, PhysActivity=0, BMI=41),      # 3 + 1 + 1 + 1 + 3: 9
        answers(Age=7, BMI=30, HighBP=1),                             # 2 + 2 + 1: 5, the high-risk line
    ])
    assert list(ada_points(X)) == [0, 2, 9, 5]
    model = AdaRiskScore().fit(X)
    assert list(model.predict(X)) == [0, 0, 1, 1]
    assert model.predict_proba(X)[2, 1] == pytest.approx(1.0)


def test_screening_operating_point_and_max_f2():
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.65, 0.3, 0.2, 0.15, 0.05])
    point = at_recall(y, proba)  # needs 4 of 5 cases: cutoff 0.6 flags 5 people, 1 false alarm
    assert point["threshold"] == 0.6 and point["recall"] == 0.8
    assert point["precision"] == 0.8 and point["specificity"] == 0.8 and point["flagged_share"] == 0.5
    assert 0 < max_f2(y, proba) <= 1


def test_preprocessor_settings_can_be_tuned_and_cloned():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({name: rng.integers(1, 5, 50) for name in FEATURES})
    X["Age"], X["GenHlth"] = rng.integers(1, 14, 50), rng.integers(1, 6, 50)
    onehot = ScreeningPreprocessor("linear-onehot").fit(X)
    assert len(onehot.get_feature_names_out()) == 45  # 12 age + 4 health + 5 education + 7 income + 17 others
    reduced = clone(onehot).set_params(kind="linear", features=tuple(FEATURE_SETS["no-healthcare-contact"]))
    assert reduced.fit_transform(X).shape == (50, 16)


def fold_table(scores, brier=None):
    """Per-fold PR-AUC (and Brier) for made-up models, in the layout fold_results produces."""
    rows = []
    for model, values in scores.items():
        for fold, value in enumerate(values):
            rows.append({"model": model, "fold": fold, "pr_auc": value,
                         "brier": (brier or {}).get(model, 0.1)})
    return pd.DataFrame(rows)


def test_selection_rule_prefers_simple_or_calibrated_models_when_gaps_are_noise():
    clear = fold_table({"logistic_regression": [0.40, 0.41, 0.40, 0.42, 0.41],
                        "xgboost": [0.45, 0.46, 0.45, 0.47, 0.46]})
    assert choose_model(clear, ["logistic_regression", "xgboost"])["chosen"] == "xgboost"
    noisy = fold_table({"logistic_regression": [0.40, 0.46, 0.40, 0.46, 0.40],
                        "xgboost": [0.46, 0.40, 0.47, 0.40, 0.46]})
    assert choose_model(noisy, ["logistic_regression", "xgboost"])["chosen"] == "logistic_regression"
    tied = fold_table({"logistic_regression": [0.40, 0.41, 0.40, 0.42, 0.41],
                       "random_forest": [0.461, 0.458, 0.462, 0.459, 0.460],
                       "xgboost": [0.459, 0.460, 0.461, 0.460, 0.461]},
                      brier={"random_forest": 0.107, "xgboost": 0.105})
    decision = choose_model(tied, ["logistic_regression", "random_forest", "xgboost"])
    assert decision["chosen"] == "xgboost" and set(decision["tied_within_noise"]) == {"random_forest", "xgboost"}


def test_corrected_t_test_matches_a_hand_calculation():
    gaps = np.array([0.02, 0.03, 0.025, 0.035, 0.02])
    # mean 0.026, SD 0.00652; corrected SE = sqrt((1/5 + 1/4) * 0.0000425) = 0.004373; t = 5.945
    statistic, p_value = corrected_t_test(gaps)
    assert statistic == pytest.approx(5.945, abs=0.001)
    assert p_value == pytest.approx(0.00401, abs=0.0001)  # t distribution with 4 degrees of freedom
    assert corrected_t_test([0.01] * 5) == (np.inf, 0.0) and corrected_t_test([0.0] * 5) == (0.0, 1.0)
    table = paired_differences(fold_table({"logistic_regression": [0.40] * 5,
                                           "xgboost": list(0.40 + gaps)}), ["logistic_regression", "xgboost"])
    assert table.loc[0, "folds_won"] == 5 and table.loc[0, "p_value"] == pytest.approx(p_value)


def test_adjusted_odds_ratios_recover_a_planted_effect():
    rng = np.random.default_rng(7)
    n = 4000
    X = pd.DataFrame({name: rng.integers(0, 2, n) for name in FEATURES})
    X["BMI"], X["Age"] = rng.integers(18, 40, n), rng.integers(1, 14, n)
    log_odds = -2 + 1.2 * X["HighBP"] + 0.1 * (X["Age"] - 7)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-log_odds))).astype(int))
    table, fit = odds_ratio_table(X, y)
    high_bp = table.set_index("feature").loc["HighBP"]
    assert high_bp.adjusted_ci_low < np.exp(1.2) < high_bp.adjusted_ci_high
    assert fit["converged"] and table.vif.max() < 2
    assert set(table.change_after_adjustment) <= {"reversed", "no longer clear", "appears only after adjustment",
                                                  "unclear either way", "weaker", "stronger", "similar"}


@pytest.fixture
def prepared(sample, tmp_path):
    three_class = sample.rename(columns={TARGET: "Diabetes_012"})
    three_class["Diabetes_012"] = [0, 1, 2, 0] * 40
    source = tmp_path / "three.csv"
    three_class.to_csv(source, index=False)
    output = tmp_path / "combined.csv"
    prepare_binary(source, output, "combined", expected_rows=160)
    return output, output.with_suffix(".provenance.json")


def test_cli_train_runs_end_to_end_without_touching_the_test_set(prepared, tmp_path):
    path, provenance = prepared
    common = ["--input", str(path), "--provenance", str(provenance), "--synthetic-rows", "160"]
    assert main(["split", *common, "--output", str(tmp_path / "split")]) == 0
    out = tmp_path / "train"
    assert main(["train", *common, "--split", str(tmp_path / "split"), "--output", str(out),
                 "--models", "logistic_regression,xgboost", "--n-iter", "1"]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["test_rows_used"] == 0 and manifest["training_rows_used"] == 128
    comparison = pd.read_csv(out / "comparison.csv")
    assert {"no_model", "ada_checklist", "logistic_regression", "xgboost"} == set(comparison.model)
    assert json.loads((out / "selection.json").read_text())["chosen"] in {"logistic_regression", "xgboost"}
    best = json.loads((out / "best_params.json").read_text())
    assert best["xgboost"]["search_pr_auc_mean"] == pytest.approx(best["xgboost"]["refit_check_pr_auc_mean"])
    oof = pd.read_csv(out / "oof_predictions.csv.gz")
    assert len(oof) == 128 and oof[["logistic_regression", "xgboost"]].notna().all().all()
    assert set(pd.read_csv(out / "permutation_importance.csv").feature) == set(FEATURES)
    pipeline = joblib.load(out / "models" / "logistic_regression.joblib")
    assert pipeline.predict_proba(pd.read_csv(path)[FEATURES]).shape == (160, 2)
    for name in ["README.md", "comparison.png", "odds_ratios.png", "permutation_importance.png", "tuning.csv",
                 "feature_sets.csv", "checklist_at_5_points.csv", "odds_ratios.csv"]:
        assert (out / name).exists(), name
    assert main(["train", *common, "--split", str(tmp_path / "split"), "--output", str(tmp_path / "bad"),
                 "--models", "svm"]) == 1

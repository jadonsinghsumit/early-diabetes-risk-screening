"""Stage 7 tests: calibration, operating points, bootstrap intervals, subgroups, and the threshold-then-test lock."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import Booster
from xgboost.core import XGBoostError

from diabetes_screening.modeling.cli import main
from diabetes_screening.modeling.cohort import FEATURES
from diabetes_screening.modeling.evaluation import (bootstrap_intervals, calibration_summary, f2_cutoff,
                                                    flagged_by_original_class, missed_case_profile,
                                                    needs_recalibration, operating_points, recalibrate,
                                                    subgroup_table)
from diabetes_screening.modeling.imbalance import model_factory
from diabetes_screening.modeling.portable import load_pipeline, snapshot_model
from diabetes_screening.preparation import prepare_binary
from diabetes_screening.schema import FIELDS, TARGET


def simulated(n=20000, seed=3):
    rng = np.random.default_rng(seed)
    risk = 1 / (1 + np.exp(-rng.normal(-1.8, 1.0, n)))
    return (rng.random(n) < risk).astype(int), risk


def test_calibration_flags_overconfident_risks_and_platt_scaling_fixes_them():
    y, risk = simulated()
    honest = calibration_summary(y, risk)
    assert honest["calibration_slope"] == pytest.approx(1, abs=0.06)
    assert honest["calibration_intercept"] == pytest.approx(0, abs=0.05)
    assert not needs_recalibration(honest)
    log_odds = np.log(risk / (1 - risk))
    extreme = 1 / (1 + np.exp(-2 * log_odds))  # same ranking, risks pushed towards 0 and 1
    summary = calibration_summary(y, extreme)
    assert summary["calibration_slope"] == pytest.approx(0.5, abs=0.04) and needs_recalibration(summary)
    fixed = calibration_summary(y, recalibrate(extreme, summary["platt"]))
    assert fixed["calibration_slope"] == pytest.approx(1, abs=0.01) and not needs_recalibration(fixed)
    assert np.array_equal(np.argsort(recalibrate(extreme, summary["platt"])), np.argsort(extreme))
    assert np.array_equal(recalibrate(extreme, None), extreme)


def test_operating_points_report_the_workload_of_each_cutoff():
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.65, 0.3, 0.2, 0.15, 0.05])
    points = operating_points(y, proba, locked_recall=0.80).set_index("rule")
    locked = points.loc["catch 80% of cases"]
    assert locked.locked and points.locked.sum() == 1
    assert locked.threshold == 0.6 and (locked.tp, locked.fp, locked.fn) == (4, 1, 1)
    assert (locked.flagged_per_1000, locked.cases_found_per_1000, locked.cases_missed_per_1000,
            locked.false_alarms_per_1000) == (500, 400, 100, 100)
    # Flagging everyone down to 0.1 catches all 5 cases with 4 false alarms: F2 = 25/29, the best available.
    assert f2_cutoff(y, proba) == 0.1 and points.loc["best F2", "f2"] == pytest.approx(25 / 29)


def test_bootstrap_intervals_hold_the_cutoff_fixed_and_are_reproducible():
    y = np.tile([0, 0, 0, 1], 100)
    perfect = bootstrap_intervals(y, y * 0.9 + 0.05, threshold=0.5, n_resamples=50)
    assert perfect["recall"] == (1.0, 1.0) and perfect["roc_auc"] == (1.0, 1.0)
    y, risk = simulated(n=2000)
    first = bootstrap_intervals(y, risk, threshold=0.2, n_resamples=200, seed=1)
    assert first == bootstrap_intervals(y, risk, threshold=0.2, n_resamples=200, seed=1)
    recall = ((risk >= 0.2) & (y == 1)).sum() / y.sum()
    assert first["recall"][0] < recall < first["recall"][1]
    assert 0 < first["recall"][1] - first["recall"][0] < 0.15


def test_subgroups_missed_cases_and_original_classes_count_correctly():
    X = pd.DataFrame({name: 0 for name in FEATURES}, index=range(8))
    X["Age"], X["Sex"], X["Income"] = [2, 2, 2, 2, 11, 11, 11, 11], [0, 1] * 4, [8] * 8
    X["BMI"], X["GenHlth"], X["PhysActivity"], X["CholCheck"], X["AnyHealthcare"] = 30, 3, 1, 1, 1
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    proba = np.array([0.9, 0.1, 0.8, 0.1, 0.9, 0.9, 0.1, 0.1])
    table = subgroup_table(X, y, proba, threshold=0.5).set_index(["group", "level"])
    young, old = table.loc[("Age", "18-44")], table.loc[("Age", "65+")]
    assert (young.rows, young.positives, young.recall, young.false_positive_rate) == (4, 2, 0.5, 0.5)
    assert (old.recall, old.false_positive_rate) == (1.0, 0.0)
    assert table.loc[("Age", "45-64"), "rows"] == 0 and np.isnan(table.loc[("Age", "45-64"), "recall"])
    assert old.recall_ci_low < 1 and young.recall_ci_low < 0.5 < young.recall_ci_high
    profile = missed_case_profile(X, y, proba, threshold=0.5).set_index("answer")
    assert profile.loc["People"].to_dict() == {"caught_cases": 3, "missed_cases": 1, "non_cases": 4}
    assert profile.loc["Age under 45 (share)", "missed_cases"] == 1.0
    by_class = flagged_by_original_class([2, 1, 0, 0, 2, 2, 0, 0], proba, threshold=0.5).set_index("label")
    assert by_class.loc["Prediabetes", "flagged_share"] == 0.0 and by_class.loc["Diabetes", "flagged"] == 3
    assert by_class.loc["No diabetes", "flagged_share"] == 0.25


def test_xgboost_pipelines_pickled_on_another_platform_still_load(tmp_path, monkeypatch):
    """A snapshot this build can't read whole (like a Linux pickle on Windows) still gives identical predictions."""
    rng = np.random.default_rng(5)
    X = pd.DataFrame({f.name: rng.integers(f.minimum, f.maximum + 1, 400) for f in FIELDS if f.name != TARGET})
    y = (X["BMI"] + 5 * X["HighBP"] + rng.normal(0, 5, 400) > 40).astype(int)
    pipeline = model_factory("xgboost", False)(y).set_params(model__n_estimators=20).fit(X, y)
    raw = bytearray(pipeline.named_steps["model"].get_booster().__getstate__()["handle"])
    marker = raw.index(b"rng_state") + len(b"rng_state")  # 'S', then the string's length, then its text
    size = {"i": 1, "U": 1, "I": 2, "l": 4, "L": 8}[chr(raw[marker + 1])]
    start = marker + 2 + size
    length = int.from_bytes(raw[marker + 2:start], "big")
    raw[start:start + length] = b"x" * length  # a generator state in a layout this build can't parse
    path = tmp_path / "foreign.joblib"
    with monkeypatch.context() as patch:
        patch.setattr(Booster, "__getstate__", lambda self: {"handle": raw})
        joblib.dump(pipeline, path, compress=3)
    original = Booster.__setstate__
    with pytest.raises(XGBoostError, match="corrupted"):
        joblib.load(path)
    assert np.array_equal(load_pipeline(path).predict_proba(X), pipeline.predict_proba(X))
    assert Booster.__setstate__ is original
    with pytest.raises(ValueError, match="UBJSON"):
        snapshot_model(b"not a snapshot")


@pytest.fixture
def prepared(sample, tmp_path):
    three_class = sample.rename(columns={TARGET: "Diabetes_012"})
    three_class["Diabetes_012"] = [0, 1, 2, 0] * 40
    source = tmp_path / "three.csv"
    three_class.to_csv(source, index=False)
    output = tmp_path / "combined.csv"
    prepare_binary(source, output, "combined", expected_rows=160)
    return output, output.with_suffix(".provenance.json"), source


def test_cli_locks_the_threshold_before_the_test_set_is_scored_once(prepared, tmp_path):
    path, provenance, source = prepared
    common = ["--input", str(path), "--provenance", str(provenance), "--synthetic-rows", "160"]
    split, training, locked = tmp_path / "split", tmp_path / "train", tmp_path / "threshold"
    assert main(["split", *common, "--output", str(split)]) == 0
    assert main(["train", *common, "--split", str(split), "--output", str(training),
                 "--models", "logistic_regression,xgboost", "--n-iter", "1", "--skip-importance"]) == 0
    staged = ["--split", str(split), "--training", str(training)]
    test_args = ["test", *common, *staged, "--threshold", str(locked), "--bootstrap", "20"]

    assert main([*test_args, "--output", str(tmp_path / "too-early")]) == 1  # no threshold.json yet
    assert not (tmp_path / "too-early").exists()

    assert main(["threshold", *common, *staged, "--output", str(locked)]) == 0
    record = json.loads((locked / "threshold.json").read_text())
    assert record["test_rows_used"] == 0 and record["training_rows_used"] == 128
    assert record["validation_at_threshold"]["recall"] >= 0.8 and 0 < record["threshold"] <= 1
    assert set(record["reference_operating_points"]) <= {"ada_checklist", "logistic_regression"}
    assert len(pd.read_csv(locked / "operating_points.csv")) == 6
    assert json.loads((locked / "manifest.json").read_text())["test_rows_used"] == 0

    out = tmp_path / "test"
    assert main([*test_args, "--source", str(source), "--output", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["test_rows_used"] == 32 and manifest["training_rows_used"] == 0
    assert manifest["original_classes_checked"]
    results = pd.read_csv(out / "test_results.csv")
    assert results.loc[results.role == "selected", "model"].item() == record["model"]
    assert (results.recall_ci_low <= results.recall).all() and (results.recall <= results.recall_ci_high).all()
    assert set(pd.read_csv(out / "flagged_by_original_class.csv").label) == {"No diabetes", "Prediabetes",
                                                                             "Diabetes"}
    for name in ["README.md", "curves.png", "calibration_test.png", "confusion_matrix.png", "subgroups.png",
                 "subgroups.csv", "missed_cases.csv", "calibration_test.csv", "reliability_test.csv"]:
        assert (out / name).exists(), name

    model = training / "models" / record["model_file"]
    model.write_bytes(model.read_bytes() + b"\0")  # a model changed after the cutoff was locked
    assert main([*test_args, "--output", str(tmp_path / "changed")]) == 1
    assert main(["threshold", *common, *staged, "--output", str(tmp_path / "bad"), "--target-recall", "1.5"]) == 1

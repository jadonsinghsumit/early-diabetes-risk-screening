"""Stage 5 tests: split integrity, leakage safety, the test-set lock and the weighting comparison."""
import gzip
import json

import numpy as np
import pandas as pd
import pytest

from diabetes_screening.modeling.cli import main
from diabetes_screening.modeling import (FEATURE_SETS, Cohort, LockedTestSetError, build_preprocessor,
                                         compare_weighting, evaluate, load_cohort, load_split, locked_test_data,
                                         make_split, model_factory, out_of_fold, profile_summary, save_split,
                                         training_data)
from diabetes_screening.modeling.cohort import FEATURES
from diabetes_screening.modeling.imbalance import grouped_cv_check
from diabetes_screening.preparation import prepare_binary
from diabetes_screening.schema import FIELDS, TARGET


def synthetic_cohort(profiles=400, repeats=1, seed=0):
    """Random valid answers; every 5th row positive (20%); `repeats` identical copies per profile."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f.name: rng.integers(f.minimum, f.maximum + 1, profiles)
                      for f in FIELDS if f.name != TARGET})[FEATURES]
    X = X.loc[X.index.repeat(repeats)].reset_index(drop=True)
    y = pd.Series((np.arange(len(X)) % 5 == 0).astype(int), name=TARGET)
    profile_id = X.groupby(FEATURES).ngroup()
    return Cohort(X=X, y=y, profile_id=profile_id, endpoint="combined", input_sha256="synthetic", provenance={})


@pytest.fixture
def prepared(sample, tmp_path):
    three_class = sample.rename(columns={TARGET: "Diabetes_012"})
    three_class["Diabetes_012"] = [0, 1, 2, 0] * 40
    source = tmp_path / "three.csv"
    three_class.to_csv(source, index=False)
    output = tmp_path / "combined.csv"
    prepare_binary(source, output, "combined", expected_rows=160)
    return output, output.with_suffix(".provenance.json")


def test_load_cohort_verifies_provenance(prepared):
    path, provenance = prepared
    cohort = load_cohort(path, provenance, expected_source_rows=160)
    assert cohort.X.shape == (160, 21) and list(cohort.X.columns) == FEATURES
    assert cohort.endpoint == "combined" and int(cohort.y.sum()) == 80
    assert profile_summary(cohort)["rows"] == 160
    path.write_text(path.read_text().replace("\n1,", "\n0,", 1))  # flip one label after preparation
    with pytest.raises(ValueError, match="hash"):
        load_cohort(path, provenance, expected_source_rows=160)


def test_profile_summary_detects_repeated_answers():
    summary = profile_summary(synthetic_cohort(profiles=50, repeats=4))
    assert summary["distinct_profiles"] == 50 and summary["largest_profile_size"] == 4
    assert summary["rows_in_repeated_profiles"] == 200
    assert summary["repeated_profiles_with_both_outcomes"] > 0


def test_split_is_stratified_disjoint_and_reproducible():
    cohort = synthetic_cohort()
    plan = make_split(cohort, seed=1)
    train, test = plan.train_index, plan.test_index
    assert len(test) == 80 and int(cohort.y.iloc[test].sum()) == 16
    assert len(np.intersect1d(train, test)) == 0 and len(train) + len(test) == 400
    assert set(plan.train_folds) == {0, 1, 2, 3, 4}
    fold_positives = [int(cohort.y.iloc[train][plan.train_folds == k].sum()) for k in range(5)]
    assert max(fold_positives) - min(fold_positives) <= 1
    assert make_split(cohort, seed=1).sha256 == plan.sha256
    assert make_split(cohort, seed=2).sha256 != plan.sha256
    # plan.cv() hands stage 6 exactly these folds, indexed within the training rows.
    for k, (_, held_out) in enumerate(plan.cv().split()):
        assert np.array_equal(held_out, np.flatnonzero(plan.train_folds == k))


def test_grouped_split_keeps_identical_profiles_together():
    cohort = synthetic_cohort(profiles=100, repeats=4)
    plan = make_split(cohort, seed=3, group_profiles=True)
    profiles = cohort.profile_id.to_numpy()
    assert not set(profiles[plan.train_index]) & set(profiles[plan.test_index])
    folds_per_profile = pd.Series(plan.train_folds).groupby(profiles[plan.train_index]).nunique()
    assert folds_per_profile.max() == 1 and plan.strategy == "stratified-group"
    with pytest.raises(ValueError, match="1/k"):
        make_split(cohort, test_size=0.3, group_profiles=True)


def test_saved_split_round_trips_and_rejects_edits(tmp_path):
    cohort = synthetic_cohort()
    plan = make_split(cohort)
    summary = save_split(plan, cohort, tmp_path / "split")
    assert summary["test"]["rows"] == 80 and summary["assignment_sha256"] == plan.sha256
    assert np.array_equal(load_split(tmp_path / "split", cohort).fold, plan.fold)
    with pytest.raises(FileExistsError):
        save_split(plan, cohort, tmp_path / "split")
    other = Cohort(cohort.X, cohort.y, cohort.profile_id, "combined", "another-file", {})
    with pytest.raises(ValueError, match="different input"):
        load_split(tmp_path / "split", other)
    assignment = tmp_path / "split" / "split_assignment.csv.gz"
    frame = pd.read_csv(assignment)
    frame.loc[frame.fold == -1, "fold"] = 0  # quietly move the test rows into training
    with gzip.open(assignment, "wt") as stream:
        frame.to_csv(stream, index=False)
    with pytest.raises(ValueError, match="edited"):
        load_split(tmp_path / "split", cohort)


def test_test_set_stays_locked_until_threshold_is_saved(tmp_path):
    cohort = synthetic_cohort()
    plan = make_split(cohort)
    X_train, y_train, folds = training_data(cohort, plan)
    assert len(X_train) == len(y_train) == len(folds) == 320
    assert not set(X_train.index) & set(plan.test_index)
    threshold = tmp_path / "threshold.json"
    with pytest.raises(LockedTestSetError, match="does not exist"):
        locked_test_data(cohort, plan, threshold)
    threshold.write_text(json.dumps({"threshold": 0.2, "split_sha256": "other"}))
    with pytest.raises(LockedTestSetError, match="different split"):
        locked_test_data(cohort, plan, threshold)
    threshold.write_text(json.dumps({"threshold": 0.2, "split_sha256": plan.sha256}))
    X_test, y_test = locked_test_data(cohort, plan, threshold)
    assert len(X_test) == 80 and int(y_test.sum()) == 16


def test_scalers_learn_only_from_the_training_part_of_each_fold():
    cohort = synthetic_cohort()
    plan = make_split(cohort)
    X, y, folds = training_data(cohort, plan)
    X = X.reset_index(drop=True)
    X.loc[folds == 0, "BMI"] = 98  # canary: extreme values only in validation fold 0
    _, models = out_of_fold(model_factory("logistic_regression", False), X, y, folds, keep_models=True)
    for fold, model in zip(range(5), models):
        scaler = model.named_steps["preprocess"].transformer_.named_transformers_["bmi"]
        assert scaler.center_[0] == pytest.approx(X.loc[folds != fold, "BMI"].median())
    # The model validated on fold 0 never saw the canary, so its scaler matches the clean rows.
    clean_median = X.loc[folds != 0, "BMI"].median()
    assert models[0].named_steps["preprocess"].transformer_.named_transformers_["bmi"].center_[0] == clean_median < 98


def test_preprocessor_kinds_and_feature_sets():
    X = synthetic_cohort().X
    reduced = build_preprocessor("linear", FEATURE_SETS["no-healthcare-contact"]).fit(X)
    names = list(reduced.get_feature_names_out())
    assert len(names) == 16 and "CholCheck" not in names and "HighBP" not in names
    assert len(build_preprocessor("tree").fit(X).get_feature_names_out()) == 21
    scaled = build_preprocessor("linear").fit_transform(X)
    assert scaled.shape == (400, 21)
    with pytest.raises(ValueError, match="kind"):
        build_preprocessor("neural")


def test_evaluate_matches_hand_counts():
    result = evaluate([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (1, 1, 1, 1)
    assert result["recall"] == result["precision"] == result["specificity"] == result["f2"] == 0.5
    assert result["roc_auc"] == 0.75 and result["prevalence"] == 0.5
    assert np.isnan(evaluate([0, 1], [0.1, 0.2])["precision"])  # nobody flagged


def test_weighting_comparison_and_xgboost_ratio():
    cohort = synthetic_cohort()
    plan = make_split(cohort)
    X, y, folds = training_data(cohort, plan)
    per_fold, agreement, timing = compare_weighting(X, y, folds, log=lambda message: None)
    assert len(per_fold) == 30 and len(agreement) == 15 and len(timing) == 6
    assert per_fold.groupby(["model", "weighting"])["rows"].sum().eq(320).all()  # training rows only
    lr = per_fold[per_fold.model == "logistic_regression"].groupby("weighting").mean_predicted_risk.mean()
    assert lr["weighted"] > lr["unweighted"]  # balanced weights push predicted risk toward 50%
    check = grouped_cv_check(X, y, folds, cohort.profile_id.iloc[plan.train_index], models=["logistic_regression"],
                             log=lambda message: None)
    assert list(check.cv_folds) == ["plain stratified", "profile-grouped"]
    xgb = model_factory("xgboost", True)(np.array([0, 0, 0, 1])).named_steps["model"]
    assert xgb.get_params()["scale_pos_weight"] == 3.0
    with pytest.raises(ValueError, match="Unknown model"):
        model_factory("svm", False)


def test_cli_split_then_imbalance(prepared, tmp_path, capsys):
    path, provenance = prepared
    common = ["--input", str(path), "--provenance", str(provenance), "--synthetic-rows", "160"]
    assert main(["split", *common, "--output", str(tmp_path / "split")]) == 0
    assert "test 32" in capsys.readouterr().out
    assert json.loads((tmp_path / "split/profiles.json").read_text())["rows"] == 160
    assert main(["imbalance", *common, "--split", str(tmp_path / "split"), "--output", str(tmp_path / "cmp"),
                 "--models", "logistic_regression,xgboost"]) == 0
    manifest = json.loads((tmp_path / "cmp/manifest.json").read_text())
    assert manifest["test_rows_used"] == 0 and manifest["training_rows_used"] == 128
    for name in ["weighting_summary.csv", "weighting_comparison.png", "grouped_cv_check.csv", "README.md"]:
        assert (tmp_path / "cmp" / name).exists()
    check = pd.read_csv(tmp_path / "cmp/grouped_cv_check.csv")
    assert set(check.cv_folds) == {"plain stratified", "profile-grouped"}
    assert main(["imbalance", *common, "--split", str(tmp_path / "split"), "--output", str(tmp_path / "lr"),
                 "--models", "logistic_regression", "--skip-grouped-check"]) == 0
    assert not (tmp_path / "lr/grouped_cv_check.csv").exists()
    assert main(["imbalance", *common, "--split", str(tmp_path / "split"), "--output", str(tmp_path / "bad"),
                 "--models", "svm"]) == 1

"""Command-level runners for roadmap stage 7: lock the cutoff on training rows, then score the test set once.

The two steps are separate commands on purpose. `threshold` reads training rows only and writes
threshold.json. That file is committed before anyone runs `test`, so the git history shows the cutoff
was fixed before a single test row was scored. `test` refuses to run unless threshold.json names this
exact input, split and model file.
"""
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import platform
import zipfile

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score

from ..endpoints import ENDPOINTS
from ..ingestion import write_json
from ..preparation import THREE_CLASS_FILE
from ..schema import EXPECTED_ROWS
from .baselines import ADA_CUTOFF, AdaRiskScore
from .cohort import load_cohort
from .evaluation import (BOOTSTRAP_METRICS, RECALIBRATION_RULE, THRESHOLD_RULE, at_cutoff, bootstrap_intervals,
                         calibration_summary, flagged_by_original_class, missed_case_profile, needs_recalibration,
                         operating_points, plot_calibration, plot_confusion, plot_curves, plot_subgroups,
                         recalibrate, reliability_table, subgroup_table)
from .imbalance import out_of_fold
from .metrics import SCREENING_RECALL, at_recall, evaluate
from .portable import load_pipeline
from .split import LockedTestSetError, load_split, locked_test_data, training_data
from .split_workflow import PACKAGES, _markdown, endpoint_label
from .training import LABELS, MODEL_COLORS

REFERENCE = "logistic_regression"  # scored next to the selected model for context; it decides nothing
CHECKLIST = "ada_checklist"
REPRODUCTION_TOLERANCE = 5e-4  # out-of-fold PR-AUC must match stage 6's refit check this closely


def _sha256(path, text=False):
    """File hash; `text=True` ignores line endings, which git may convert on checkout (core.autocrlf)."""
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if text else data).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _training_outputs(training_directory, cohort, plan):
    """The stage 6 outputs this stage builds on, after checking they belong to this cohort and split."""
    training_directory = Path(training_directory)
    manifest = _read_json(training_directory / "manifest.json")
    if manifest.get("input_sha256") != cohort.input_sha256 or manifest.get("split_sha256") != plan.sha256:
        raise ValueError("Stage 6 outputs belong to a different input file or split")
    selection = _read_json(training_directory / "selection.json")
    paths = {name: training_directory / "models" / f"{name}.joblib" for name in [selection["chosen"], REFERENCE]}
    if not paths[selection["chosen"]].exists():
        raise ValueError(f"The selected model's pipeline is missing: {paths[selection['chosen']]}")
    return selection["chosen"], _read_json(training_directory / "best_params.json"), {
        name: path for name, path in paths.items() if path.exists()}


def _fold_pr_auc(y, proba, folds):
    return float(np.mean([average_precision_score(y[folds == k], proba[folds == k]) for k in np.unique(folds)]))


def run_threshold(source, provenance, split_directory, training_directory, destination,
                  target_recall=SCREENING_RECALL, expected_source_rows=EXPECTED_ROWS, log=print):
    """Stage 7a: lock the cutoff on out-of-fold predictions. The test rows are never loaded."""
    if not 0 < target_recall < 1:
        raise ValueError("The target recall must be between 0 and 1")
    destination = Path(destination)
    cohort = load_cohort(source, provenance, expected_source_rows)
    plan = load_split(split_directory, cohort)
    chosen, best, paths = _training_outputs(training_directory, cohort, plan)
    X, y, folds = training_data(cohort, plan)
    y = y.to_numpy()
    destination.mkdir(parents=True, exist_ok=False)
    label = endpoint_label(cohort.endpoint)
    log(f"Training rows: {len(y):,} ({int(y.sum()):,} positive). Test rows locked: {len(plan.test_index):,}")

    # A fresh copy of each saved pipeline is refitted on four folds and scores the fifth, so every
    # training row gets a risk from a model that never saw it. Stage 6 kept these only locally.
    oof, models = {}, {}
    for name, path in paths.items():
        log(f"  out-of-fold predictions: {name}")
        pipeline = load_pipeline(path)
        oof[name], _ = out_of_fold(lambda y_fit, p=pipeline: clone(p), X, y, folds)
        now, then = _fold_pr_auc(y, oof[name], folds), best[name]["refit_check_pr_auc_mean"]
        models[name] = {"model_file": path.name, "model_sha256": _sha256(path), "oof_pr_auc_mean": now,
                        "stage6_pr_auc_mean": then,
                        "reproduces_stage6": bool(abs(now - then) <= REPRODUCTION_TOLERANCE)}
        if not models[name]["reproduces_stage6"]:
            log(f"  warning: {name} out-of-fold PR-AUC {now:.4f} differs from stage 6 ({then:.4f}); "
                "check the library versions against the stage 6 manifest")

    points = operating_points(y, oof[chosen], target_recall)
    locked = points[points.locked].iloc[0].to_dict()
    calibration = {name: calibration_summary(y, proba) for name, proba in oof.items()}
    platt = calibration[chosen]["platt"] if needs_recalibration(calibration[chosen]) else None
    reference = {CHECKLIST: {"threshold": ADA_CUTOFF, "rule": "5 or more checklist points (the paper test's cutoff)"}}
    if REFERENCE in oof and REFERENCE != chosen:
        reference[REFERENCE] = {"model_file": models[REFERENCE]["model_file"],
                                "model_sha256": models[REFERENCE]["model_sha256"],
                                "threshold": at_recall(y, oof[REFERENCE], target_recall)["threshold"],
                                "rule": THRESHOLD_RULE.format(recall=target_recall).replace("selected", "reference")}
    record = {
        "stage": 7, "endpoint": cohort.endpoint, "input_sha256": cohort.input_sha256, "split_sha256": plan.sha256,
        "model": chosen, "model_file": models[chosen]["model_file"], "model_sha256": models[chosen]["model_sha256"],
        "target_recall": target_recall, "rule": THRESHOLD_RULE.format(recall=target_recall),
        "threshold": float(locked["threshold"]),
        "validation_at_threshold": {key: value.item() if isinstance(value, np.generic) else value
                                    for key, value in locked.items() if key not in {"rule", "locked"}},
        "calibration_rule": RECALIBRATION_RULE, "validation_calibration": calibration[chosen], "recalibration": platt,
        "reference_operating_points": reference,
        "training_rows_used": int(len(y)), "test_rows_used": 0,
    }
    write_json(destination / "threshold.json", record)
    points.to_csv(destination / "operating_points.csv", index=False)
    reliability = pd.concat([reliability_table(y, proba).assign(model=name) for name, proba in oof.items()])
    reliability.to_csv(destination / "calibration_validation.csv", index=False)
    plot_calibration([(LABELS[name], MODEL_COLORS[name], reliability[reliability.model == name]) for name in oof],
                     destination / "calibration_validation.png", f"Out-of-fold calibration ({label})")
    manifest = {
        "stage": "7a", "endpoint": cohort.endpoint, "input_sha256": cohort.input_sha256, "split_sha256": plan.sha256,
        "seed": plan.seed, "training_rows_used": int(len(y)), "test_rows_used": 0, "models": models,
        "python": platform.python_version(), "packages": {n: version(n) for n in PACKAGES + ["statsmodels"]},
    }
    write_json(destination / "manifest.json", manifest)
    _threshold_readme(destination, label, plan, y, chosen, record, points, calibration, models)
    return record


def _percent(value):
    return f"{value:.1%}"


def _tidy(frame, digits=3):
    """Round float columns for display; adding 0.0 turns a rounded -0.0 into 0.0."""
    return frame.assign(**{c: frame[c].round(digits) + 0.0 for c in frame.select_dtypes("float").columns})


def _threshold_readme(destination, label, plan, y, chosen, record, points, calibration, models):
    v = record["validation_at_threshold"]
    table = points.assign(rule=np.where(points.locked, "**" + points.rule + " (locked)**", points.rule))[
        ["rule", "threshold", "recall", "precision", "specificity", "flagged_share", "flagged_per_1000",
         "cases_found_per_1000", "cases_missed_per_1000", "false_alarms_per_1000"]]
    table = table.assign(**{c: table[c].round().astype(int) for c in table.columns if c.endswith("_per_1000")})
    calib = pd.DataFrame([{"model": LABELS[name], "observed_rate": c["observed_rate"],
                           "mean_predicted_risk": c["mean_predicted_risk"], "brier": c["brier"],
                           "calibration_intercept": c["calibration_intercept"],
                           "calibration_slope": c["calibration_slope"]} for name, c in calibration.items()])
    check = pd.DataFrame([{"model": LABELS[name], "oof_pr_auc_now": m["oof_pr_auc_mean"],
                           "stage6_refit_check": m["stage6_pr_auc_mean"], "reproduces": m["reproduces_stage6"]}
                          for name, m in models.items()])
    decision = ("not needed: the selected model's risks already meet the rule, so they are used as they are"
                if record["recalibration"] is None else
                f"applied: Platt intercept {record['recalibration']['intercept']:.3f}, "
                f"slope {record['recalibration']['slope']:.3f}")
    text = [
        f"# Stage 7a: the locked cutoff ({label})", "",
        f"Everything on this page comes from the {len(y):,} training rows ({int(y.sum()):,} positive) and the "
        f"five saved folds of split `{plan.sha256[:12]}`. The {len(plan.test_index):,} test rows were not loaded.",
        "",
        f"**Locked cutoff: {record['threshold']:.4f}** on the predicted risk from {LABELS[chosen]} "
        f"(`{record['model_file']}`, SHA-256 `{record['model_sha256'][:12]}…`). Anyone at or above it is flagged.",
        "", f"Rule: {record['rule']}", "",
        f"On the out-of-fold predictions this cutoff catches {_percent(v['recall'])} of cases and flags "
        f"{_percent(v['flagged_share'])} of people; {_percent(v['precision'])} of the people it flags have the "
        f"condition. Per 1,000 people screened that is about {v['flagged_per_1000']:.0f} flagged, "
        f"{v['cases_found_per_1000']:.0f} cases found, {v['cases_missed_per_1000']:.0f} missed and "
        f"{v['false_alarms_per_1000']:.0f} false alarms.", "",
        "## Other cutoffs (for information only)", "",
        "What each alternative would cost on the same out-of-fold predictions. Only the locked row is used.", "",
        _markdown(table), "",
        "## Calibration on the out-of-fold predictions", "",
        f"Rule, fixed before this run: {record['calibration_rule']}", "",
        _markdown(_tidy(calib)), "", f"Recalibration: {decision}.", "",
        "![Out-of-fold calibration](calibration_validation.png)", "",
        "## Check against stage 6", "",
        "Refitting the saved pipelines fold by fold should reproduce the stage 6 refit check.", "",
        _markdown(check, digits=4), "",
    ]
    (destination / "README.md").write_text("\n".join(text), encoding="utf-8")


def original_classes(source, cohort):
    """Original three-class labels (0 none, 1 prediabetes, 2 diabetes), lined up with the cohort's rows.

    `prepare` keeps the source's row order and only drops the rows its endpoint excludes, so after the same
    exclusion the rows line up one to one. The file must be the one recorded in the provenance, and its
    recoded labels must reproduce the cohort's outcome row by row.
    """
    data = Path(source).read_bytes()
    metadata = cohort.provenance
    if hashlib.sha256(data).hexdigest() not in {metadata.get("source_sha256"), metadata.get("source_csv_sha256")}:
        raise ValueError("The three-class source does not match the file recorded in the provenance")
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = [n for n in archive.namelist() if Path(n).name == THREE_CLASS_FILE]
            if len(members) != 1:
                raise ValueError(f"Expected exactly one {THREE_CLASS_FILE} in the archive")
            data = archive.read(members[0])
    codes = pd.read_csv(io.BytesIO(data), usecols=["Diabetes_012"])["Diabetes_012"]
    mapped = codes.map(ENDPOINTS[cohort.endpoint])
    kept = mapped.notna().to_numpy()
    if kept.sum() != len(cohort.y) or not np.array_equal(mapped[kept].astype(int).to_numpy(), cohort.y.to_numpy()):
        raise ValueError("The three-class source rows do not line up with the prepared cohort")
    return codes.to_numpy()[kept]


def _locked_record(threshold_directory, training_directory, cohort, plan):
    """Read threshold.json and confirm it was made for this input, split and these exact model files."""
    record = _read_json(Path(threshold_directory) / "threshold.json")
    if record.get("input_sha256") != cohort.input_sha256 or record.get("endpoint") != cohort.endpoint:
        raise LockedTestSetError("threshold.json was made for a different input file or endpoint")
    models = {record["model"]: (record["model_file"], record["model_sha256"])}
    models.update({name: (ref["model_file"], ref["model_sha256"])
                   for name, ref in record["reference_operating_points"].items() if "model_file" in ref})
    paths = {}
    for name, (file, digest) in models.items():
        paths[name] = Path(training_directory) / "models" / file
        if not paths[name].exists() or _sha256(paths[name]) != digest:
            raise LockedTestSetError(f"{paths[name]} is missing or changed since the cutoff was locked")
    return record, paths


def run_test(source, provenance, split_directory, threshold_directory, training_directory, destination,
             three_class_source=None, n_resamples=1000, expected_source_rows=EXPECTED_ROWS, log=print):
    """Stage 7b: score the locked test set once, at the cutoffs saved in threshold.json."""
    destination = Path(destination)
    cohort = load_cohort(source, provenance, expected_source_rows)
    plan = load_split(split_directory, cohort)
    X, y = locked_test_data(cohort, plan, Path(threshold_directory) / "threshold.json")  # refuses if not locked
    record, paths = _locked_record(threshold_directory, training_directory, cohort, plan)
    original = original_classes(three_class_source, cohort)[plan.test_index] if three_class_source else None
    y = y.to_numpy()
    destination.mkdir(parents=True, exist_ok=False)
    label, chosen = endpoint_label(cohort.endpoint), record["model"]
    log(f"Scoring the locked test set once: {len(y):,} rows ({int(y.sum()):,} positive)")

    scores = {name: load_pipeline(path).predict_proba(X)[:, 1] for name, path in paths.items()}
    scores[CHECKLIST] = AdaRiskScore().predict_proba(X)[:, 1]
    cutoffs = {chosen: record["threshold"],
               **{name: ref["threshold"] for name, ref in record["reference_operating_points"].items()}}
    rows, points = [], {}
    for name, proba in scores.items():
        log(f"  {name}: metrics and {n_resamples:,} bootstrap resamples")
        result = evaluate(y, proba, threshold=cutoffs[name])
        intervals = bootstrap_intervals(y, proba, cutoffs[name], n_resamples, plan.seed)
        row = {"model": name, "role": "selected" if name == chosen else "reference", "threshold": cutoffs[name],
               "rows": result["rows"], "positives": result["positives"], "prevalence": result["prevalence"]}
        for metric in BOOTSTRAP_METRICS:
            row.update({metric: result[metric], f"{metric}_ci_low": intervals[metric][0],
                        f"{metric}_ci_high": intervals[metric][1]})
        rows.append({**row, **at_cutoff(y, proba, cutoffs[name])})
        points[name] = result
    results = pd.DataFrame(rows)

    calibrated = {name: scores[name] for name in paths}
    if record["recalibration"]:
        calibrated[f"{chosen}_recalibrated"] = recalibrate(scores[chosen], record["recalibration"])
    calibration = {name: calibration_summary(y, proba) for name, proba in calibrated.items()}
    reliability = pd.concat([reliability_table(y, proba).assign(model=name) for name, proba in calibrated.items()])
    subgroups = subgroup_table(X, y, scores[chosen], record["threshold"])
    missed = missed_case_profile(X, y, scores[chosen], record["threshold"])
    by_class = None if original is None else flagged_by_original_class(original, scores[chosen], record["threshold"])

    results.to_csv(destination / "test_results.csv", index=False)
    pd.DataFrame([{"model": name, **{k: v for k, v in c.items() if k != "platt"}} for name, c in calibration.items()]
                 ).to_csv(destination / "calibration_test.csv", index=False)
    reliability.to_csv(destination / "reliability_test.csv", index=False)
    subgroups.to_csv(destination / "subgroups.csv", index=False)
    missed.to_csv(destination / "missed_cases.csv", index=False)
    if by_class is not None:
        by_class.to_csv(destination / "flagged_by_original_class.csv", index=False)
    title = f"Locked test set ({label})"
    plot_curves(y, scores, points, destination / "curves.png", title)
    plot_calibration([(_series_label(name), _series_color(name), reliability[reliability.model == name])
                      for name in calibrated], destination / "calibration_test.png", f"Test-set calibration ({label})")
    plot_confusion(points[chosen], destination / "confusion_matrix.png",
                   f"{LABELS[chosen]} at the locked cutoff, test set\n{label}")
    overall = {"recall": points[chosen]["recall"], "false_positive_rate": 1 - points[chosen]["specificity"]}
    plot_subgroups(subgroups, overall, destination / "subgroups.png",
                   f"{LABELS[chosen]} by subgroup at the locked cutoff, test set ({label})", MODEL_COLORS[chosen])
    manifest = {
        "stage": "7b", "endpoint": cohort.endpoint, "input_sha256": cohort.input_sha256, "split_sha256": plan.sha256,
        "threshold_file_sha256": _sha256(Path(threshold_directory) / "threshold.json", text=True),
        "model": chosen, "model_sha256": record["model_sha256"], "threshold": record["threshold"],
        "test_rows_used": int(len(y)), "training_rows_used": 0,
        "bootstrap": {"resamples": n_resamples, "seed": plan.seed, "interval": "95% percentile, cutoff held fixed"},
        "original_classes_checked": original is not None,
        "python": platform.python_version(), "packages": {n: version(n) for n in PACKAGES + ["statsmodels"]},
    }
    write_json(destination / "manifest.json", manifest)
    _test_readme(destination, label, plan, record, results, calibration, subgroups, missed, by_class)
    return manifest


def _series_label(name):
    base = LABELS[name.removesuffix("_recalibrated")]
    return f"{base} (recalibrated)" if name.endswith("_recalibrated") else base


def _series_color(name):
    return "#8a8880" if name.endswith("_recalibrated") else MODEL_COLORS[name]


def _with_interval(row, metric):
    return f"{row[metric]:.3f} ({row[f'{metric}_ci_low']:.3f}–{row[f'{metric}_ci_high']:.3f})"


def _test_readme(destination, label, plan, record, results, calibration, subgroups, missed, by_class):
    chosen = record["model"]
    headline = pd.DataFrame({"model": results.model.map(LABELS), "cutoff": results.threshold.round(4)})
    for metric in ["recall", "precision", "specificity", "f2", "flagged_share", "pr_auc", "roc_auc"]:
        headline[metric] = [_with_interval(row, metric) for _, row in results.iterrows()]
    workload = results[["model", "flagged_per_1000", "cases_per_1000", "cases_found_per_1000",
                        "cases_missed_per_1000", "false_alarms_per_1000"]].assign(
        model=results.model.map(LABELS), flags_that_are_cases=results.precision.map(_percent))
    profile = missed.assign(**{column: [f"{value:,.0f}" if answer == "People" else f"{value:.3f}"
                                        for answer, value in zip(missed.answer, missed[column])]
                               for column in missed.columns if column != "answer"})
    main = results.set_index("model").loc[chosen]
    calib = pd.DataFrame([{"model": _series_label(name), "observed_rate": c["observed_rate"],
                           "mean_predicted_risk": c["mean_predicted_risk"], "brier": c["brier"],
                           "calibration_intercept": c["calibration_intercept"],
                           "calibration_slope": c["calibration_slope"]} for name, c in calibration.items()])
    groups = subgroups[["group", "level", "rows", "positives", "recall", "recall_ci_low", "recall_ci_high",
                        "false_positive_rate", "precision"]]
    text = [
        f"# Stage 7b: final results on the locked test set ({label})", "",
        f"The {int(main['rows']):,} test rows of split `{plan.sha256[:12]}` ({int(main['positives']):,} positive) "
        f"were scored once, at the cutoffs saved in `threshold.json` before any test row was read. Intervals are "
        "95% bootstrap percentile intervals (respondents resampled with replacement, cutoff held fixed).", "",
        f"**{LABELS[chosen]} at the locked cutoff ({record['threshold']:.4f})** catches {_percent(main['recall'])} "
        f"of cases (target {record['target_recall']:.0%}) and flags {_percent(main['flagged_share'])} of people; "
        f"{_percent(main['precision'])} of the people it flags have the condition.", "",
        "## Headline metrics", "",
        "Logistic regression and the paper checklist are shown for context only; neither was used to decide "
        "anything here. The checklist's \"probability\" is points / 9, so its Brier score is left out.", "",
        _markdown(headline), "",
        "## Per 1,000 people screened", "",
        _markdown(workload, digits=1), "",
        "![Confusion matrix](confusion_matrix.png)", "",
        "![PR and ROC curves](curves.png)", "",
        "## Calibration on the test set", "",
        _markdown(_tidy(calib)), "",
        "![Test-set calibration](calibration_test.png)", "",
        "## Subgroups (selected model, locked cutoff)", "",
        "Wilson 95% intervals. Counts are shown because small groups give wide intervals.", "",
        _markdown(groups), "",
        "![Subgroups](subgroups.png)", "",
        "## Who the model misses", "",
        "Average answers of the cases it caught, the cases it missed, and everyone without the condition.", "",
        _markdown(profile), "",
    ]
    if by_class is not None:
        text += ["Share flagged within each original survey class (for the two positive classes this is recall):", "",
                 _markdown(by_class[["label", "rows", "flagged", "not_flagged", "flagged_share",
                                     "flagged_share_ci_low", "flagged_share_ci_high"]]), ""]
    (destination / "README.md").write_text("\n".join(text), encoding="utf-8")

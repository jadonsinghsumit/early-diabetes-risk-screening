"""Command line for roadmap stages 5-7: `diabetes-screening-model split | imbalance | train | threshold | test`.

This is a separate entry point from Sumit's `diabetes-screening` command (stages 1-4), so his
code stays untouched. Both commands use the same conventions: a new output folder every time,
input files checked against their provenance, and a non-zero exit status when something is wrong.

AI-use disclosure: drafted with AI assistance (Claude, Anthropic); see docs/splitting.md and docs/evaluation.md.
"""
import argparse
from pathlib import Path
import sys
import zipfile

from ..schema import EXPECTED_ROWS
from .imbalance import MODELS
from .metrics import SCREENING_RECALL
from .split import LockedTestSetError


def _model_list(text):
    models = [name.strip() for name in text.split(",") if name.strip()]
    unknown = sorted(set(models) - set(MODELS))
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}; choose from {list(MODELS)}")
    return models


def _parser():
    parser = argparse.ArgumentParser(prog="diabetes-screening-model",
                                     description="BRFSS diabetes screening: roadmap stages 5 to 7")
    commands = parser.add_subparsers(dest="command", required=True)

    split = commands.add_parser("split", help="Stage 5: stratified train/test split and fixed CV folds")
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--test-size", type=float, default=0.2)
    split.add_argument("--folds", type=int, default=5)
    split.add_argument("--group-profiles", action="store_true",
                       help="Keep respondents with identical answers on one side (sensitivity option)")

    imbalance = commands.add_parser("imbalance", help="Stage 5: weighted vs unweighted models on the CV folds")
    imbalance.add_argument("--models", default=",".join(MODELS),
                           help="Comma-separated subset of logistic_regression, random_forest, xgboost")
    imbalance.add_argument("--skip-grouped-check", action="store_true",
                           help="Skip the profile-grouped cross-validation comparison")

    train = commands.add_parser("train", help="Stage 6: tune, compare and interpret models on the saved folds")
    train.add_argument("--models", default=",".join(MODELS))
    train.add_argument("--n-iter", type=int, help="Candidates per search (default: 12 LR, 8 RF, 16 XGBoost)")
    train.add_argument("--skip-importance", action="store_true", help="Skip permutation importance (faster)")

    threshold = commands.add_parser("threshold", help="Stage 7: lock the cutoff on out-of-fold predictions")
    threshold.add_argument("--target-recall", type=float, default=SCREENING_RECALL,
                           help="Share of training cases the cutoff must flag (default 0.8)")

    test = commands.add_parser("test", help="Stage 7: score the locked test set once, at the saved cutoff")
    test.add_argument("--threshold", required=True, type=Path,
                      help="Folder written by `threshold`; commit it before running this")
    test.add_argument("--source", type=Path,
                      help="Original three-class file, to split missed cases into prediabetes and diabetes")
    test.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples for the intervals")

    for command in (split, imbalance, train, threshold, test):
        command.add_argument("--input", required=True, type=Path, help="Prepared CSV written by `prepare`")
        command.add_argument("--provenance", required=True, type=Path, help="Its .provenance.json file")
        command.add_argument("--output", required=True, type=Path, help="New folder; existing ones are never reused")
        command.add_argument("--synthetic-rows", type=int, help=argparse.SUPPRESS)  # test fixtures only
    for command in (imbalance, train, threshold, test):
        command.add_argument("--split", required=True, type=Path, help="Folder written by `split`")
    for command in (threshold, test):
        command.add_argument("--training", required=True, type=Path, help="Folder written by `train`")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    source_rows = args.synthetic_rows or EXPECTED_ROWS
    try:
        if args.command == "split":
            from .split_workflow import run_split
            summary = run_split(args.input, args.provenance, args.output, args.seed, args.test_size, args.folds,
                                args.group_profiles, source_rows)
            print(f"Split {summary['rows']:,} rows ({summary['strategy']}): train {summary['train']['rows']:,} "
                  f"({summary['train']['positives']:,} positive), test {summary['test']['rows']:,} "
                  f"({summary['test']['positives']:,} positive, locked). Outputs: {args.output}")
        elif args.command == "imbalance":
            from .split_workflow import run_imbalance
            run_imbalance(args.input, args.provenance, args.split, args.output, _model_list(args.models),
                          not args.skip_grouped_check, source_rows)
            print(f"Finished the stage 5 comparison. Outputs: {args.output}")
        elif args.command == "train":
            from .training_workflow import run_training
            run_training(args.input, args.provenance, args.split, args.output, _model_list(args.models),
                         args.n_iter, not args.skip_importance, expected_source_rows=source_rows)
            print(f"Finished stage 6. Outputs: {args.output}")
        elif args.command == "threshold":
            from .evaluation_workflow import run_threshold
            record = run_threshold(args.input, args.provenance, args.split, args.training, args.output,
                                   args.target_recall, source_rows)
            print(f"Locked cutoff {record['threshold']:.4f} for {record['model']}. Commit {args.output} "
                  "before running `test`.")
        else:
            from .evaluation_workflow import run_test
            manifest = run_test(args.input, args.provenance, args.split, args.threshold, args.training, args.output,
                                args.source, args.bootstrap, source_rows)
            print(f"Scored {manifest['test_rows_used']:,} test rows once. Outputs: {args.output}")
    except (OSError, ValueError, zipfile.BadZipFile, LockedTestSetError) as error:
        print(f"Stopped: {error}", file=sys.stderr)
        return 1
    return 0

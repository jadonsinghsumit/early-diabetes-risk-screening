"""Roadmap stages 5+: leakage-safe splitting, preprocessing, scoring and model comparison.

Typical use in stage 6 (tuning) and stage 7 (threshold):

    cohort = load_cohort("data/derived/combined_BRFSS2015.csv",
                         "data/derived/combined_BRFSS2015.provenance.json")
    plan = load_split("reports/modeling/combined/split", cohort)
    X_train, y_train, folds = training_data(cohort, plan)
    search = RandomizedSearchCV(pipeline, space, cv=plan.cv(), scoring="average_precision")
"""
from .cohort import FEATURE_SETS, Cohort, load_cohort, profile_summary
from .imbalance import MODELS, compare_weighting, model_factory, out_of_fold
from .metrics import evaluate
from .preprocess import build_preprocessor
from .split import (SplitPlan, LockedTestSetError, load_split, locked_test_data, make_split, save_split,
                    training_data)

__all__ = ["FEATURE_SETS", "MODELS", "Cohort", "SplitPlan", "LockedTestSetError", "build_preprocessor",
           "compare_weighting", "evaluate", "load_cohort", "load_split", "locked_test_data", "make_split",
           "model_factory", "out_of_fold", "profile_summary", "save_split", "training_data"]

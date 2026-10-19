"""Stage 5: one train/test split everyone shares, fixed cross-validation folds, and a locked test set.

Every row gets one code in `fold`: -1 means the locked test set, 0..k-1 the cross-validation
fold of a training row. Saving that single array (plus seed and input hash) lets every teammate
reuse the exact same split, and lets later stages prove which rows they touched.

AI-use disclosure: drafted with AI assistance (Claude, Anthropic); see docs/splitting.md.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import PredefinedSplit, StratifiedGroupKFold, StratifiedKFold, train_test_split

from ..ingestion import write_json

TEST = -1
ASSIGNMENT_FILE = "split_assignment.csv.gz"
STRATEGIES = {"stratified", "stratified-group"}


class LockedTestSetError(RuntimeError):
    """The held-out test set was requested before a threshold was locked for this split."""


@dataclass(frozen=True)
class SplitPlan:
    fold: np.ndarray
    seed: int
    test_size: float
    n_splits: int
    strategy: str
    endpoint: str
    input_sha256: str

    @property
    def train_index(self):
        return np.flatnonzero(self.fold != TEST)

    @property
    def test_index(self):
        return np.flatnonzero(self.fold == TEST)

    @property
    def train_folds(self):
        """CV fold of each training row, in training-row order."""
        return self.fold[self.train_index]

    @property
    def sha256(self):
        return hashlib.sha256(np.ascontiguousarray(self.fold, dtype="<i1").tobytes()).hexdigest()

    def cv(self):
        """scikit-learn splitter over the training rows, for cross_validate or RandomizedSearchCV."""
        return PredefinedSplit(self.train_folds)


def assign_folds(y, n_splits=5, seed=42, groups=None):
    """Stratified fold number per row; with `groups`, each group stays inside one fold."""
    y = np.asarray(y)
    if groups is None:
        splitter, split_args = StratifiedKFold(n_splits, shuffle=True, random_state=seed), (y, y)
    else:
        splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
        split_args = (y, y, np.asarray(groups))
    folds = np.full(len(y), TEST, dtype=np.int8)
    for number, (_, held_out) in enumerate(splitter.split(*split_args)):
        folds[held_out] = number
    return folds


def make_split(cohort, seed=42, test_size=0.2, n_splits=5, group_profiles=False):
    """Stratified train/test split, then stratified CV folds inside the training rows only.

    `group_profiles=True` keeps respondents with identical answers on one side of every split.
    It needs 1 / test_size to be a whole number, because the test set is one group-aware fold.
    """
    y = cohort.y.to_numpy()
    rows = np.arange(len(y))
    if group_profiles:
        parts = round(1 / test_size)
        if not np.isclose(parts * test_size, 1):
            raise ValueError("Grouped splitting needs test_size = 1/k, for example 0.2")
        outer = assign_folds(y, parts, seed, groups=cohort.profile_id.to_numpy())
        train, test = rows[outer != 0], rows[outer == 0]
    else:
        train, test = train_test_split(rows, test_size=test_size, stratify=y, random_state=seed)
    fold = np.full(len(y), TEST, dtype=np.int8)
    groups = cohort.profile_id.to_numpy()[train] if group_profiles else None
    fold[np.sort(train)] = assign_folds(y[np.sort(train)], n_splits, seed, groups)
    return SplitPlan(fold=fold, seed=seed, test_size=test_size, n_splits=n_splits,
                     strategy="stratified-group" if group_profiles else "stratified",
                     endpoint=cohort.endpoint, input_sha256=cohort.input_sha256)


def describe_split(plan, cohort):
    """Counts per side and per fold, plus how many test rows share a profile with training rows."""
    y, profiles = cohort.y.to_numpy(), cohort.profile_id.to_numpy()
    train, test = plan.train_index, plan.test_index

    def side(index):
        return {"rows": int(len(index)), "positives": int(y[index].sum()),
                "prevalence": float(y[index].mean()) if len(index) else None}

    overlap = np.isin(profiles[test], profiles[train])
    return {
        "rows": int(len(y)), "train": side(train), "test": side(test),
        "folds": [{"fold": k, **side(train[plan.train_folds == k])} for k in range(plan.n_splits)],
        "test_rows_sharing_a_profile_with_training": int(overlap.sum()),
        "share_of_test_rows_sharing_a_profile": float(overlap.mean()) if len(test) else 0.0,
    }


def save_split(plan, cohort, directory):
    """Write split.json (readable summary) and the per-row assignment; never write feature data."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    assignment = pd.DataFrame({"row": np.arange(len(plan.fold)), "fold": plan.fold})
    # mtime=0 and Unix line endings keep the gzip bytes identical across runs and platforms.
    assignment.to_csv(directory / ASSIGNMENT_FILE, index=False, compression={"method": "gzip", "mtime": 0},
                      lineterminator="\n")
    summary = {
        "endpoint": plan.endpoint, "input_sha256": plan.input_sha256, "strategy": plan.strategy,
        "seed": plan.seed, "test_size": plan.test_size, "n_splits": plan.n_splits,
        "assignment_file": ASSIGNMENT_FILE, "assignment_sha256": plan.sha256,
        "fold_codes": {"-1": "locked test set", "0..k-1": "cross-validation fold of a training row"},
        **describe_split(plan, cohort),
        "scikit_learn": sklearn.__version__,
        "test_set_policy": ("Read the test rows only through locked_test_data(), after threshold.json "
                            "records this assignment_sha256 (roadmap stage 7)."),
    }
    write_json(directory / "split.json", summary)
    return summary


def load_split(directory, cohort):
    """Reload a saved split and refuse it if the cohort or the assignment changed since saving."""
    directory = Path(directory)
    summary = json.loads((directory / "split.json").read_text(encoding="utf-8"))
    if summary["input_sha256"] != cohort.input_sha256 or summary["endpoint"] != cohort.endpoint:
        raise ValueError("Saved split belongs to a different input file or endpoint")
    fold = pd.read_csv(directory / summary["assignment_file"])["fold"].to_numpy(dtype=np.int8)
    plan = SplitPlan(fold=fold, seed=summary["seed"], test_size=summary["test_size"],
                     n_splits=summary["n_splits"], strategy=summary["strategy"],
                     endpoint=summary["endpoint"], input_sha256=summary["input_sha256"])
    if len(fold) != len(cohort.y) or plan.sha256 != summary["assignment_sha256"]:
        raise ValueError("Split assignment does not match split.json; it may have been edited")
    return plan


def training_data(cohort, plan):
    """Training features and labels plus the fold of each training row. Never includes test rows."""
    index = plan.train_index
    return cohort.X.iloc[index], cohort.y.iloc[index], plan.train_folds


def locked_test_data(cohort, plan, threshold_path):
    """Release the test rows only after a threshold for this exact split has been saved.

    The threshold file must exist and carry `split_sha256` equal to this plan's assignment hash, so
    the operating point is provably chosen on training data before any test row is scored.
    """
    path = Path(threshold_path)
    if not path.exists():
        raise LockedTestSetError(f"{path} does not exist; lock the threshold on training folds first")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("split_sha256") != plan.sha256 or "threshold" not in record:
        raise LockedTestSetError("threshold.json is missing a threshold or was made for a different split")
    index = plan.test_index
    return cohort.X.iloc[index], cohort.y.iloc[index]

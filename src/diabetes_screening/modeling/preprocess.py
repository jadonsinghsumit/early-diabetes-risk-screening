"""Preprocessing that can't leak: every transformation lives inside the model pipeline.

The scalers sit inside a scikit-learn Pipeline, so when a pipeline is fitted on a training fold it
learns medians, means and spreads from that fold alone. Validation and test rows only ever get
transformed, never learned from.

AI-use disclosure: drafted with AI assistance (Claude, Anthropic); see docs/splitting.md.
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

from ..schema import FIELDS
from .cohort import BINARY, FEATURE_SETS, HEALTH_DAYS, ORDINAL

KINDS = {"linear", "linear-onehot", "tree"}
_CODES = {field.name: list(range(field.minimum, field.maximum + 1)) for field in FIELDS}


def build_preprocessor(kind="linear", features=None):
    """Column transformer for one model family. Columns not listed in `features` are dropped.

    linear: BMI is robust-scaled (median and IQR) because its tail runs out to 98. Health days and
        the four ordinal codes (age band, general health, education, income) are standardized and
        kept as ordered numbers. The 14 yes/no answers pass through as 0/1. Putting everything on a
        similar scale keeps the L2 penalty in logistic regression fair across features.
    linear-onehot: same, except the ordinal codes become one column per level (first level dropped),
        so logistic regression can learn a risk curve that isn't a straight line across age bands.
    tree: every selected column passes through untouched, because tree splits don't care about scale.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    features = list(FEATURE_SETS["full"] if features is None else features)

    def keep(columns):
        return [name for name in columns if name in features]

    if kind == "tree":
        steps = [("raw", "passthrough", features)]
    else:
        ordinal = keep(ORDINAL)
        if kind == "linear":
            ordinal_step = ("standardized_ordinal", StandardScaler(), ordinal)
        else:
            # Fixed codebook categories, so every fold builds exactly the same columns.
            ordinal_step = ("onehot_ordinal", OneHotEncoder(categories=[_CODES[n] for n in ordinal], drop="first",
                                                            sparse_output=False), ordinal)
        steps = [("bmi", RobustScaler(), keep(["BMI"])),
                 ("standardized", StandardScaler(), keep(HEALTH_DAYS)),
                 ordinal_step,
                 ("binary", "passthrough", keep(BINARY))]
        steps = [step for step in steps if step[2]]
    return ColumnTransformer(steps, remainder="drop", verbose_feature_names_out=False)


class ScreeningPreprocessor(TransformerMixin, BaseEstimator):
    """`build_preprocessor` wrapped as an estimator, so `kind` and `features` become pipeline settings.

    That lets a hyperparameter search try `preprocess__kind`, and lets us refit a tuned model on a
    smaller feature set with `set_params(preprocess__features=...)`.
    """

    def __init__(self, kind="linear", features=None):
        self.kind = kind
        self.features = features

    def fit(self, X, y=None):
        self.transformer_ = build_preprocessor(self.kind, self.features).fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self, "transformer_")
        return self.transformer_.transform(X)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "transformer_")
        return self.transformer_.get_feature_names_out()

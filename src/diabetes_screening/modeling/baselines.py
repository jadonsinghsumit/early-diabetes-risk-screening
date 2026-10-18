"""Two yardsticks every model has to beat before it's worth anything.

1. No model: everyone gets the same predicted risk (the training prevalence). Its PR-AUC is just the
   prevalence, so it marks the floor.
2. A paper checklist: the "Diabetes Risk Test" that NIDDK publishes, adapted from the American Diabetes
   Association's test, using the answers this survey has. Clinics already use this kind of score, so
   it's the bar a model has to clear to be worth the extra complexity.
   Source: NIDDK's "Diabetes Risk Test" PDF (link in docs/training.md).

AI-use disclosure: drafted with AI assistance (Claude, Anthropic); see docs/training.md.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier

ADA_MAX_POINTS = 9
ADA_HIGH_RISK_POINTS = 5  # the paper test calls 5 or more "increased risk"


def ada_points(X):
    """Points from the adapted Diabetes Risk Test (0-9 here; the paper version tops out at 10).

    Age: under 40 = 0, 40-49 = 1, 50-59 = 2, 60+ = 3 (survey age bands 5-6, 7-8, 9-13).
    Male = 1. Told blood pressure is high = 1. No physical activity in the past 30 days = 1.
    Weight: BMI 25-29.9 = 1, 30-39.9 = 2, 40+ = 3. The paper test uses a height-weight chart whose
    bands work out to these BMI cut points (at 5'0": 128, 153 and 204 lb).
    Family history and gestational diabetes (1 point each on paper) aren't in the survey, so some
    people score a point lower here than they would on paper.
    """
    age = np.select([X["Age"] >= 9, X["Age"] >= 7, X["Age"] >= 5], [3, 2, 1], default=0)
    weight = np.select([X["BMI"] >= 40, X["BMI"] >= 30, X["BMI"] >= 25], [3, 2, 1], default=0)
    return age + X["Sex"].to_numpy() + X["HighBP"].to_numpy() + (1 - X["PhysActivity"].to_numpy()) + weight


class AdaRiskScore(ClassifierMixin, BaseEstimator):
    """The checklist as a scikit-learn classifier, so it runs through exactly the same evaluation code.

    It learns nothing from the data. Its "probability" is points / 9, which ranks people correctly
    but is not a calibrated risk, so read its PR-AUC and operating-point numbers, not its Brier score.
    """

    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        score = ada_points(X) / ADA_MAX_POINTS
        return np.column_stack([1 - score, score])

    def predict(self, X):
        return (ada_points(X) >= ADA_HIGH_RISK_POINTS).astype(int)


BASELINES = {
    "no_model": lambda y_fit: DummyClassifier(strategy="prior"),
    "ada_checklist": lambda y_fit: AdaRiskScore(),
}
ADA_CUTOFF = (ADA_HIGH_RISK_POINTS - 0.5) / ADA_MAX_POINTS  # sits between 4 and 5 points, so 5+ is flagged

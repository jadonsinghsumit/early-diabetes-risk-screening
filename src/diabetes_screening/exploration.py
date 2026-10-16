"""Descriptive tables and headless plots for the cleaned analysis cohort."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .schema import FIELDS, TARGET

DISTRIBUTIONS = ["BMI", "Age", "MentHlth", "PhysHlth", "GenHlth"]


def _save(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def explore(frame, destination, endpoint="combined"):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    counts = frame[TARGET].value_counts().reindex([0, 1], fill_value=0)
    balance = pd.DataFrame({"count": counts, "fraction": counts / len(frame)})
    balance.index.name = TARGET
    balance.to_csv(destination / "class_balance.csv")
    figure, axis = plt.subplots(figsize=(6, 4))
    labels = (["No diabetes", "Prediabetes or diabetes"] if endpoint == "combined"
              else ["No diabetes or prediabetes", "Diabetes"])
    axis.bar(labels, counts)
    axis.set_ylabel("Respondents")
    axis.set_title("Class balance in the analysis table (unweighted)")
    _save(figure, destination / "class_balance.png")

    group_summary = frame.groupby(TARGET)[DISTRIBUTIONS].agg(["count", "mean", "median", "std"])
    group_summary.to_csv(destination / "group_summary.csv")
    for column in DISTRIBUTIONS:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(data=frame, x=column, hue=TARGET, stat="density", common_norm=False,
                     discrete=True, element="step", ax=axes[0])
        sns.boxplot(data=frame, x=TARGET, y=column, showfliers=False, ax=axes[1])
        axes[1].set_title("Boxplot (outliers hidden for display only)")
        _save(figure, destination / f"distribution_{column}.png")

    categorical = [field.name for field in FIELDS if field.kind in {"binary", "ordinal"}
                   and field.name != TARGET]
    associations = []
    figure, axes = plt.subplots(6, 3, figsize=(14, 22))
    for column, axis in zip(categorical, axes.flat):
        table = frame.groupby(column)[TARGET].agg(["count", "sum", "mean"]).reset_index()
        table.insert(0, "feature", column)
        table = table.rename(columns={column: "code", "sum": "positive_count", "mean": "positive_fraction"})
        associations.append(table)
        axis.bar(table["code"].astype(str), table["positive_fraction"])
        axis.set(title=column, xlabel="Source code", ylabel="Positive fraction", ylim=(0, 1))
    for axis in list(axes.flat)[len(categorical):]:
        axis.set_visible(False)
    _save(figure, destination / "categorical_factors.png")
    pd.concat(associations, ignore_index=True).to_csv(destination / "categorical_factors.csv", index=False)

    for method in ["pearson", "spearman"]:
        correlations = frame.corr(method=method)
        correlations.to_csv(destination / f"correlations_{method}.csv")
        figure, axis = plt.subplots(figsize=(13, 11))
        sns.heatmap(correlations, cmap="vlag", center=0, vmin=-1, vmax=1, ax=axis)
        axis.set_title(f"{method.title()} correlations on original coded variables")
        _save(figure, destination / f"correlations_{method}.png")

    prevalence = balance.loc[1, "fraction"]
    medians = frame.groupby(TARGET)[DISTRIBUTIONS].median()
    observations = ["# Exploratory observations", "",
                    f"Analysis cohort: {len(frame):,} rows; positive fraction: {prevalence:.2%}.",
                    f"Binary endpoint: {endpoint}.",
                    "This is the observed table fraction, not a survey-weighted population estimate.", ""]
    if {0, 1}.issubset(medians.index):
        for column in DISTRIBUTIONS:
            observations.append(f"- {column}: median {medians.loc[0, column]:g} in group 0; "
                                f"{medians.loc[1, column]:g} in group 1.")
    observations += ["", "Age and general health are ordinal codes. Correlations depend on coding; "
                     "use Spearman as an ordered-rank complement to Pearson. Group differences can "
                     "reflect age, healthcare access, and other confounding. No causal or diagnostic "
                     "conclusions follow. Later modeling must keep hold-out data isolated; this full-cohort "
                     "exploration is descriptive and must not be used to tune test-set decisions."]
    (destination / "observations.md").write_text("\n".join(observations) + "\n", encoding="utf-8")
    return balance

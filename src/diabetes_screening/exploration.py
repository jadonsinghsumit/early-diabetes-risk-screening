"""Descriptive tables and headless plots for the cleaned analysis cohort."""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .schema import FIELDS, TARGET
from .endpoints import LABELS

DISTRIBUTIONS = ["BMI", "Age", "MentHlth", "PhysHlth", "GenHlth"]


def _save(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def explore(frame, destination, endpoint="combined"):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    counts = frame[TARGET].value_counts().reindex([0, 1], fill_value=0)
    balance = pd.DataFrame({"count": counts, "fraction": counts / len(frame)})
    balance.index.name = TARGET
    balance.to_csv(destination / "class_balance.csv")
    labels = LABELS[endpoint]
    figure, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(labels, counts)
    for bar, count, fraction in zip(bars, counts, balance["fraction"]):
        axis.annotate(f"{count:,} ({fraction:.2%})", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha="center", va="bottom", fontsize=9)
    axis.set_ylim(0, max(counts.max(), 1) * 1.12)
    axis.set_xlabel("Outcome group")
    axis.set_ylabel("Respondents")
    axis.set_title("Class balance in the analysis table (unweighted)")
    _save(figure, destination / "class_balance.png")

    # Plots show class names instead of the legacy 0/1 codes; tables keep the original codes.
    named = frame.assign(Group=frame[TARGET].map(dict(enumerate(labels))))

    group_summary = frame.groupby(TARGET)[DISTRIBUTIONS].agg(["count", "mean", "median", "std"])
    group_summary.to_csv(destination / "group_summary.csv")
    for column in DISTRIBUTIONS:
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(data=named, x=column, hue="Group", hue_order=labels, stat="density", common_norm=False,
                     discrete=True, element="step", ax=axes[0])
        axes[0].set(title=f"{column}: distribution by group (each group sums to 1)", xlabel=column, ylabel="Density")
        sns.boxplot(data=named, x="Group", y=column, order=labels, showfliers=False, ax=axes[1])
        axes[1].set(title="Boxplot (outliers hidden for display only)", xlabel="Outcome group", ylabel=column)
        _save(figure, destination / f"distribution_{column}.png")

    categorical = [field.name for field in FIELDS if field.kind in {"binary", "ordinal"}
                   and field.name != TARGET]
    if len(categorical) > 18:
        raise ValueError("categorical_factors.png has room for 18 panels; enlarge the grid")
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
    factors = pd.concat(associations, ignore_index=True)
    medians = frame.groupby(TARGET)[DISTRIBUTIONS].median()
    observations = ["# Exploratory observations", "",
                    f"Analysis cohort: {len(frame):,} rows; positive fraction: {prevalence:.2%}.",
                    f"Binary endpoint: {endpoint}.",
                    "This is the observed table fraction, not a survey-weighted population estimate.", ""]
    if {0, 1}.issubset(medians.index):
        for column in DISTRIBUTIONS:
            observations.append(f"- {column}: median {medians.loc[0, column]:g} in group 0; "
                                f"{medians.loc[1, column]:g} in group 1.")
    spearman = frame.corr(method="spearman")[TARGET].drop(TARGET).dropna()
    if len(spearman):
        top = spearman.reindex(spearman.abs().sort_values(ascending=False).index)[:5]
        observations += ["", "Strongest Spearman correlations with the outcome (descriptive; coded variables): "
                         + "; ".join(f"{name} {value:+.3f}" for name, value in top.items()) + "."]
    ratios = []
    for field in FIELDS:
        if field.kind == "binary" and field.name != TARGET:
            rates = factors.loc[factors["feature"] == field.name].set_index("code")["positive_fraction"]
            if {0, 1}.issubset(rates.index) and rates[0] > 0:
                ratios.append((field.name, rates[1] / rates[0], rates[1], rates[0]))
    ratios.sort(key=lambda item: abs(math.log(item[1])) if item[1] > 0 else 0, reverse=True)
    if ratios:
        observations += ["", "Largest unadjusted prevalence ratios among binary factors (code 1 vs code 0; "
                         "not odds ratios): " + "; ".join(
                             f"{name} {ratio:.2f}x ({high:.2%} vs {low:.2%})" for name, ratio, high, low in ratios[:5]) + "."]
    observations += ["", "Age and general health are ordinal codes. Correlations depend on coding; "
                     "use Spearman as an ordered-rank complement to Pearson. Group differences can "
                     "reflect age, healthcare access, and other confounding. No causal or diagnostic "
                     "conclusions follow. Later modeling must keep hold-out data isolated; this full-cohort "
                     "exploration is descriptive and must not be used to tune test-set decisions."]
    (destination / "observations.md").write_text("\n".join(observations) + "\n", encoding="utf-8")
    return balance

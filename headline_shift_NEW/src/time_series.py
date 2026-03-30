"""
Time-series analysis and visualization of ideology/sentiment trends.
Generates publication-quality plots with confidence intervals.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from src.utils import PLOTS_DIR, TARGET_PUBLICATIONS, ELECTION_YEARS, YEAR_RANGE

# publication color palette — matches conventional associations
PUB_COLORS = {
    "CNN": "#CC0000",
    "Fox News": "#003366",
    "Washington Post": "#2E8B57",
    "New York Times": "#999999",
}

# style setup
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "figure.figsize": (12, 6),
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})


def _add_election_markers(ax, y_range=None):
    """Add vertical lines for election years."""
    for year in ELECTION_YEARS:
        if YEAR_RANGE[0] <= year <= YEAR_RANGE[1]:
            ax.axvline(x=year, color="gray", linestyle="--", alpha=0.4, linewidth=1)
            label = "Pres." if year % 4 == 0 else "Mid."
            if y_range is not None:
                ax.text(year, y_range[1] * 0.95, f" {label}", fontsize=8,
                        color="gray", alpha=0.7, va="top")


def _save_plot(fig, name: str):
    path = os.path.join(PLOTS_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ Saved → {path}")
    return path


def plot_ideology_trends(df: pd.DataFrame) -> str:
    """
    Plot ideology numeric score (-1=left, 0=center, +1=right) over time per publication.
    Aggregated quarterly with 95% CI.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    for pub in TARGET_PUBLICATIONS:
        sub = df[df["publication"] == pub].copy()
        if len(sub) == 0:
            continue
        # quarterly aggregation
        sub["year_q"] = sub["date"].dt.to_period("Q")
        quarterly = sub.groupby("year_q")["ideology_numeric"].agg(["mean", "std", "count"])
        quarterly = quarterly[quarterly["count"] >= 5]  # min sample size
        quarterly["se"] = quarterly["std"] / np.sqrt(quarterly["count"])
        quarterly["ci95"] = 1.96 * quarterly["se"]
        quarterly.index = quarterly.index.to_timestamp()

        ax.plot(quarterly.index, quarterly["mean"], label=pub,
                color=PUB_COLORS.get(pub, None), linewidth=2)
        ax.fill_between(quarterly.index,
                         quarterly["mean"] - quarterly["ci95"],
                         quarterly["mean"] + quarterly["ci95"],
                         alpha=0.15, color=PUB_COLORS.get(pub, None))

    _add_election_markers(ax, y_range=(-1, 1))
    ax.set_xlabel("Time")
    ax.set_ylabel("Ideology Score (Left ← → Right)")
    ax.set_title("Ideology Trend by Publication (Quarterly, 95% CI)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    return _save_plot(fig, "ideology_trends")


def plot_sentiment_trends(df: pd.DataFrame) -> str:
    """Plot VADER sentiment compound score over time per publication."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for pub in TARGET_PUBLICATIONS:
        sub = df[df["publication"] == pub].copy()
        if len(sub) == 0:
            continue
        sub["year_q"] = sub["date"].dt.to_period("Q")
        quarterly = sub.groupby("year_q")["sentiment_score"].agg(["mean", "std", "count"])
        quarterly = quarterly[quarterly["count"] >= 5]
        quarterly["se"] = quarterly["std"] / np.sqrt(quarterly["count"])
        quarterly["ci95"] = 1.96 * quarterly["se"]
        quarterly.index = quarterly.index.to_timestamp()

        ax.plot(quarterly.index, quarterly["mean"], label=pub,
                color=PUB_COLORS.get(pub, None), linewidth=2)
        ax.fill_between(quarterly.index,
                         quarterly["mean"] - quarterly["ci95"],
                         quarterly["mean"] + quarterly["ci95"],
                         alpha=0.15, color=PUB_COLORS.get(pub, None))

    _add_election_markers(ax, y_range=(-1, 1))
    ax.set_xlabel("Time")
    ax.set_ylabel("Sentiment Score (Negative ← → Positive)")
    ax.set_title("Sentiment Trend by Publication (Quarterly, 95% CI)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    return _save_plot(fig, "sentiment_trends")


def plot_emotionality_trends(df: pd.DataFrame) -> str:
    """Plot emotionality (|compound|) over time — are headlines getting more emotional?"""
    fig, ax = plt.subplots(figsize=(14, 7))

    for pub in TARGET_PUBLICATIONS:
        sub = df[df["publication"] == pub].copy()
        if len(sub) == 0:
            continue
        sub["year_q"] = sub["date"].dt.to_period("Q")
        quarterly = sub.groupby("year_q")["emotionality"].agg(["mean", "std", "count"])
        quarterly = quarterly[quarterly["count"] >= 5]
        quarterly["se"] = quarterly["std"] / np.sqrt(quarterly["count"])
        quarterly["ci95"] = 1.96 * quarterly["se"]
        quarterly.index = quarterly.index.to_timestamp()

        ax.plot(quarterly.index, quarterly["mean"], label=pub,
                color=PUB_COLORS.get(pub, None), linewidth=2)
        ax.fill_between(quarterly.index,
                         quarterly["mean"] - quarterly["ci95"],
                         quarterly["mean"] + quarterly["ci95"],
                         alpha=0.15, color=PUB_COLORS.get(pub, None))

    _add_election_markers(ax, y_range=(0, 1))
    ax.set_xlabel("Time")
    ax.set_ylabel("Emotionality (|Sentiment|)")
    ax.set_title("Headline Emotionality Over Time (Quarterly, 95% CI)")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    return _save_plot(fig, "emotionality_trends")


def plot_ideology_by_topic(df: pd.DataFrame) -> str:
    """Heatmap: mean ideology score by publication × topic."""
    pivot = df.pivot_table(
        values="ideology_numeric", index="topic_area", columns="publication",
        aggfunc="mean"
    )
    # reorder columns to standard order
    cols = [c for c in TARGET_PUBLICATIONS if c in pivot.columns]
    pivot = pivot[cols]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Mean Ideology Score by Topic × Publication")
    ax.set_ylabel("Topic Area")
    ax.set_xlabel("")
    fig.tight_layout()
    return _save_plot(fig, "ideology_by_topic")


def plot_sentiment_distribution(df: pd.DataFrame) -> str:
    """Violin plot of sentiment distributions per publication."""
    fig, ax = plt.subplots(figsize=(12, 6))
    pubs_present = [p for p in TARGET_PUBLICATIONS if p in df["publication"].values]
    sub = df[df["publication"].isin(pubs_present)]

    palette = [PUB_COLORS.get(p, "#888") for p in pubs_present]
    sns.violinplot(data=sub, x="publication", y="sentiment_score",
                   order=pubs_present, palette=palette, ax=ax, inner="quartile")
    ax.set_title("Sentiment Score Distribution by Publication")
    ax.set_xlabel("")
    ax.set_ylabel("Sentiment Score")
    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    return _save_plot(fig, "sentiment_distribution")


def plot_ideology_distribution(df: pd.DataFrame) -> str:
    """Stacked bar: ideology label proportions per publication per year."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, pub in enumerate(TARGET_PUBLICATIONS):
        ax = axes[idx // 2][idx % 2]
        sub = df[df["publication"] == pub].copy()
        if len(sub) == 0:
            ax.set_title(f"{pub} (no data)")
            continue
        sub["year"] = sub["date"].dt.year
        ct = pd.crosstab(sub["year"], sub["ideology_score"], normalize="index")
        for col in ["left", "center", "right"]:
            if col not in ct.columns:
                ct[col] = 0.0
        ct = ct[["left", "center", "right"]]
        ct.plot.bar(stacked=True, ax=ax, color=["#2166AC", "#F7F7F7", "#B2182B"],
                     edgecolor="gray", linewidth=0.3)
        ax.set_title(pub, fontweight="bold")
        ax.set_ylabel("Proportion")
        ax.set_xlabel("")
        ax.legend(title="Ideology", fontsize=8)
        ax.set_ylim(0, 1)

    fig.suptitle("Ideology Label Distribution by Year", fontsize=15, y=1.02)
    fig.tight_layout()
    return _save_plot(fig, "ideology_distribution_by_year")


def plot_election_effect(df: pd.DataFrame) -> str:
    """Compare emotionality in election vs non-election years."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["is_election"] = df["year"].isin(ELECTION_YEARS)

    fig, ax = plt.subplots(figsize=(10, 6))
    pubs_present = [p for p in TARGET_PUBLICATIONS if p in df["publication"].values]
    
    means = df.groupby(["publication", "is_election"])["emotionality"].mean().unstack()
    means = means.reindex(pubs_present)
    
    x = np.arange(len(pubs_present))
    width = 0.35
    ax.bar(x - width/2, means[False], width, label="Non-Election Year",
           color="#4393C3", alpha=0.8)
    ax.bar(x + width/2, means[True], width, label="Election Year",
           color="#D6604D", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pubs_present, rotation=15)
    ax.set_ylabel("Mean Emotionality")
    ax.set_title("Headline Emotionality: Election vs Non-Election Years")
    ax.legend()
    fig.tight_layout()
    return _save_plot(fig, "election_effect")


def generate_all_plots(df: pd.DataFrame) -> list:
    """Run all visualizations, return list of saved paths."""
    print("\n══ Generating Visualizations ══")
    df["date"] = pd.to_datetime(df["date"])
    
    paths = []
    paths.append(plot_ideology_trends(df))
    paths.append(plot_sentiment_trends(df))
    paths.append(plot_emotionality_trends(df))
    paths.append(plot_ideology_by_topic(df))
    paths.append(plot_sentiment_distribution(df))
    paths.append(plot_ideology_distribution(df))
    paths.append(plot_election_effect(df))
    
    print(f"\n  Generated {len(paths)} plots total.")
    return paths

"""Cross-sectional momentum groups and forward-return statistics."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .mom_percentiles import GROUP_BINS, GROUP_LABELS, SAMPLE_ORDER


PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results" / "signals"
INPUT_PATH = RESULTS_DIR / "data" / "mom_momra_forward_returns.csv"
OUTPUT_DIR = RESULTS_DIR / "csmom_groups"


def cross_sectional_percentile(signal: pd.Series) -> pd.Series:
    """Rank one day's ETF signals from 0 (weakest) to 100 (strongest)."""
    valid = signal.dropna()
    result = pd.Series(np.nan, index=signal.index, dtype=float)
    if len(valid) == 1:
        result.loc[valid.index] = 50.0
    elif len(valid) > 1:
        ranks = valid.rank(method="average", ascending=True)
        result.loc[valid.index] = 100.0 * (ranks - 1.0) / (len(valid) - 1.0)
    return result


def prepare_csmom_panel(signal_panel: pd.DataFrame) -> pd.DataFrame:
    """Calculate same-day cross-sectional MOM ranks and assign quintile groups."""
    panel = signal_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel["csmom_percentile"] = (
        panel.groupby("date", group_keys=False)["mom"].transform(cross_sectional_percentile)
    )
    panel["csmom_group"] = pd.cut(
        panel["csmom_percentile"],
        bins=GROUP_BINS,
        labels=GROUP_LABELS,
        right=False,
    )
    return panel.dropna(subset=["csmom_percentile", "csmom_group"]).copy()


def calculate_group_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """Calculate future-return count, mean, and standard deviation per group."""
    stats = (
        panel.groupby(["sample", "csmom_group"], observed=False)["forward_5d_return"]
        .agg(observations="size", mean_return="mean", std_return="std")
        .reset_index()
    )
    complete_index = pd.MultiIndex.from_product(
        [SAMPLE_ORDER, GROUP_LABELS], names=["sample", "csmom_group"]
    )
    return stats.set_index(["sample", "csmom_group"]).reindex(complete_index).reset_index()


def plot_group_stats(stats: pd.DataFrame, metric: str, ylabel: str, output_path: str | Path) -> None:
    """Plot one future-return statistic across CSMOM groups and date samples."""
    fig, ax = plt.subplots(figsize=(12, 6.2))
    positions = np.arange(len(GROUP_LABELS))
    width = 0.23

    for i, sample in enumerate(SAMPLE_ORDER):
        sample_stats = (
            stats.loc[stats["sample"].eq(sample)]
            .set_index("csmom_group")
            .reindex(GROUP_LABELS)
        )
        x = positions + (i - 1) * width
        values = sample_stats[metric].to_numpy(dtype=float)
        counts = sample_stats["observations"].fillna(0).to_numpy(dtype=int)
        bars = ax.bar(x, values, width=width, label=sample.title())
        for bar, value, count in zip(bars, values, counts):
            if np.isnan(value):
                continue
            offset = 4 if value >= 0 else -12
            ax.annotate(
                f"n={count:,}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.axhline(0, color="0.25", linewidth=0.9)
    ax.set_xticks(positions, GROUP_LABELS)
    ax.set_xlabel("CSMOM percentile group (ranked across ETFs on the same day)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by cross-sectional momentum group")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_csmom_outputs(
    input_path: str | Path = INPUT_PATH,
    output_dir: str | Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Save CSMOM ranks, grouped statistics, and the required bar plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_panel = pd.read_csv(input_path)
    panel = prepare_csmom_panel(signal_panel)
    stats = calculate_group_stats(panel)

    panel.to_csv(output_dir / "csmom_grouped_panel.csv", index=False)
    stats.to_csv(output_dir / "csmom_group_return_stats.csv", index=False)
    plot_group_stats(
        stats,
        metric="mean_return",
        ylabel="Mean following 5-trading-day return",
        output_path=output_dir / "csmom_group_mean_returns.png",
    )
    plot_group_stats(
        stats,
        metric="std_return",
        ylabel="Standard deviation of following 5-trading-day return",
        output_path=output_dir / "csmom_group_return_std.png",
    )
    return stats


if __name__ == "__main__":
    print(make_csmom_outputs().to_string(index=False))

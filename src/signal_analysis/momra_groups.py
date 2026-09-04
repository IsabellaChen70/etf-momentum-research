"""Fixed MOMRA groups and their following five-day returns."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results" / "signals"
INPUT_PATH = RESULTS_DIR / "data" / "mom_momra_forward_returns.csv"
OUTPUT_DIR = RESULTS_DIR / "momra_fixed_groups"
SAMPLE_ORDER = ["training", "validation", "test"]
GROUP_LABELS = ["< −1.5", "[−1.5, −0.5)", "[−0.5, 0.5)", "[0.5, 1.5)", "≥ 1.5"]
GROUP_BINS = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf]


def calculate_momra_group_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """Assign the five required MOMRA groups and summarize future returns."""
    grouped_panel = panel.copy()
    grouped_panel["momra_group"] = pd.cut(
        grouped_panel["momra"],
        bins=GROUP_BINS,
        labels=GROUP_LABELS,
        right=False,
    )

    stats = (
        grouped_panel.groupby(["sample", "momra_group"], observed=False)["forward_5d_return"]
        .agg(observations="size", mean_return="mean", std_return="std")
        .reset_index()
    )
    complete_index = pd.MultiIndex.from_product(
        [SAMPLE_ORDER, GROUP_LABELS], names=["sample", "momra_group"]
    )
    return stats.set_index(["sample", "momra_group"]).reindex(complete_index).reset_index()


def plot_group_stats(stats: pd.DataFrame, metric: str, ylabel: str, output_path: str | Path) -> None:
    """Plot one requested return statistic for all five groups and samples."""
    fig, ax = plt.subplots(figsize=(12, 6.2))
    positions = np.arange(len(GROUP_LABELS))
    width = 0.23

    for i, sample in enumerate(SAMPLE_ORDER):
        sample_stats = stats.loc[stats["sample"].eq(sample)].set_index("momra_group").reindex(GROUP_LABELS)
        x = positions + (i - 1) * width
        values = sample_stats[metric].to_numpy(dtype=float)
        counts = sample_stats["observations"].fillna(0).to_numpy(dtype=int)
        bars = ax.bar(x, values, width=width, label=sample.title())

        for bar, value, count in zip(bars, values, counts):
            if np.isnan(value):
                ax.annotate(
                    "n=0", (bar.get_x() + bar.get_width() / 2, 0),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8,
                )
            else:
                offset = 4 if value >= 0 else -12
                ax.annotate(
                    f"n={count:,}", (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, offset), textcoords="offset points", ha="center", va="bottom", fontsize=8,
                )

    ax.axhline(0, color="0.25", linewidth=0.9)
    ax.set_xticks(positions, GROUP_LABELS)
    ax.set_xlabel("MOMRA group")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by fixed MOMRA group")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_momra_group_outputs(
    input_path: str | Path = INPUT_PATH,
    output_dir: str | Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Save the statistics table and its mean and volatility bar plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(input_path)
    stats = calculate_momra_group_stats(panel)
    stats.to_csv(output_dir / "momra_fixed_group_return_stats.csv", index=False)
    plot_group_stats(
        stats,
        metric="mean_return",
        ylabel="Mean following 5-trading-day return",
        output_path=output_dir / "momra_fixed_group_mean_returns.png",
    )
    plot_group_stats(
        stats,
        metric="std_return",
        ylabel="Standard deviation of following 5-trading-day return",
        output_path=output_dir / "momra_fixed_group_return_std.png",
    )
    return stats


if __name__ == "__main__":
    print(make_momra_group_outputs().to_string(index=False))

"""Scatterplots of momentum signals versus next-five-day returns."""

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
OUTPUT_DIR = RESULTS_DIR / "plots"
SAMPLE_ORDER = ["training", "validation", "test"]


def plot_signal_vs_forward_return(panel: pd.DataFrame, signal_column: str, output_path: str | Path) -> None:
    """Create one three-panel scatterplot, separated by date sample."""
    display_names = {
        "mom": "MOM (20-day average daily return)",
        "momra": "MOMRA z-score (MOM / standard error)",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)

    for ax, sample in zip(axes, SAMPLE_ORDER):
        data = panel.loc[panel["sample"].eq(sample), [signal_column, "forward_5d_return"]].dropna()
        x = data[signal_column].to_numpy()
        y = data["forward_5d_return"].to_numpy()
        correlation = np.corrcoef(x, y)[0, 1]

        ax.scatter(x, y, s=5, alpha=0.12, color="#1f77b4", edgecolors="none")
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, slope * x_line + intercept, color="#d55e00", linewidth=2, label="Linear fit")
        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.axvline(0, color="0.35", linewidth=0.8)
        ax.set_title(f"{sample.title()}\nN = {len(data):,}; correlation = {correlation:.3f}")
        ax.set_xlabel(display_names[signal_column])
        ax.grid(alpha=0.22)

    axes[0].set_ylabel("Following 5-trading-day return")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    if signal_column == "mom":
        for ax in axes:
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle(f"{display_names[signal_column]} versus following 5-day ETF return", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_scatterplots(
    input_path: str | Path = INPUT_PATH,
    output_dir: str | Path = OUTPUT_DIR,
) -> None:
    """Load the prepared signal data and save two diagnostic scatterplots."""
    panel = pd.read_csv(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_signal_vs_forward_return(panel, "mom", output_dir / "mom_vs_forward_5d_return.png")
    plot_signal_vs_forward_return(panel, "momra", output_dir / "momra_vs_forward_5d_return.png")


if __name__ == "__main__":
    make_scatterplots()

"""Rolling historical percentile groups for MOMRA."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .data_split import load_close_prices
from .mom_percentiles import (
    GROUP_BINS,
    GROUP_LABELS,
    LOOKBACK_DAYS,
    PERCENTILE_WINDOW,
    SAMPLE_ORDER,
    rolling_historical_percentile,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results" / "signals"
INPUT_PATH = RESULTS_DIR / "data" / "mom_momra_forward_returns.csv"
OUTPUT_DIR = RESULTS_DIR / "momra_percentile_groups"


def calculate_momra_percentiles(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate MOMRA and rank it against each ETF's prior 252 readings."""
    daily_returns = prices.pct_change(fill_method=None)
    mom = daily_returns.rolling(LOOKBACK_DAYS, min_periods=LOOKBACK_DAYS).mean()
    volatility = daily_returns.rolling(LOOKBACK_DAYS, min_periods=LOOKBACK_DAYS).std(ddof=1)
    momentum_risk = volatility.div(np.sqrt(LOOKBACK_DAYS))
    momra = mom.div(momentum_risk.replace(0.0, np.nan))

    percentiles = pd.DataFrame(index=momra.index, columns=momra.columns, dtype=float)
    for symbol in momra.columns:
        percentiles[symbol] = rolling_historical_percentile(momra[symbol], PERCENTILE_WINDOW)
    return percentiles


def prepare_percentile_panel(
    signal_panel: pd.DataFrame,
    momra_percentiles: pd.DataFrame,
) -> pd.DataFrame:
    """Merge clean outcomes with rolling MOMRA percentiles and assign groups."""
    percentile_long = momra_percentiles.stack().rename("momra_percentile").reset_index()
    percentile_long.columns = ["date", "symbol", "momra_percentile"]

    panel = signal_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.merge(percentile_long, on=["date", "symbol"], how="left", validate="one_to_one")
    panel = panel.dropna(subset=["momra_percentile"]).copy()
    panel["momra_percentile_group"] = pd.cut(
        panel["momra_percentile"],
        bins=GROUP_BINS,
        labels=GROUP_LABELS,
        right=False,
    )
    return panel


def calculate_group_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """Calculate count, mean, and standard deviation of future returns by group."""
    stats = (
        panel.groupby(["sample", "momra_percentile_group"], observed=False)["forward_5d_return"]
        .agg(observations="size", mean_return="mean", std_return="std")
        .reset_index()
    )
    complete_index = pd.MultiIndex.from_product(
        [SAMPLE_ORDER, GROUP_LABELS], names=["sample", "momra_percentile_group"]
    )
    return stats.set_index(["sample", "momra_percentile_group"]).reindex(complete_index).reset_index()


def plot_group_stats(stats: pd.DataFrame, metric: str, ylabel: str, output_path: str | Path) -> None:
    """Plot one future-return statistic across MOMRA percentile groups."""
    fig, ax = plt.subplots(figsize=(12, 6.2))
    positions = np.arange(len(GROUP_LABELS))
    width = 0.23

    for i, sample in enumerate(SAMPLE_ORDER):
        sample_stats = (
            stats.loc[stats["sample"].eq(sample)]
            .set_index("momra_percentile_group")
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
    ax.set_xlabel("MOMRA percentile group (ranked against prior 252 trading days)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by rolling MOMRA percentile group")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_momra_percentile_outputs(
    input_path: str | Path = INPUT_PATH,
    output_dir: str | Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Save the ranked panel, summary statistics, and required bar plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_panel = pd.read_csv(input_path)
    prices = load_close_prices()
    momra_percentiles = calculate_momra_percentiles(prices)
    panel = prepare_percentile_panel(signal_panel, momra_percentiles)
    stats = calculate_group_stats(panel)

    panel.to_csv(output_dir / "momra_rolling_percentile_panel.csv", index=False)
    stats.to_csv(output_dir / "momra_percentile_group_return_stats.csv", index=False)
    plot_group_stats(
        stats,
        metric="mean_return",
        ylabel="Mean following 5-trading-day return",
        output_path=output_dir / "momra_percentile_group_mean_returns.png",
    )
    plot_group_stats(
        stats,
        metric="std_return",
        ylabel="Standard deviation of following 5-trading-day return",
        output_path=output_dir / "momra_percentile_group_return_std.png",
    )
    return stats


if __name__ == "__main__":
    print(make_momra_percentile_outputs().to_string(index=False))

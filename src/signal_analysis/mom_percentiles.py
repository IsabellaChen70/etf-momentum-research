"""Rolling historical percentile groups for MOM."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .data_split import load_close_prices


PROJECT_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_DIR / "results" / "signals"
INPUT_PATH = RESULTS_DIR / "data" / "mom_momra_forward_returns.csv"
OUTPUT_DIR = RESULTS_DIR / "mom_percentile_groups"
LOOKBACK_DAYS = 20
PERCENTILE_WINDOW = 252
SAMPLE_ORDER = ["training", "validation", "test"]
GROUP_LABELS = ["0–20", "20–40", "40–60", "60–80", "80–100"]
GROUP_BINS = [-np.finfo(float).eps, 20.0, 40.0, 60.0, 80.0, np.nextafter(100.0, np.inf)]


def rolling_historical_percentile(series: pd.Series, window: int = PERCENTILE_WINDOW) -> pd.Series:
    """Rank today's MOM from 0 to 100 against only the prior ``window`` values."""
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)

    for i in range(window, len(values)):
        current = values[i]
        history = values[i - window : i]
        if np.isnan(current) or np.isnan(history).any():
            continue
        below = np.count_nonzero(history < current)
        equal = np.count_nonzero(history == current)
        result[i] = 100.0 * (below + 0.5 * equal) / window

    return pd.Series(result, index=series.index, name=series.name)


def calculate_mom_percentiles(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate 20-day MOM and its 252-day historical percentile per ETF."""
    daily_returns = prices.pct_change(fill_method=None)
    mom = daily_returns.rolling(LOOKBACK_DAYS, min_periods=LOOKBACK_DAYS).mean()
    percentiles = pd.DataFrame(index=mom.index, columns=mom.columns, dtype=float)
    for symbol in mom.columns:
        percentiles[symbol] = rolling_historical_percentile(mom[symbol])
    return percentiles


def prepare_percentile_panel(
    signal_panel: pd.DataFrame,
    mom_percentiles: pd.DataFrame,
) -> pd.DataFrame:
    """Merge clean five-day outcomes with MOM percentiles and assign groups."""
    percentile_long = mom_percentiles.stack().rename("mom_percentile").reset_index()
    percentile_long.columns = ["date", "symbol", "mom_percentile"]

    panel = signal_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.merge(percentile_long, on=["date", "symbol"], how="left", validate="one_to_one")
    panel = panel.dropna(subset=["mom_percentile"]).copy()
    panel["mom_percentile_group"] = pd.cut(
        panel["mom_percentile"],
        bins=GROUP_BINS,
        labels=GROUP_LABELS,
        right=False,
    )
    return panel


def calculate_group_stats(panel: pd.DataFrame) -> pd.DataFrame:
    """Calculate count, mean, and standard deviation of future returns by group."""
    stats = (
        panel.groupby(["sample", "mom_percentile_group"], observed=False)["forward_5d_return"]
        .agg(observations="size", mean_return="mean", std_return="std")
        .reset_index()
    )
    complete_index = pd.MultiIndex.from_product(
        [SAMPLE_ORDER, GROUP_LABELS], names=["sample", "mom_percentile_group"]
    )
    return stats.set_index(["sample", "mom_percentile_group"]).reindex(complete_index).reset_index()


def plot_group_stats(stats: pd.DataFrame, metric: str, ylabel: str, output_path: str | Path) -> None:
    """Plot one future-return statistic across percentile groups and samples."""
    fig, ax = plt.subplots(figsize=(12, 6.2))
    positions = np.arange(len(GROUP_LABELS))
    width = 0.23

    for i, sample in enumerate(SAMPLE_ORDER):
        sample_stats = (
            stats.loc[stats["sample"].eq(sample)]
            .set_index("mom_percentile_group")
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
    ax.set_xlabel("MOM percentile group (ranked against prior 252 trading days)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by rolling MOM percentile group")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_mom_percentile_outputs(
    input_path: str | Path = INPUT_PATH,
    output_dir: str | Path = OUTPUT_DIR,
) -> pd.DataFrame:
    """Save the ranked panel, summary statistics, and required bar plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_panel = pd.read_csv(input_path)
    prices = load_close_prices()
    mom_percentiles = calculate_mom_percentiles(prices)
    panel = prepare_percentile_panel(signal_panel, mom_percentiles)
    stats = calculate_group_stats(panel)

    panel.to_csv(output_dir / "mom_rolling_percentile_panel.csv", index=False)
    stats.to_csv(output_dir / "mom_percentile_group_return_stats.csv", index=False)
    plot_group_stats(
        stats,
        metric="mean_return",
        ylabel="Mean following 5-trading-day return",
        output_path=output_dir / "mom_percentile_group_mean_returns.png",
    )
    plot_group_stats(
        stats,
        metric="std_return",
        ylabel="Standard deviation of following 5-trading-day return",
        output_path=output_dir / "mom_percentile_group_return_std.png",
    )
    return stats


if __name__ == "__main__":
    print(make_mom_percentile_outputs().to_string(index=False))

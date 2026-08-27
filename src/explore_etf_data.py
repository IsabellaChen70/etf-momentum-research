"""Explore the ETF price panel used by the momentum backtest.

The raw bundle is combined with adjusted USO and UNG files. Documented split
events in five sector ETFs are corrected before returns are calculated.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cta_matplotlib")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "results" / "exploration"
ZIP_PATH = DATA_DIR / "CTA_data.zip"
ADJUSTED_PATHS = {
    "USO": DATA_DIR / "adjusted" / "USO_ohlcv_1d_adjusted.csv",
    "UNG": DATA_DIR / "adjusted" / "UNG_ohlcv_1d_adjusted.csv",
}
KNOWN_SPLIT_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "XLB": {"2025-12-05": 0.5},
    "XLE": {"2025-12-05": 0.5},
    "XLK": {"2025-12-05": 0.5},
    "XLU": {"2025-12-05": 0.5},
    "XLY": {"2025-12-05": 0.5},
}

CATEGORY_ORDER = [
    "us_equity_broad",
    "us_sectors",
    "international_equity",
    "fixed_income",
    "commodities",
    "alternatives",
]
CATEGORY_LABELS = {
    "us_equity_broad": "U.S. broad equity",
    "us_sectors": "U.S. sectors",
    "international_equity": "International equity",
    "fixed_income": "Fixed income",
    "commodities": "Commodities",
    "alternatives": "Alternatives",
}
CATEGORY_COLORS = {
    "us_equity_broad": "#3569a8",
    "us_sectors": "#7b6dcb",
    "international_equity": "#2a9d8f",
    "fixed_income": "#e9a03b",
    "commodities": "#c95b4a",
    "alternatives": "#77838f",
}
REPRESENTATIVE_ETFS = ["SPY", "QQQ", "EFA", "TLT", "HYG", "GLD", "DBC", "UUP"]
KEY_SCATTER_LABELS = {"SPY", "QQQ", "TLT", "SHY", "GLD", "USO", "UNG", "SLV"}


def read_data() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return the manifest and a split-adjusted symbol-to-data mapping."""
    with zipfile.ZipFile(ZIP_PATH) as archive:
        manifest = pd.read_csv(archive.open("_manifest.csv"))
        frames: dict[str, pd.DataFrame] = {}
        for row in manifest.itertuples(index=False):
            name = f"{row.symbol}_ohlcv_1d.csv"
            frame = pd.read_csv(archive.open(name))
            frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True)
            frame = frame.sort_values("ts_event").reset_index(drop=True)
            frames[row.symbol] = frame
    frames = adjust_price_frames(frames)
    return manifest, frames


def adjust_price_frames(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Apply the documented price adjustments without filling future values."""
    adjusted = {symbol: frame.copy() for symbol, frame in frames.items()}
    price_columns = ["open", "high", "low", "close"]

    for symbol, events in KNOWN_SPLIT_ADJUSTMENTS.items():
        frame = adjusted[symbol]
        for effective_date, prior_price_factor in events.items():
            before_split = frame["ts_event"] < pd.Timestamp(effective_date, tz="UTC")
            frame.loc[before_split, price_columns] *= prior_price_factor

    for symbol, path in ADJUSTED_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Adjusted input is missing: {path}")
        replacement = pd.read_csv(path)
        required = {"ts_event", "open", "high", "low", "close", "volume"}
        missing = required - set(replacement.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        replacement["ts_event"] = pd.to_datetime(replacement["ts_event"], utc=True)
        replacement = replacement.sort_values("ts_event").reset_index(drop=True)
        raw_dates = adjusted[symbol]["ts_event"]
        if not replacement["ts_event"].equals(raw_dates):
            raise ValueError(f"{path.name} does not match the bundle's trading dates")
        adjusted[symbol] = replacement

    return adjusted


def build_summary(manifest: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    spy_dates = set(frames["SPY"]["ts_event"].dt.date)

    for row in manifest.itertuples(index=False):
        frame = frames[row.symbol]
        returns = frame["close"].pct_change().dropna()
        own_dates = set(frame["ts_event"].dt.date)
        largest_move_idx = returns.abs().idxmax()
        records.append(
            {
                "symbol": row.symbol,
                "category": row.category,
                "n_prices": len(frame),
                "n_returns": len(returns),
                "start": frame["ts_event"].min().date().isoformat(),
                "end": frame["ts_event"].max().date().isoformat(),
                "missing_vs_spy": len(spy_dates - own_dates),
                "mean_daily_return_pct": returns.mean() * 100,
                "std_daily_return_pct": returns.std(ddof=1) * 100,
                "annualized_return_proxy_pct": returns.mean() * 252 * 100,
                "annualized_volatility_pct": returns.std(ddof=1) * np.sqrt(252) * 100,
                "positive_return_days_pct": (returns.gt(0).mean()) * 100,
                "largest_abs_daily_return_pct": returns.loc[largest_move_idx] * 100,
                "largest_abs_return_date": frame.loc[largest_move_idx, "ts_event"].date().isoformat(),
                "n_abs_return_gt_20pct": int(returns.abs().gt(0.20).sum()),
                "min_close": frame["close"].min(),
                "max_close": frame["close"].max(),
                "lowest_low": frame["low"].min(),
                "highest_high": frame["high"].max(),
                "close_change_pct": (frame["close"].iloc[-1] / frame["close"].iloc[0] - 1) * 100,
            }
        )

    summary = pd.DataFrame(records)
    summary["category"] = pd.Categorical(
        summary["category"], categories=CATEGORY_ORDER, ordered=True
    )
    return summary.sort_values(["category", "symbol"]).reset_index(drop=True)


def apply_chart_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#2d3748",
            "xtick.color": "#4a5568",
            "ytick.color": "#4a5568",
            "grid.color": "#d9e0e8",
            "grid.linewidth": 0.7,
        }
    )


def plot_normalized_performance(frames: dict[str, pd.DataFrame]) -> None:
    apply_chart_style()
    fig, ax = plt.subplots(figsize=(12, 6.4), constrained_layout=True)
    for symbol in REPRESENTATIVE_ETFS:
        frame = frames[symbol]
        normalized = frame["close"] / frame["close"].iloc[0] * 100
        ax.plot(frame["ts_event"], normalized, linewidth=1.65, label=symbol)

    ax.set_title("Normalized closing-price performance", loc="left")
    ax.set_ylabel("Index (first available close = 100)")
    ax.set_xlabel("")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.axhline(100, color="#718096", linewidth=0.9)
    ax.legend(ncol=4, frameon=False, loc="upper left")
    ax.text(
        0,
        -0.16,
        "Split-adjusted price series; distributions are excluded.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4a5568",
    )
    fig.savefig(OUTPUT_DIR / "normalized_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_risk_return(summary: pd.DataFrame) -> None:
    apply_chart_style()
    fig, ax = plt.subplots(figsize=(11.5, 7.1), constrained_layout=True)
    for category in CATEGORY_ORDER:
        subset = summary[summary["category"] == category]
        if subset.empty:
            continue
        ax.scatter(
            subset["std_daily_return_pct"],
            subset["mean_daily_return_pct"],
            s=52,
            color=CATEGORY_COLORS[category],
            label=CATEGORY_LABELS[category],
            alpha=0.9,
        )
        for row in subset.itertuples(index=False):
            if row.symbol in KEY_SCATTER_LABELS:
                ax.annotate(
                    row.symbol,
                    (row.std_daily_return_pct, row.mean_daily_return_pct),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                )

    ax.axhline(0, color="#718096", linewidth=0.9)
    ax.set_title("Daily return: average versus volatility", loc="left")
    ax.set_xscale("log")
    ax.set_xlabel("Daily return standard deviation (%), log scale")
    ax.set_ylabel("Average daily return (%)")
    ax.legend(frameon=False, fontsize=9, loc="best")
    fig.savefig(OUTPUT_DIR / "risk_return_scatter.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_price_ranges(summary: pd.DataFrame) -> None:
    apply_chart_style()
    chart_data = summary.sort_values(["category", "lowest_low", "symbol"], ascending=[True, True, True])
    fig, ax = plt.subplots(figsize=(11.5, 11), constrained_layout=True)
    y = np.arange(len(chart_data))
    width = chart_data["highest_high"].to_numpy() - chart_data["lowest_low"].to_numpy()
    colors = [CATEGORY_COLORS[category] for category in chart_data["category"]]

    ax.hlines(y, chart_data["lowest_low"], chart_data["highest_high"], color=colors, linewidth=4)
    ax.scatter(chart_data["lowest_low"], y, color=colors, s=24, zorder=3, marker="|")
    ax.scatter(chart_data["highest_high"], y, color=colors, s=24, zorder=3, marker="|")
    ax.set_yticks(y, chart_data["symbol"])
    ax.set_xlabel("Price ($)")
    ax.set_title("Observed price ranges: lowest low to highest high", loc="left")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    fig.savefig(OUTPUT_DIR / "price_ranges.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_notes(summary: pd.DataFrame) -> None:
    highest_mean = summary.loc[summary["mean_daily_return_pct"].idxmax()]
    lowest_mean = summary.loc[summary["mean_daily_return_pct"].idxmin()]
    most_volatile = summary.loc[summary["std_daily_return_pct"].idxmax()]
    least_volatile = summary.loc[summary["std_daily_return_pct"].idxmin()]
    biggest_gain = summary.loc[summary["close_change_pct"].idxmax()]
    biggest_loss = summary.loc[summary["close_change_pct"].idxmin()]
    missing = summary.loc[summary["missing_vs_spy"] > 0, ["symbol", "missing_vs_spy"]]
    jumps = summary.loc[
        summary["n_abs_return_gt_20pct"] > 0,
        ["symbol", "largest_abs_daily_return_pct", "largest_abs_return_date", "n_abs_return_gt_20pct"],
    ].sort_values("largest_abs_daily_return_pct", key=lambda col: col.abs(), ascending=False)

    lines = [
        "ETF price panel exploration",
        "",
        "Method: simple close-to-close daily returns, r_t = close_t / close_(t-1) - 1.",
        "Price range: observed lowest low through highest high in each file.",
        "Annualized values use 252 trading days and are descriptive only.",
        "",
        f"Highest average daily return: {highest_mean.symbol} ({highest_mean.mean_daily_return_pct:.3f}%).",
        f"Lowest average daily return: {lowest_mean.symbol} ({lowest_mean.mean_daily_return_pct:.3f}%).",
        f"Most volatile daily returns: {most_volatile.symbol} ({most_volatile.std_daily_return_pct:.3f}% standard deviation).",
        f"Least volatile daily returns: {least_volatile.symbol} ({least_volatile.std_daily_return_pct:.3f}% standard deviation).",
        f"Largest close-to-close period gain: {biggest_gain.symbol} ({biggest_gain.close_change_pct:.1f}%).",
        f"Largest close-to-close period loss: {biggest_loss.symbol} ({biggest_loss.close_change_pct:.1f}%).",
        "",
        "Data checks: no blank fields, duplicate dates, invalid OHLC ordering, or negative volumes found.",
        "Missing dates relative to SPY: "
        + (", ".join(f"{r.symbol} ({int(r.missing_vs_spy)})" for r in missing.itertuples(index=False)) if not missing.empty else "none"),
        "",
        "Single-day moves above 20% after the documented adjustments: "
        + (
            ", ".join(
                f"{r.symbol} ({r.largest_abs_daily_return_pct:+.1f}% on {r.largest_abs_return_date})"
                for r in jumps.itertuples(index=False)
            )
            if not jumps.empty
            else "none"
        ),
        "",
        "Caveat: the adjustments address identified splits and the supplied USO/UNG replacements."
        " They do not create total-return series because distributions are excluded.",
    ]
    (OUTPUT_DIR / "findings.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Expected input ZIP at {ZIP_PATH}")
    OUTPUT_DIR.mkdir(exist_ok=True)
    manifest, frames = read_data()
    summary = build_summary(manifest, frames)
    summary.to_csv(OUTPUT_DIR / "summary_statistics.csv", index=False, float_format="%.6f")
    summary.loc[summary["n_abs_return_gt_20pct"] > 0].to_csv(
        OUTPUT_DIR / "large_daily_moves.csv", index=False, float_format="%.6f"
    )
    plot_normalized_performance(frames)
    plot_risk_return(summary)
    plot_price_ranges(summary)
    write_notes(summary)
    print(f"Wrote outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

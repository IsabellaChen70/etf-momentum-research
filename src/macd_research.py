"""Signal-proportional MOMRA and MACD portfolio research.

This module reuses the split-adjusted 37-ETF price loader and the fixed-share,
five-day overlapping-cohort backtester from the earlier research stages. All model
selection uses validation Sharpe; test results are calculated only after a
parameter set has been selected.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_DIR / "results" / "macd"
TABLE_DIR = RESULT_DIR / "tables"
PLOT_DIR = RESULT_DIR / "plots"

from src.signal_analysis.data_split import (
    TEST_START,
    TRAIN_START,
    VALIDATION_START,
    load_close_prices,
)
from src.signal_analysis.signals import assign_clean_samples
from src.momentum_backtest import multi_asset_backtest


HOLDING_DAYS = 5
EXECUTION_LAG_DAYS = 1
MOMRA_LOOKBACK = 20
LONG_GROSS = 1.50
SHORT_GROSS = 0.50
PERIODS_PER_YEAR = 252

SAMPLE_ORDER = ["training", "validation", "test"]
SAMPLE_BOUNDS = {
    "training": (TRAIN_START, VALIDATION_START),
    "validation": (VALIDATION_START, TEST_START),
    "test": (TEST_START, None),
}

SMA_PAIRS = [
    (1, 5),
    (1, 10),
    (5, 10),
    (5, 20),
    (10, 20),
    (10, 60),
    (20, 60),
]
EWMA_ALPHAS = [1 / 2, 1 / 3, 1 / 5, 1 / 8, 1 / 10]

QUINTILE_LABELS = ["0–20", "20–40", "40–60", "60–80", "80–100"]
QUINTILE_BINS = [
    -np.finfo(float).eps,
    20.0,
    40.0,
    60.0,
    80.0,
    np.nextafter(100.0, np.inf),
]


def calculate_momra(prices: pd.DataFrame, lookback: int = MOMRA_LOOKBACK) -> pd.DataFrame:
    """Calculate the z-score MOMRA signal from close-to-close returns."""
    daily_return = prices.pct_change(fill_method=None)
    mom = daily_return.rolling(lookback, min_periods=lookback).mean()
    volatility = daily_return.rolling(lookback, min_periods=lookback).std(ddof=1)
    standard_error = volatility.div(np.sqrt(lookback))
    return mom.div(standard_error.replace(0.0, np.nan))


def calculate_sma_macd(prices: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """Return normalized MACD, ``SMA_fast / SMA_slow - 1``."""
    if fast >= slow or fast < 1:
        raise ValueError("MACD requires 1 <= fast < slow")
    fast_average = prices.rolling(fast, min_periods=fast).mean()
    slow_average = prices.rolling(slow, min_periods=slow).mean()
    return fast_average.div(slow_average).sub(1.0)


def calculate_ewma_macd(
    prices: pd.DataFrame,
    alpha_fast: float,
    alpha_slow: float,
) -> pd.DataFrame:
    """Return normalized MACD from recursive fast and slow EWMAs."""
    if not 0 < alpha_slow < alpha_fast <= 1:
        raise ValueError("EWMA MACD requires 0 < alpha_slow < alpha_fast <= 1")
    fast_average = prices.ewm(alpha=alpha_fast, adjust=False, min_periods=1).mean()
    slow_average = prices.ewm(alpha=alpha_slow, adjust=False, min_periods=1).mean()
    return fast_average.div(slow_average).sub(1.0)


def smooth_signal(signal: pd.DataFrame, smoother_days: int) -> pd.DataFrame:
    """Smooth an asset-level signal with an equal-weight rolling mean."""
    if smoother_days < 1:
        raise ValueError("smoother_days must be at least one")
    if smoother_days == 1:
        return signal.copy()
    return signal.rolling(smoother_days, min_periods=smoother_days).mean()


def proportional_150_50_weights(signal: pd.Series) -> pd.Series:
    """Demean one signal cross-section and scale it to exact 150/50 exposure."""
    weights = pd.Series(0.0, index=signal.index, dtype=float)
    clean = signal.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return weights

    centered = clean - clean.mean()
    positive = centered > 0
    negative = centered < 0
    positive_total = centered.loc[positive].sum()
    negative_total = -centered.loc[negative].sum()
    if positive_total <= 0 or negative_total <= 0:
        return weights

    weights.loc[centered.index[positive]] = (
        LONG_GROSS * centered.loc[positive] / positive_total
    )
    weights.loc[centered.index[negative]] = (
        SHORT_GROSS * centered.loc[negative] / negative_total
    )
    return weights


def signal_to_targets(signal: pd.DataFrame) -> pd.DataFrame:
    """Convert every available daily signal cross-section into portfolio targets."""
    targets = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for date, row in signal.iterrows():
        targets.loc[date] = proportional_150_50_weights(row)
    return targets


def sample_dates(index: pd.DatetimeIndex, sample: str) -> pd.DatetimeIndex:
    """Return the dates belonging to one non-overlapping evaluation sample."""
    start, end = SAMPLE_BOUNDS[sample]
    mask = index >= start
    if end is not None:
        mask &= index < end
    return index[mask]


def backtest_one_sample(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    sample: str,
) -> pd.DataFrame:
    """Backtest one sample independently, retaining one prior signal date."""
    dates = sample_dates(prices.index, sample)
    if dates.empty:
        raise ValueError(f"No price dates for {sample}")

    first_location = prices.index.get_loc(dates[0])
    warmup_location = max(0, first_location - 1)
    subset_index = prices.index[warmup_location : prices.index.get_loc(dates[-1]) + 1]
    result = multi_asset_backtest(
        prices.loc[subset_index],
        targets.loc[subset_index],
        holding_days=HOLDING_DAYS,
    )
    return result.loc[dates].copy()


def backtest_all_samples(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    samples: tuple[str, ...] = tuple(SAMPLE_ORDER),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Create targets once and run the requested independent sample backtests."""
    signal = signal.reindex(index=prices.index, columns=prices.columns)
    targets = signal_to_targets(signal)
    results = {
        sample: backtest_one_sample(prices, targets, sample)
        for sample in samples
    }
    return targets, results


def performance_metrics(result: pd.DataFrame) -> dict[str, float | str | int]:
    """Calculate conventional daily-return portfolio statistics."""
    returns = result["portfolio_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n_return_days = max(len(returns) - 1, 1)
    start_value = float(result["portfolio_value"].iloc[0])
    ending_value = float(result["portfolio_value"].iloc[-1])
    annualized_return = (ending_value / start_value) ** (PERIODS_PER_YEAR / n_return_days) - 1
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
    daily_std = float(returns.std(ddof=1))
    sharpe = float(returns.mean() / daily_std * np.sqrt(PERIODS_PER_YEAR)) if daily_std else np.nan
    running_peak = result["portfolio_value"].cummax()
    max_drawdown = float((result["portfolio_value"] / running_peak - 1.0).min())

    return {
        "start": result.index.min().date().isoformat(),
        "end": result.index.max().date().isoformat(),
        "trading_days": len(result),
        "ending_value": ending_value,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "average_gross_exposure": float(result["gross_exposure"].mean()),
        "average_daily_turnover": float(result["turnover"].mean()),
    }


def summarize_strategy(
    results: dict[str, pd.DataFrame],
    **parameters: object,
) -> list[dict[str, object]]:
    """Return one performance row per sample for a strategy specification."""
    rows: list[dict[str, object]] = []
    for sample in SAMPLE_ORDER:
        if sample not in results:
            continue
        rows.append({**parameters, "sample": sample, **performance_metrics(results[sample])})
    return rows


def validation_ranking(table: pd.DataFrame) -> pd.DataFrame:
    """Add a validation-Sharpe rank without consulting test performance."""
    ranked = table.copy()
    validation = ranked.loc[ranked["sample"].eq("validation")].copy()
    validation["validation_rank"] = validation["sharpe_ratio"].rank(
        ascending=False,
        method="min",
    )
    parameter_columns = [
        column
        for column in ranked.columns
        if column
        not in {
            "sample",
            "start",
            "end",
            "trading_days",
            "ending_value",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "average_gross_exposure",
            "average_daily_turnover",
            "validation_rank",
        }
    ]
    rank_lookup = validation[parameter_columns + ["validation_rank"]]
    return ranked.merge(rank_lookup, on=parameter_columns, how="left", validate="many_to_one")


def best_parameter_rows(table: pd.DataFrame) -> pd.DataFrame:
    """Return all parameter rows tied for the highest validation Sharpe."""
    validation = table.loc[table["sample"].eq("validation")].copy()
    best_value = validation["sharpe_ratio"].max()
    return validation.loc[np.isclose(validation["sharpe_ratio"], best_value, rtol=0, atol=1e-12)]


def stitch_equity(results: dict[str, pd.DataFrame]) -> pd.Series:
    """Join independently tested samples into one curve, resetting positions at boundaries."""
    returns = pd.concat(
        [results[sample]["portfolio_return"] for sample in SAMPLE_ORDER]
    ).sort_index()
    equity = (1.0 + returns.fillna(0.0)).cumprod()
    equity.name = "portfolio_value"
    return equity


def plot_equity_curves(
    curves: dict[str, pd.Series],
    title: str,
    output_path: Path,
) -> None:
    """Plot one or more portfolio-level cumulative return curves."""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for label, curve in curves.items():
        ax.plot(curve.index, curve, linewidth=1.65, label=label)
    for boundary, label in [(VALIDATION_START, "Validation"), (TEST_START, "Test")]:
        ax.axvline(boundary, color="0.45", linewidth=0.9, linestyle="--")
        ax.annotate(label, (boundary, 1.0), xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.axhline(1.0, color="0.35", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value (starting value = 1)")
    ax.grid(alpha=0.25)
    if len(curves) > 1:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def cross_sectional_percentile(signal: pd.Series) -> pd.Series:
    """Rank one date's available ETF signals from 0 to 100."""
    clean = signal.dropna()
    result = pd.Series(np.nan, index=signal.index, dtype=float)
    if len(clean) == 1:
        result.loc[clean.index] = 50.0
    elif len(clean) > 1:
        ranks = clean.rank(method="average", ascending=True)
        result.loc[clean.index] = 100.0 * (ranks - 1.0) / (len(clean) - 1.0)
    return result


def calculate_macd_quintile_stats(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group selected MACD cross-sectionally and summarize clean forward returns."""
    exit_offset = EXECUTION_LAG_DAYS + HOLDING_DAYS
    entry_price = prices.shift(-EXECUTION_LAG_DAYS)
    exit_price = prices.shift(-exit_offset)
    forward_return = exit_price.div(entry_price).sub(1.0)
    forward_end_dates = pd.Series(prices.index, index=prices.index).shift(-exit_offset)
    sample = assign_clean_samples(prices.index, forward_end_dates)

    percentile = signal.apply(cross_sectional_percentile, axis=1)
    panel = pd.concat(
        {
            "macd_signal": signal.stack(future_stack=True),
            "macd_percentile": percentile.stack(future_stack=True),
            "forward_5d_return": forward_return.stack(future_stack=True),
        },
        axis=1,
    ).reset_index()
    panel.columns = ["date", "symbol", "macd_signal", "macd_percentile", "forward_5d_return"]
    panel["sample"] = panel["date"].map(sample)
    panel = panel.dropna(
        subset=["macd_signal", "macd_percentile", "forward_5d_return", "sample"]
    ).copy()
    panel["macd_quintile"] = pd.cut(
        panel["macd_percentile"],
        bins=QUINTILE_BINS,
        labels=QUINTILE_LABELS,
        right=False,
    )

    stats = (
        panel.groupby(["sample", "macd_quintile"], observed=False)["forward_5d_return"]
        .agg(observations="size", mean_return="mean", std_return="std")
        .reset_index()
    )
    complete = pd.MultiIndex.from_product(
        [SAMPLE_ORDER, QUINTILE_LABELS], names=["sample", "macd_quintile"]
    )
    stats = stats.set_index(["sample", "macd_quintile"]).reindex(complete).reset_index()
    return panel, stats


def plot_quintile_stat(
    stats: pd.DataFrame,
    metric: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """Plot a grouped mean or standard deviation for the selected MACD signal."""
    fig, ax = plt.subplots(figsize=(12, 6.2))
    positions = np.arange(len(QUINTILE_LABELS))
    width = 0.23
    for i, sample in enumerate(SAMPLE_ORDER):
        sample_stats = (
            stats.loc[stats["sample"].eq(sample)]
            .set_index("macd_quintile")
            .reindex(QUINTILE_LABELS)
        )
        values = sample_stats[metric].to_numpy(dtype=float)
        counts = sample_stats["observations"].fillna(0).to_numpy(dtype=int)
        x = positions + (i - 1) * width
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
    ax.set_xticks(positions, QUINTILE_LABELS)
    ax.set_xlabel("MACD quintile (ranked across ETFs on the same day)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by selected MACD quintile")
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate_sma_grid(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[int, int], dict[str, pd.DataFrame]]]:
    """Backtest the seven prespecified SMA MACD pairs."""
    rows: list[dict[str, object]] = []
    result_map: dict[tuple[int, int], dict[str, pd.DataFrame]] = {}
    for fast, slow in SMA_PAIRS:
        signal = calculate_sma_macd(prices, fast, slow)
        _, results = backtest_all_samples(prices, signal, ("training", "validation"))
        result_map[(fast, slow)] = results
        rows.extend(summarize_strategy(results, fast_days=fast, slow_days=slow))
    return validation_ranking(pd.DataFrame(rows)), result_map


def evaluate_smoothers(
    prices: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    dict[tuple[int, int, int], dict[str, pd.DataFrame]],
    tuple[int, int, int],
]:
    """Jointly search every compatible MACD pair and smoother from 2 to fast-1."""
    rows: list[dict[str, object]] = []
    result_map: dict[tuple[int, int, int], dict[str, pd.DataFrame]] = {}
    for fast_days, slow_days in SMA_PAIRS:
        base_signal = calculate_sma_macd(prices, fast_days, slow_days)
        for days in range(2, fast_days):
            signal = smooth_signal(base_signal, days)
            _, results = backtest_all_samples(prices, signal, ("training", "validation"))
            key = (fast_days, slow_days, days)
            result_map[key] = results
            rows.extend(
                summarize_strategy(
                    results,
                    fast_days=fast_days,
                    slow_days=slow_days,
                    smoother_days=days,
                )
            )
    if not rows:
        raise ValueError("No MACD pair admits a smoother shorter than its fast window")
    table = validation_ranking(pd.DataFrame(rows))
    best = best_parameter_rows(table).iloc[0]
    selected = (
        int(best["fast_days"]),
        int(best["slow_days"]),
        int(best["smoother_days"]),
    )
    return table, result_map, selected


def ewma_pairs() -> list[tuple[float, float]]:
    """Return all critical alpha pairs with a faster fast average."""
    return [
        (alpha_fast, alpha_slow)
        for alpha_fast in EWMA_ALPHAS
        for alpha_slow in EWMA_ALPHAS
        if alpha_fast > alpha_slow
    ]


def evaluate_ewma_grid(
    prices: pd.DataFrame,
    smoother_days: int,
) -> tuple[pd.DataFrame, dict[tuple[float, float], dict[str, pd.DataFrame]]]:
    """Backtest the EWMA MACD alpha grid with the selected signal smoother."""
    rows: list[dict[str, object]] = []
    result_map: dict[tuple[float, float], dict[str, pd.DataFrame]] = {}
    for alpha_fast, alpha_slow in ewma_pairs():
        raw_signal = calculate_ewma_macd(prices, alpha_fast, alpha_slow)
        signal = smooth_signal(raw_signal, smoother_days)
        _, results = backtest_all_samples(prices, signal, ("training", "validation"))
        key = (alpha_fast, alpha_slow)
        result_map[key] = results
        rows.extend(
            summarize_strategy(
                results,
                alpha_fast=alpha_fast,
                alpha_slow=alpha_slow,
                smoother_days=smoother_days,
            )
        )
    return validation_ranking(pd.DataFrame(rows)), result_map


def audit_target_exposure(targets: pd.DataFrame, label: str) -> dict[str, object]:
    """Check exact target long/short totals on every active signal date."""
    long_total = targets.clip(lower=0).sum(axis=1)
    short_total = -targets.clip(upper=0).sum(axis=1)
    active = long_total.gt(0) | short_total.gt(0)
    valid = active & long_total.gt(0) & short_total.gt(0)
    if valid.any():
        if not np.allclose(long_total.loc[valid], LONG_GROSS, atol=1e-10):
            raise AssertionError(f"{label}: long targets do not sum to 150%")
        if not np.allclose(short_total.loc[valid], SHORT_GROSS, atol=1e-10):
            raise AssertionError(f"{label}: short targets do not sum to 50%")
    return {
        "check": f"{label}_target_exposure",
        "status": "pass",
        "active_dates": int(active.sum()),
        "max_long_error": float((long_total.loc[valid] - LONG_GROSS).abs().max()) if valid.any() else 0.0,
        "max_short_error": float((short_total.loc[valid] - SHORT_GROSS).abs().max()) if valid.any() else 0.0,
    }


def audit_results(
    prices: pd.DataFrame,
    target_sets: dict[str, pd.DataFrame],
    final_results: dict[str, dict[str, pd.DataFrame]],
    quintile_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Run structural and numerical checks for the research design."""
    checks = [audit_target_exposure(targets, label) for label, targets in target_sets.items()]

    max_abs_return = float(prices.pct_change(fill_method=None).abs().max().max())
    if max_abs_return >= 1.0:
        raise AssertionError("A split-like adjusted-price return of at least 100% remains")
    checks.append(
        {
            "check": "split_adjusted_price_returns",
            "status": "pass",
            "active_dates": len(prices),
            "max_long_error": max_abs_return,
            "max_short_error": np.nan,
        }
    )

    for strategy, sample_results in final_results.items():
        for sample, result in sample_results.items():
            numeric = result.select_dtypes(include=[np.number])
            if not np.isfinite(numeric.to_numpy()).all():
                raise AssertionError(f"{strategy}/{sample} contains non-finite backtest values")
            expected_dates = sample_dates(prices.index, sample)
            if not result.index.equals(expected_dates):
                raise AssertionError(f"{strategy}/{sample} has incorrect sample dates")
    checks.append(
        {
            "check": "finite_results_and_sample_boundaries",
            "status": "pass",
            "active_dates": sum(len(sample_dates(prices.index, s)) for s in SAMPLE_ORDER),
            "max_long_error": 0.0,
            "max_short_error": 0.0,
        }
    )

    synthetic_dates = pd.bdate_range("2020-01-01", periods=8)
    synthetic_prices = pd.DataFrame(
        {"asset": 100.0 + np.arange(len(synthetic_dates))},
        index=synthetic_dates,
    )
    synthetic_targets = pd.DataFrame(0.0, index=synthetic_dates, columns=["asset"])
    synthetic_targets.iloc[0, 0] = 1.0
    synthetic_result = multi_asset_backtest(
        synthetic_prices,
        synthetic_targets,
        holding_days=HOLDING_DAYS,
    )
    pnl_dates = synthetic_result.index[synthetic_result["portfolio_return"].ne(0.0)]
    held_dates = synthetic_result.index[synthetic_result["weight_asset"].ne(0.0)]
    if not pnl_dates.equals(synthetic_dates[2:7]):
        raise AssertionError("A signal does not earn exactly five returns after a one-day lag")
    if not held_dates.equals(synthetic_dates[1:6]):
        raise AssertionError("A cohort is not held for the expected five close dates")
    checks.append(
        {
            "check": "one_day_lag_and_five_return_day_holding",
            "status": "pass",
            "active_dates": len(pnl_dates),
            "max_long_error": 0.0,
            "max_short_error": 0.0,
        }
    )

    sample_end = quintile_panel.groupby("sample")["date"].max()
    if pd.Timestamp(sample_end["training"]) >= VALIDATION_START:
        raise AssertionError("Training forward outcomes cross into validation")
    if pd.Timestamp(sample_end["validation"]) >= TEST_START:
        raise AssertionError("Validation forward outcomes cross into test")
    checks.append(
        {
            "check": "clean_forward_return_boundaries",
            "status": "pass",
            "active_dates": quintile_panel["date"].nunique(),
            "max_long_error": 0.0,
            "max_short_error": 0.0,
        }
    )
    return pd.DataFrame(checks)


def run_macd_analysis() -> dict[str, object]:
    """Run the MACD research stage and save its tables and figures."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_close_prices()

    # Define the baseline now; defer its historical-test backtest until selection ends.
    baseline_signal = -calculate_momra(prices)

    # Search on training and validation only.
    sma_table, _ = evaluate_sma_grid(prices)
    best_sma_rows = best_parameter_rows(sma_table)
    best_sma = best_sma_rows.iloc[0]
    best_fast = int(best_sma["fast_days"])
    best_slow = int(best_sma["slow_days"])
    selected_sma_signal = calculate_sma_macd(prices, best_fast, best_slow)

    # Jointly search each compatible MACD pair and smoother length.
    smoother_table, _, selected_smoother_spec = evaluate_smoothers(prices)
    smooth_fast, smooth_slow, best_smoother = selected_smoother_spec
    selected_smoothed_signal = smooth_signal(
        calculate_sma_macd(prices, smooth_fast, smooth_slow),
        best_smoother,
    )

    # Replace fast and slow SMAs with EWMAs and search candidate alpha pairs.
    ewma_table, _ = evaluate_ewma_grid(prices, best_smoother)
    best_ewma_rows = best_parameter_rows(ewma_table)
    best_ewma = best_ewma_rows.iloc[0]
    best_alpha_fast = float(best_ewma["alpha_fast"])
    best_alpha_slow = float(best_ewma["alpha_slow"])
    selected_ewma_signal = smooth_signal(
        calculate_ewma_macd(prices, best_alpha_fast, best_alpha_slow),
        best_smoother,
    )
    # Freeze every selected specification before calculating historical-test results.
    quintile_panel, quintile_stats = calculate_macd_quintile_stats(
        prices, selected_sma_signal
    )
    baseline_targets, baseline_results = backtest_all_samples(prices, baseline_signal)
    selected_sma_targets, selected_sma_results = backtest_all_samples(
        prices, selected_sma_signal
    )
    selected_smoothed_targets, selected_smoothed_results = backtest_all_samples(
        prices, selected_smoothed_signal
    )
    selected_ewma_targets, selected_ewma_results = backtest_all_samples(
        prices, selected_ewma_signal
    )

    final_results = {
        "MOMRA reversal baseline": baseline_results,
        f"SMA MACD {best_fast}/{best_slow}": selected_sma_results,
        f"Smoothed SMA MACD {smooth_fast}/{smooth_slow} ({best_smoother}d)": selected_smoothed_results,
    }
    final_results[f"EWMA MACD ({best_alpha_fast:.3f}/{best_alpha_slow:.3f})"] = selected_ewma_results
    final_rows: list[dict[str, object]] = []
    for strategy, results in final_results.items():
        final_rows.extend(summarize_strategy(results, strategy=strategy))
    final_table = pd.DataFrame(final_rows)

    # Save tables and selected-signal research data.
    table_paths = {
        "sma": TABLE_DIR / "sma_macd_grid_search.csv",
        "smoother": TABLE_DIR / "macd_smoother_search.csv",
        "ewma": TABLE_DIR / "ewma_macd_grid_search.csv",
        "quintile": TABLE_DIR / "macd_quintile_return_stats.csv",
        "panel": TABLE_DIR / "macd_quintile_panel.csv",
        "final": TABLE_DIR / "final_strategy_performance.csv",
        "equity": TABLE_DIR / "final_portfolio_values.csv",
        "audit": TABLE_DIR / "validation_checks.csv",
    }
    sma_table.to_csv(table_paths["sma"], index=False)
    smoother_table.to_csv(table_paths["smoother"], index=False)
    ewma_table.to_csv(table_paths["ewma"], index=False)
    quintile_stats.to_csv(table_paths["quintile"], index=False)
    quintile_panel.to_csv(table_paths["panel"], index=False)
    final_table.to_csv(table_paths["final"], index=False)

    final_curves = {label: stitch_equity(results) for label, results in final_results.items()}
    pd.DataFrame(final_curves).to_csv(table_paths["equity"], index_label="date")

    # Portfolio and grouped-return plots.
    plot_equity_curves(
        {"MOMRA reversal baseline": final_curves["MOMRA reversal baseline"]},
        "Signal-Proportional MOMRA Reversal Portfolio",
        PLOT_DIR / "momra_reversal_baseline.png",
    )
    plot_equity_curves(
        {f"SMA MACD {best_fast}/{best_slow}": stitch_equity(selected_sma_results)},
        "Selected SMA MACD Portfolio",
        PLOT_DIR / "selected_sma_macd_portfolio.png",
    )
    smoother_curves = {
        f"SMA MACD {smooth_fast}/{smooth_slow}, {best_smoother}-day smoother": stitch_equity(
            selected_smoothed_results
        )
    }
    plot_equity_curves(
        smoother_curves,
        "Selected MACD Smoother Portfolio",
        PLOT_DIR / "macd_smoother_comparison.png",
    )
    plot_equity_curves(
        {
            f"EWMA MACD α={best_alpha_fast:.3f}/{best_alpha_slow:.3f}": stitch_equity(
                selected_ewma_results
            )
        },
        "Selected EWMA MACD Portfolio",
        PLOT_DIR / "selected_ewma_macd_portfolio.png",
    )
    plot_equity_curves(
        final_curves,
        "Final MACD Strategy Comparison",
        PLOT_DIR / "final_strategy_comparison.png",
    )
    plot_quintile_stat(
        quintile_stats,
        "mean_return",
        "Mean following 5-trading-day return",
        PLOT_DIR / "macd_quintile_mean_return.png",
    )
    plot_quintile_stat(
        quintile_stats,
        "std_return",
        "Standard deviation of following 5-trading-day return",
        PLOT_DIR / "macd_quintile_return_std.png",
    )

    target_sets = {
        "momra_baseline": baseline_targets,
        "selected_sma_macd": selected_sma_targets,
        "selected_smoothed_macd": selected_smoothed_targets,
        "selected_ewma_macd": selected_ewma_targets,
    }
    audit = audit_results(prices, target_sets, final_results, quintile_panel)
    audit.to_csv(table_paths["audit"], index=False)

    return {
        "prices": prices,
        "sma_grid": sma_table,
        "smoother_grid": smoother_table,
        "ewma_grid": ewma_table,
        "quintile_stats": quintile_stats,
        "final_performance": final_table,
        "audit": audit,
        "selected_sma": (best_fast, best_slow),
        "selected_smoother": best_smoother,
        "selected_smoothed_macd": (smooth_fast, smooth_slow),
        "selected_ewma": (best_alpha_fast, best_alpha_slow),
    }


if __name__ == "__main__":
    output = run_macd_analysis()
    print(f"Selected SMA MACD: {output['selected_sma']}")
    print(
        f"Selected smoother: {output['selected_smoother']} day(s) "
        f"on SMA MACD {output['selected_smoothed_macd']}"
    )
    print(f"Selected EWMA alphas: {output['selected_ewma']}")

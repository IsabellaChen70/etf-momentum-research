"""Forward-safe feature processing and execution-aligned relative returns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


HOLDING_DAYS = 5
EXECUTION_LAG_DAYS = 1
PREPROCESS_LOOKBACK = 126
PREPROCESS_MIN_PERIODS = 63
MOM_WINDOWS = (3, 5, 10, 20)
SMA_MACD_PAIRS = ((1, 5), (1, 10), (5, 10), (5, 20), (10, 20), (10, 60), (20, 60))


@dataclass(frozen=True)
class ProcessedFeature:
    raw: pd.DataFrame
    lower_bound: pd.DataFrame
    upper_bound: pd.DataFrame
    normalized: pd.DataFrame


def calculate_raw_features(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Calculate the prior project signals from information known at each close."""
    returns = prices.pct_change(fill_method=None)
    features: dict[str, pd.DataFrame] = {}
    for days in MOM_WINDOWS:
        features[f"mom_{days}"] = returns.rolling(days, min_periods=days).mean()

    mom20 = features["mom_20"]
    standard_error = returns.rolling(20, min_periods=20).std(ddof=1).div(np.sqrt(20))
    features["momra_20"] = mom20.div(standard_error.replace(0.0, np.nan))

    rank = mom20.rank(axis=1, method="average")
    denominator = mom20.notna().sum(axis=1).sub(1).replace(0, np.nan)
    features["csmom_20"] = rank.sub(1).div(denominator, axis=0)

    for fast, slow in SMA_MACD_PAIRS:
        fast_average = prices.rolling(fast, min_periods=fast).mean()
        slow_average = prices.rolling(slow, min_periods=slow).mean()
        features[f"sma_macd_{fast}_{slow}"] = fast_average.div(slow_average).sub(1.0)

    fast_ewma = prices.ewm(alpha=0.5, adjust=False).mean()
    slow_ewma = prices.ewm(alpha=1 / 3, adjust=False).mean()
    features["ewma_macd_0500_0333"] = fast_ewma.div(slow_ewma).sub(1.0)
    return features


def preprocess_feature(
    raw: pd.DataFrame,
    lookback: int = PREPROCESS_LOOKBACK,
    min_periods: int = PREPROCESS_MIN_PERIODS,
) -> ProcessedFeature:
    """Winsorize and normalize using only observations before the current date."""
    clean = raw.replace([np.inf, -np.inf], np.nan)
    lower = clean.rolling(lookback, min_periods=min_periods).quantile(0.025).shift(1)
    upper = clean.rolling(lookback, min_periods=min_periods).quantile(0.975).shift(1)
    winsorized = clean.clip(lower=lower, upper=upper, axis=None)
    historical_mean = winsorized.rolling(lookback, min_periods=min_periods).mean().shift(1)
    historical_std = winsorized.rolling(lookback, min_periods=min_periods).std(ddof=1).shift(1)
    normalized = winsorized.sub(historical_mean).div(historical_std.replace(0.0, np.nan))
    return ProcessedFeature(clean, lower, upper, normalized.replace([np.inf, -np.inf], np.nan))


def cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove each date's common component and scale its ETF dispersion."""
    centered = frame.sub(frame.mean(axis=1), axis=0)
    dispersion = centered.std(axis=1, ddof=1).replace(0.0, np.nan)
    return centered.div(dispersion, axis=0).clip(-5.0, 5.0)


def execution_aligned_returns(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Calculate absolute and cross-section-relative returns earned from t+1 to t+6."""
    exit_offset = EXECUTION_LAG_DAYS + HOLDING_DAYS
    absolute = prices.shift(-exit_offset).div(prices.shift(-EXECUTION_LAG_DAYS)).sub(1.0)
    relative = absolute.sub(absolute.mean(axis=1), axis=0)
    dates = pd.Series(prices.index, index=prices.index)
    return (
        absolute,
        relative,
        dates.shift(-EXECUTION_LAG_DAYS),
        dates.shift(-exit_offset),
    )


def build_cross_sectional_panel(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, ProcessedFeature]]:
    """Build the cross-sectional modeling panel used by the final pipeline."""
    processed = {
        name: preprocess_feature(raw)
        for name, raw in calculate_raw_features(prices).items()
    }
    cross_sectional = {
        name: cross_sectional_zscore(item.normalized)
        for name, item in processed.items()
    }
    feature_names = list(cross_sectional)
    feature_panel = pd.concat(
        {name: frame.stack(future_stack=True) for name, frame in cross_sectional.items()},
        axis=1,
    )
    feature_panel.index.names = ["date", "symbol"]

    absolute, relative, entry_date, label_end_date = execution_aligned_returns(prices)
    panel = feature_panel.copy()
    panel["future_5d_return"] = absolute.stack(future_stack=True)
    panel["future_5d_relative_return"] = relative.stack(future_stack=True)
    panel = panel.reset_index()
    panel["entry_date"] = panel["date"].map(entry_date)
    panel["label_end_date"] = panel["date"].map(label_end_date)
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    return panel, feature_names, processed

"""Prepare MOM, MOMRA, and execution-aligned five-day forward returns.

Signals are calculated on the full continuous adjusted-price history before
being assigned to a sample.  This lets a signal on the first validation or
test date use the 20 prior trading days, while all signal inputs remain in the
past.  Rows whose five-day outcome crosses a sample boundary are excluded from
training and validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_split import (
    TEST_START,
    TRAIN_START,
    VALIDATION_START,
    load_close_prices,
)


LOOKBACK_DAYS = 20
FORWARD_DAYS = 5
EXECUTION_LAG_DAYS = 1
PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "results" / "signals" / "data"


def assign_clean_samples(dates: pd.DatetimeIndex, forward_end_dates: pd.Series) -> pd.Series:
    """Label signal dates while preventing forward returns from crossing splits."""
    sample = pd.Series(pd.NA, index=dates, dtype="string")

    training = (dates >= TRAIN_START) & (dates < VALIDATION_START) & (forward_end_dates < VALIDATION_START)
    validation = (dates >= VALIDATION_START) & (dates < TEST_START) & (forward_end_dates < TEST_START)
    test = (dates >= TEST_START) & forward_end_dates.notna()

    sample.loc[training] = "training"
    sample.loc[validation] = "validation"
    sample.loc[test] = "test"
    return sample


def calculate_signal_panel(
    prices: pd.DataFrame,
    lookback_days: int = LOOKBACK_DAYS,
    forward_days: int = FORWARD_DAYS,
) -> pd.DataFrame:
    """Return one long table of MOM, MOMRA, and the next-five-day return.

    MOM is the mean of the previous ``lookback_days`` close-to-close returns.
    MOMRA is the z-score of that mean: MOM divided by its estimated standard
    error, ``return standard deviation / sqrt(lookback_days)``.
    A signal observed at today's close is executable one session later.  The
    forward return therefore starts at the next close and ends after
    ``forward_days`` return-days, matching the overlapping-cohort backtester.
    """
    if lookback_days < 2 or forward_days < 1:
        raise ValueError("lookback_days must be at least 2 and forward_days at least 1")

    daily_return = prices.pct_change(fill_method=None)
    mom = daily_return.rolling(lookback_days, min_periods=lookback_days).mean()
    rolling_volatility = daily_return.rolling(lookback_days, min_periods=lookback_days).std(ddof=1)
    momentum_risk = rolling_volatility.div(np.sqrt(lookback_days))
    momra = mom.div(momentum_risk.replace(0.0, np.nan))
    exit_offset = EXECUTION_LAG_DAYS + forward_days
    entry_price = prices.shift(-EXECUTION_LAG_DAYS)
    exit_price = prices.shift(-exit_offset)
    forward_5d_return = exit_price.div(entry_price).sub(1.0)

    forward_end_dates = pd.Series(prices.index, index=prices.index).shift(-exit_offset)
    sample_by_date = assign_clean_samples(prices.index, forward_end_dates)

    panel = pd.concat(
        {
            "close": prices.stack(),
            "daily_return": daily_return.stack(),
            "mom": mom.stack(),
            "rolling_volatility": rolling_volatility.stack(),
            "momentum_risk": momentum_risk.stack(),
            "momra": momra.stack(),
            "forward_5d_return": forward_5d_return.stack(),
        },
        axis=1,
    )
    panel.index.names = ["date", "symbol"]
    panel = panel.reset_index()
    panel["forward_end_date"] = panel["date"].map(forward_end_dates)
    panel["sample"] = panel["date"].map(sample_by_date)

    # A usable observation has a complete historical signal, a complete
    # five-day future outcome, and belongs to one clean date sample.
    panel = panel.dropna(
        subset=["mom", "rolling_volatility", "momentum_risk", "momra", "forward_5d_return", "sample"]
    )
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)


def signal_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize the usable signal rows for a quick audit."""
    return (
        panel.groupby("sample", observed=True)
        .agg(
            first_signal_date=("date", "min"),
            last_signal_date=("date", "max"),
            etfs=("symbol", "nunique"),
            observations=("symbol", "size"),
            mean_mom=("mom", "mean"),
            mean_momra=("momra", "mean"),
            mean_forward_5d_return=("forward_5d_return", "mean"),
        )
        .reset_index()
    )


def save_signal_data(output_dir: str | Path = OUTPUT_DIR) -> pd.DataFrame:
    """Calculate and save the signal panel plus a summary CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prices = load_close_prices()
    panel = calculate_signal_panel(prices)
    summary = signal_summary(panel)
    panel.to_csv(output_dir / "mom_momra_forward_returns.csv", index=False)
    summary.to_csv(output_dir / "signal_summary.csv", index=False)
    return summary


if __name__ == "__main__":
    print(save_signal_data().to_string(index=False))

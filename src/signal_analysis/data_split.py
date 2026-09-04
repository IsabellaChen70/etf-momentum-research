"""Create chronological training, validation, and historical test samples.

The price series is loaded through the shared loader, which corrects the known
split artifacts in the raw close data.  Later signal calculations should be
made on the full history first, then filtered by the signal date into these
three samples.  That preserves the prior 20 days of history at the beginning
of validation and test periods without using future information.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


from src.momentum_backtest import load_close_prices


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_DIR / "results" / "signals" / "data_splits"


TRAIN_START = pd.Timestamp("2020-01-01")
VALIDATION_START = pd.Timestamp("2024-06-01")
TEST_START = pd.Timestamp("2025-06-01")


def split_price_data(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split prices into non-overlapping samples based on each trading date.

    The stated boundary dates fall on weekends in some years.  Boolean date
    filters therefore place the first available trading day on or after each
    boundary into the next sample.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must use a DatetimeIndex")

    splits = {
        "training": prices.loc[(prices.index >= TRAIN_START) & (prices.index < VALIDATION_START)].copy(),
        "validation": prices.loc[(prices.index >= VALIDATION_START) & (prices.index < TEST_START)].copy(),
        "test": prices.loc[prices.index >= TEST_START].copy(),
    }

    if any(frame.empty for frame in splits.values()):
        empty = [name for name, frame in splits.items() if frame.empty]
        raise ValueError(f"No observations in: {empty}")
    return splits


def split_summary(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a compact audit table for the saved samples."""
    return pd.DataFrame(
        {
            name: {
                "first_trading_date": frame.index.min().date().isoformat(),
                "last_trading_date": frame.index.max().date().isoformat(),
                "trading_days": len(frame),
                "etfs": frame.shape[1],
                "price_observations": frame.size,
            }
            for name, frame in splits.items()
        }
    ).T


def save_splits(output_dir: str | Path = OUTPUT_DIR) -> pd.DataFrame:
    """Load adjusted prices, save three CSV files, and return their summary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prices = load_close_prices()
    splits = split_price_data(prices)
    for name, frame in splits.items():
        frame.to_csv(output_dir / f"{name}_prices.csv", index_label="date")

    summary = split_summary(splits)
    summary.to_csv(output_dir / "split_summary.csv", index_label="sample")
    return summary


if __name__ == "__main__":
    print(save_splits().to_string())

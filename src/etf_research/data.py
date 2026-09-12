"""Canonical data loading and quality checks."""

from __future__ import annotations

from pathlib import Path
import hashlib
import zipfile

import numpy as np
import pandas as pd


KNOWN_SPLIT_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "XLB": {"2025-12-05": 0.5},
    "XLE": {"2025-12-05": 0.5},
    "XLK": {"2025-12-05": 0.5},
    "XLU": {"2025-12-05": 0.5},
    "XLY": {"2025-12-05": 0.5},
}


def source_manifest(project_root: str | Path) -> pd.DataFrame:
    """Record the exact local source files used by a research run."""
    root = Path(project_root)
    paths = [
        root / "data" / "CTA_data.zip",
        root / "data" / "adjusted" / "USO_ohlcv_1d_adjusted.csv",
        root / "data" / "adjusted" / "UNG_ohlcv_1d_adjusted.csv",
    ]
    rows = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return pd.DataFrame(rows)


def _read_close_csv(file_object: object, symbol: str) -> pd.Series:
    data = pd.read_csv(file_object)
    required = {"ts_event", "close"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{symbol} data is missing columns: {sorted(missing)}")
    dates = pd.to_datetime(data["ts_event"], utc=True).dt.tz_localize(None).dt.normalize()
    close = pd.Series(data["close"].to_numpy(dtype=float), index=dates, name=symbol)
    if close.index.has_duplicates:
        raise ValueError(f"{symbol} data contains duplicate dates")
    if not np.isfinite(close).all() or close.le(0).any():
        raise ValueError(f"{symbol} data contains invalid close prices")
    return close.sort_index()


def load_adjusted_prices(project_root: str | Path) -> pd.DataFrame:
    """Load the 37-ETF panel and apply the documented local adjustments.

    These are price-return series. The source bundle does not provide a
    complete dividend-adjusted total-return history for every ETF.
    """
    root = Path(project_root)
    zip_path = root / "data" / "CTA_data.zip"
    override_dir = root / "data" / "adjusted"
    with zipfile.ZipFile(zip_path) as archive:
        manifest = pd.read_csv(archive.open("_manifest.csv"))
        series = [
            _read_close_csv(archive.open(f"{symbol}_ohlcv_1d.csv"), symbol)
            for symbol in manifest["symbol"]
        ]

    prices = pd.concat(series, axis=1).sort_index()
    # The ETFs use one US trading calendar. Restrict the analysis to sessions
    # with a genuine observation for every ETF so stale closes cannot create
    # artificial zero returns or delayed jumps.
    prices = prices.dropna(how="any")

    for symbol, events in KNOWN_SPLIT_ADJUSTMENTS.items():
        if symbol not in prices:
            continue
        for effective_date, prior_price_factor in events.items():
            date = pd.Timestamp(effective_date)
            if date not in prices.index:
                raise ValueError(f"Missing split date {effective_date} for {symbol}")
            prices.loc[prices.index < date, symbol] *= prior_price_factor

    for symbol in ("USO", "UNG"):
        path = override_dir / f"{symbol}_ohlcv_1d_adjusted.csv"
        close = _read_close_csv(path, symbol)
        missing_dates = prices.index.difference(close.index)
        if len(missing_dates):
            raise ValueError(f"{path.name} misses {len(missing_dates)} panel dates")
        prices.loc[:, symbol] = close.reindex(prices.index)

    if prices.isna().any().any():
        missing = prices.isna().sum()
        missing = missing.loc[missing.gt(0)].to_dict()
        raise ValueError(f"Unresolved missing prices: {missing}")
    return prices.astype(float)


def data_quality_report(prices: pd.DataFrame) -> pd.DataFrame:
    """Return per-ETF observations, range, and largest adjusted move."""
    returns = prices.pct_change(fill_method=None)
    rows = []
    for symbol in prices:
        absolute = returns[symbol].abs()
        max_date = absolute.idxmax()
        rows.append(
            {
                "symbol": symbol,
                "observations": int(prices[symbol].count()),
                "first_date": prices[symbol].first_valid_index(),
                "last_date": prices[symbol].last_valid_index(),
                "minimum_price": float(prices[symbol].min()),
                "maximum_price": float(prices[symbol].max()),
                "largest_absolute_daily_return": float(absolute.loc[max_date]),
                "largest_move_date": max_date,
            }
        )
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)

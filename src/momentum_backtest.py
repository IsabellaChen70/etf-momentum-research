"""Multi-asset momentum backtester for a 37-ETF price panel.

The strategy uses the mean of the prior 20 daily close-to-close returns as its
one-month momentum signal. A new signal is traded every day; each day's trade
is held for five trading days, so the live portfolio contains five overlapping
trade cohorts. Transaction costs are intentionally excluded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import zipfile

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "results" / "momentum"
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "CTA_data.zip"
ADJUSTED_CLOSE_PATHS: dict[str, Path] = {
    "USO": PROJECT_DIR / "data" / "adjusted" / "USO_ohlcv_1d_adjusted.csv",
    "UNG": PROJECT_DIR / "data" / "adjusted" / "UNG_ohlcv_1d_adjusted.csv",
}
WeightFunction = Callable[[pd.Series], pd.Series]

# USO and UNG are loaded directly from the supplied adjusted CSVs above.  The
# remaining factors correct documented splits in the raw data bundle.
KNOWN_SPLIT_ADJUSTMENTS: dict[str, dict[str, float]] = {
    "XLB": {"2025-12-05": 0.5},       # 2-for-1 forward split
    "XLE": {"2025-12-05": 0.5},
    "XLK": {"2025-12-05": 0.5},
    "XLU": {"2025-12-05": 0.5},
    "XLY": {"2025-12-05": 0.5},
}


def load_close_prices(
    zip_path: str | Path = DEFAULT_DATA_PATH,
    symbols: list[str] | None = None,
    adjust_known_splits: bool = True,
) -> pd.DataFrame:
    """Load close prices for each requested ETF in the input archive.

    Missing ETF observations are forward-filled after the individual series
    are aligned to one common trading-date index. This gives a zero return on
    a date when a particular ETF has no observation but the other ETFs trade.

    When adjustments are enabled, USO and UNG are replaced directly with the
    supplied adjusted CSV close series. Other documented stock splits are
    corrected from the raw ZIP so no split masquerades as a one-day return.
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        manifest = pd.read_csv(archive.open("_manifest.csv"))
        if symbols is not None:
            requested = set(symbols)
            manifest = manifest[manifest["symbol"].isin(requested)].copy()
            absent = requested - set(manifest["symbol"])
            if absent:
                raise ValueError(f"Symbols not in the ZIP: {sorted(absent)}")

        close_series = []
        for symbol in manifest["symbol"]:
            data = pd.read_csv(archive.open(f"{symbol}_ohlcv_1d.csv"))
            dates = pd.to_datetime(data["ts_event"], utc=True).dt.tz_localize(None)
            close_series.append(pd.Series(data["close"].to_numpy(), index=dates, name=symbol))

    prices = pd.concat(close_series, axis=1).sort_index().ffill()
    if adjust_known_splits:
        prices = adjust_prices_for_known_splits(prices)
        prices = apply_adjusted_close_overrides(prices)
    return prices


def apply_adjusted_close_overrides(prices: pd.DataFrame) -> pd.DataFrame:
    """Replace USO/UNG closes with the supplied, directly loaded adjusted CSVs."""
    adjusted = prices.copy()
    for symbol, csv_path in ADJUSTED_CLOSE_PATHS.items():
        if symbol not in adjusted.columns:
            continue
        if not csv_path.exists():
            raise FileNotFoundError(f"Adjusted close file is missing: {csv_path}")

        data = pd.read_csv(csv_path)
        required_columns = {"ts_event", "close"}
        missing_columns = required_columns - set(data.columns)
        if missing_columns:
            raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing_columns)}")

        dates = pd.to_datetime(data["ts_event"], utc=True).dt.tz_localize(None)
        close = pd.Series(data["close"].to_numpy(dtype=float), index=dates, name=symbol)
        if close.index.has_duplicates:
            raise ValueError(f"{csv_path.name} contains duplicate dates")

        missing_dates = adjusted.index.difference(close.index)
        if len(missing_dates):
            raise ValueError(
                f"{csv_path.name} is missing {len(missing_dates)} required trading dates"
            )
        adjusted.loc[:, symbol] = close.reindex(adjusted.index)
    return adjusted


def adjust_prices_for_known_splits(prices: pd.DataFrame) -> pd.DataFrame:
    """Put raw close prices on a split-adjusted basis for known bundle events.

    The function intentionally makes only documented mechanical corrections;
    it does not smooth or alter genuine market moves.
    """
    adjusted = prices.copy()
    for symbol, events in KNOWN_SPLIT_ADJUSTMENTS.items():
        if symbol not in adjusted:
            continue
        for effective_date, prior_price_factor in events.items():
            date = pd.Timestamp(effective_date)
            if date not in adjusted.index:
                raise ValueError(f"Split date {effective_date} for {symbol} is missing from prices")
            adjusted.loc[adjusted.index < date, symbol] *= prior_price_factor
    return adjusted


def one_month_average_return(prices: pd.DataFrame, lookback_days: int = 20) -> pd.DataFrame:
    """Calculate average daily returns over the prior approximately one month."""
    daily_returns = prices.pct_change(fill_method=None)
    return daily_returns.rolling(lookback_days, min_periods=lookback_days).mean()


def equal_positive_negative_weights(
    signal: pd.Series,
    long_gross: float = 1.50,
    short_gross: float = 0.50,
) -> pd.Series:
    """Allocate equally within positive- and negative-momentum groups."""
    weights = pd.Series(0.0, index=signal.index)
    positive = signal > 0
    negative = signal < 0

    if positive.any():
        weights.loc[positive] = long_gross / positive.sum()
    if negative.any():
        weights.loc[negative] = -short_gross / negative.sum()
    return weights


def relative_momentum_weights(
    signal: pd.Series,
    long_gross: float = 1.50,
    short_gross: float = 0.50,
) -> pd.Series:
    """Split the universe into higher- and lower-momentum assets.

    The cutoff is the top half of the cross-sectional momentum ranking. With
    three assets, the two strongest are long and the weakest is short.

    This matches Portfolio 1's 150% long / 50% short exposure. With three
    assets, the two strongest receive +75% each and the weakest receives -50%.
    """
    weights = pd.Series(0.0, index=signal.index)
    valid_signal = signal.dropna()
    if len(valid_signal) < 2:
        return weights

    n_assets = len(valid_signal)
    n_long = int(np.ceil(n_assets / 2))
    ranked = valid_signal.rank(method="first", ascending=False)
    long_assets = ranked[ranked <= n_long].index
    short_assets = ranked[ranked > n_long].index

    weights.loc[long_assets] = long_gross / len(long_assets)
    weights.loc[short_assets] = -short_gross / len(short_assets)
    return weights


def daily_trade_targets(
    signals: pd.DataFrame,
    weight_function: WeightFunction,
) -> pd.DataFrame:
    """Create a new target-weight trade for every date with a complete signal."""
    targets = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    for date, signal in signals.dropna().iterrows():
        targets.loc[date] = weight_function(signal)
    return targets


def multi_asset_backtest(
    prices: pd.DataFrame,
    daily_weight_targets: pd.DataFrame,
    holding_days: int = 5,
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """Backtest fixed-share, overlapping multiple-asset trade cohorts.

    A target formed at close *t* enters at close *t + 1*. The cohort then earns
    the next ``holding_days`` close-to-close returns.
    One equal capital slice is assigned to each daily trade cohort, so after the
    initial ramp-up there are ``holding_days`` overlapping cohorts. At entry,
    each cohort takes a fixed-share long or short position. Its weights are not
    reset while it is held; they naturally drift with prices until the cohort
    closes after ``holding_days`` return-days. Portfolio weights may be
    leveraged and do not need to sum to one; unused capital or borrowing is
    implicit in the portfolio value.
    """
    if holding_days < 1:
        raise ValueError("holding_days must be at least 1")
    if not prices.index.equals(daily_weight_targets.index):
        raise ValueError("prices and daily targets must have the same date index")
    if list(prices.columns) != list(daily_weight_targets.columns):
        raise ValueError("prices and daily targets must have matching columns")

    columns = prices.columns
    n_dates = len(prices)
    portfolio_values = np.full(n_dates, np.nan)
    portfolio_returns = np.zeros(n_dates)
    net_exposures = np.zeros(n_dates)
    gross_exposures = np.zeros(n_dates)
    turnovers = np.zeros(n_dates)
    held_weights = pd.DataFrame(0.0, index=prices.index, columns=columns)

    # Each item is a date index and a vector of shares.  Keeping shares (rather
    # than weights) is what prevents within-cohort daily rebalancing.
    active_cohorts: list[tuple[int, pd.Series]] = []
    portfolio_value = float(initial_capital)

    for i, date in enumerate(prices.index):
        today_price = prices.iloc[i]
        prior_value = portfolio_value

        if i > 0 and active_cohorts:
            price_change = today_price - prices.iloc[i - 1]
            daily_pnl = sum((shares * price_change).sum() for _, shares in active_cohorts)
            portfolio_value += daily_pnl
            portfolio_returns[i] = daily_pnl / prior_value if prior_value else np.nan

        # A cohort earns exactly holding_days close-to-close returns.  Close
        # its final-day positions before opening today's new cohort.
        closing_cohorts = [(entry_i, shares) for entry_i, shares in active_cohorts if i - entry_i >= holding_days]
        active_cohorts = [(entry_i, shares) for entry_i, shares in active_cohorts if i - entry_i < holding_days]
        closing_notional = sum((shares * today_price).abs().sum() for _, shares in closing_cohorts)

        # The target known at yesterday's close is entered at today's close.
        # Dividing the current account value by five makes the five overlapping
        # cohorts add up to the requested 150% long / 50% short exposure.
        opening_notional = 0.0
        if i > 0 and portfolio_value > 0:
            target = daily_weight_targets.iloc[i - 1].fillna(0.0)
            cohort_capital = portfolio_value / holding_days
            new_shares = cohort_capital * target / today_price
            if new_shares.ne(0.0).any():
                active_cohorts.append((i, new_shares))
                opening_notional = (new_shares * today_price).abs().sum()

        if active_cohorts and portfolio_value:
            aggregate_shares = sum(
                (shares for _, shares in active_cohorts),
                start=pd.Series(0.0, index=columns),
            )
            current_weights = aggregate_shares * today_price / portfolio_value
        else:
            current_weights = pd.Series(0.0, index=columns)

        held_weights.iloc[i] = current_weights
        portfolio_values[i] = portfolio_value
        net_exposures[i] = current_weights.sum()
        gross_exposures[i] = current_weights.abs().sum()
        turnovers[i] = (opening_notional + closing_notional) / portfolio_value if portfolio_value else np.nan

    result = pd.DataFrame(
        {
            "portfolio_return": portfolio_returns,
            "portfolio_value": portfolio_values,
            "net_exposure": net_exposures,
            "gross_exposure": gross_exposures,
            "turnover": turnovers,
        },
        index=prices.index,
    )
    result.index.name = "date"
    return result.join(held_weights.add_prefix("weight_"))


def performance_summary(backtest: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """Return simple performance diagnostics for a backtest result."""
    returns = backtest["portfolio_return"]
    active_returns = returns[returns.ne(0)]
    n_periods = len(returns)
    ending_value = backtest["portfolio_value"].iloc[-1]
    annualized_return = (ending_value / backtest["portfolio_value"].iloc[0]) ** (periods_per_year / n_periods) - 1
    annualized_volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    running_peak = backtest["portfolio_value"].cummax()
    max_drawdown = (backtest["portfolio_value"] / running_peak - 1).min()

    return pd.Series(
        {
            "start": backtest.index.min().date().isoformat(),
            "end": backtest.index.max().date().isoformat(),
            "ending_value": ending_value,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "return_to_volatility": annualized_return / annualized_volatility if annualized_volatility else np.nan,
            "max_drawdown": max_drawdown,
            "average_gross_exposure": backtest["gross_exposure"].mean(),
            "average_daily_turnover": backtest["turnover"].mean(),
            "active_days": len(active_returns),
        }
    )


def plot_portfolio_values(
    results: dict[str, pd.DataFrame],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot cumulative values for the two momentum portfolios."""
    labels = {
        "equal_sign_groups_150_50": "Equal sign groups (150/50)",
        "relative_rank_groups_150_50": "Relative rank groups (150/50)",
    }
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for name, result in results.items():
        ax.plot(result.index, result["portfolio_value"], linewidth=1.7, label=labels.get(name, name))

    ax.set_title("Daily Momentum Portfolio Backtests")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value (starting capital = 1)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def run_momentum_backtests(
    symbols: list[str] | None = None,
    lookback_days: int = 20,
    holding_days: int = 5,
    zip_path: str | Path = DEFAULT_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Run both allocation rules and return prices, signals, results, and summary."""
    prices = load_close_prices(zip_path=zip_path, symbols=symbols)
    signals = one_month_average_return(prices, lookback_days=lookback_days)

    p1_targets = daily_trade_targets(
        signals,
        lambda signal: equal_positive_negative_weights(signal, long_gross=1.50, short_gross=0.50),
    )
    p2_targets = daily_trade_targets(
        signals,
        lambda signal: relative_momentum_weights(signal, long_gross=1.50, short_gross=0.50),
    )

    results = {
        "equal_sign_groups_150_50": multi_asset_backtest(prices, p1_targets, holding_days=holding_days),
        "relative_rank_groups_150_50": multi_asset_backtest(prices, p2_targets, holding_days=holding_days),
    }
    summary = pd.DataFrame({name: performance_summary(result) for name, result in results.items()}).T
    return prices, signals, results, summary


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, _, backtest_results, performance = run_momentum_backtests()
    performance.to_csv(OUTPUT_DIR / "portfolio_performance.csv", index_label="portfolio")
    plot_portfolio_values(backtest_results, OUTPUT_DIR / "portfolio_comparison.png")
    print(performance.to_string())

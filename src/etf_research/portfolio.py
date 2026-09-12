"""Constrained portfolio construction and cost-aware cohort backtesting."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize
from sklearn.covariance import LedoitWolf


HOLDING_DAYS = 5
EXECUTION_LAG_DAYS = 1
PERIODS_PER_YEAR = 252
MIN_SIGNAL_DISPERSION = 1e-10


def covariance_for_date(
    returns: pd.DataFrame,
    date: pd.Timestamp,
    lookback: int = 126,
    min_periods: int = 63,
) -> pd.DataFrame:
    """Estimate a five-day covariance matrix from returns available by date."""
    window = returns.loc[:date].tail(lookback).dropna(how="any")
    if len(window) < min_periods:
        raise ValueError(f"Insufficient covariance history at {date.date()}")
    estimator = LedoitWolf(assume_centered=False).fit(window.to_numpy(dtype=float))
    covariance = estimator.covariance_ * HOLDING_DAYS
    return pd.DataFrame(covariance, index=returns.columns, columns=returns.columns)


def optimize_market_neutral_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float,
    previous_weights: pd.Series | None = None,
    gross_limit: float = 2.0,
    position_limit: float = 0.15,
    turnover_penalty: float = 0.0025,
) -> tuple[pd.Series, dict[str, float | str]]:
    """Optimize risky holdings while allowing the portfolio to remain in cash."""
    expected = expected_returns.reindex(covariance.index).replace([np.inf, -np.inf], np.nan)
    if expected.isna().any():
        raise ValueError("Expected returns are incomplete")
    expected = expected - expected.mean()
    if float(expected.std(ddof=1)) <= MIN_SIGNAL_DISPERSION:
        zero = pd.Series(0.0, index=expected.index)
        return zero, {
            "status": "cash_no_signal",
            "target_net_exposure": 0.0,
            "target_gross_exposure": 0.0,
            "largest_absolute_weight": 0.0,
            "expected_5d_return": 0.0,
            "expected_5d_volatility": 0.0,
        }

    mu = expected.to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)
    n_assets = len(mu)
    if previous_weights is None:
        previous = np.zeros(n_assets)
    else:
        previous = previous_weights.reindex(expected.index).fillna(0.0).to_numpy(dtype=float)

    initial = np.concatenate([previous, np.abs(previous)])
    net_row = np.concatenate([np.ones(n_assets), np.zeros(n_assets)])[None, :]
    gross_row = np.concatenate([np.zeros(n_assets), np.ones(n_assets)])[None, :]
    z_minus_w = np.concatenate([-np.eye(n_assets), np.eye(n_assets)], axis=1)
    z_plus_w = np.concatenate([np.eye(n_assets), np.eye(n_assets)], axis=1)
    absolute_matrix = np.vstack([gross_row, z_minus_w, z_plus_w])
    absolute_lower = np.concatenate(
        [[-np.inf], np.zeros(n_assets), np.zeros(n_assets)]
    )
    absolute_upper = np.concatenate(
        [[gross_limit], np.full(n_assets * 2, np.inf)]
    )

    constraints = [
        LinearConstraint(net_row, [0.0], [0.0]),
        LinearConstraint(absolute_matrix, absolute_lower, absolute_upper),
    ]
    bounds = Bounds(
        np.concatenate(
            [np.full(n_assets, -position_limit), np.zeros(n_assets)]
        ),
        np.concatenate(
            [np.full(n_assets, position_limit), np.full(n_assets, position_limit)]
        ),
    )

    def objective(x: np.ndarray) -> float:
        weights = x[:n_assets]
        variance = float(weights @ sigma @ weights)
        turnover = float(np.square(weights - previous).sum())
        return (
            0.5 * risk_aversion * variance
            - float(mu @ weights)
            + 0.5 * turnover_penalty * turnover
        )

    def gradient(x: np.ndarray) -> np.ndarray:
        weights = x[:n_assets]
        weight_gradient = (
            risk_aversion * sigma @ weights
            - mu
            + turnover_penalty * (weights - previous)
        )
        return np.concatenate([weight_gradient, np.zeros(n_assets)])

    solution = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
    )
    if not solution.success:
        raise RuntimeError(f"Portfolio optimization failed: {solution.message}")

    weights = pd.Series(solution.x[:n_assets], index=expected.index)
    net = float(weights.sum())
    gross = float(weights.abs().sum())
    if abs(net) > 1e-7:
        raise AssertionError(f"Market-neutral net exposure is {net}")
    if gross > gross_limit + 1e-7:
        raise AssertionError(f"Gross exposure {gross} exceeds {gross_limit}")
    if weights.abs().max() > position_limit + 1e-7:
        raise AssertionError("An ETF exceeds the position limit")

    vector = weights.to_numpy(dtype=float)
    return weights, {
        "status": "optimized",
        "target_net_exposure": net,
        "target_gross_exposure": gross,
        "largest_absolute_weight": float(weights.abs().max()),
        "expected_5d_return": float(mu @ vector),
        "expected_5d_volatility": float(np.sqrt(max(vector @ sigma @ vector, 0.0))),
    }


def build_optimized_targets(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    risk_aversion: float,
    gross_limit: float,
    position_limit: float,
    turnover_penalty: float,
    covariance_lookback: int,
    covariance_min_periods: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create sequential targets using only historical covariance observations."""
    targets = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    returns = prices.pct_change(fill_method=None)
    previous = pd.Series(0.0, index=prices.columns)
    diagnostics = []
    for date in signal_dates:
        forecast = signal.loc[date].reindex(prices.columns)
        if forecast.isna().any():
            previous = pd.Series(0.0, index=prices.columns)
            continue
        covariance = covariance_for_date(
            returns,
            date,
            lookback=covariance_lookback,
            min_periods=covariance_min_periods,
        )
        weights, row = optimize_market_neutral_weights(
            forecast,
            covariance,
            risk_aversion,
            previous_weights=previous,
            gross_limit=gross_limit,
            position_limit=position_limit,
            turnover_penalty=turnover_penalty,
        )
        targets.loc[date] = weights
        previous = weights
        diagnostics.append({"date": date, "risk_aversion": risk_aversion, **row})
    return targets, pd.DataFrame(diagnostics)


def build_rank_targets(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    gross_limit: float = 2.0,
) -> pd.DataFrame:
    """Create an equal-weight top-minus-bottom quintile benchmark."""
    targets = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for date in signal_dates:
        forecast = signal.loc[date].dropna()
        if len(forecast) < 5 or forecast.std(ddof=1) <= MIN_SIGNAL_DISPERSION:
            continue
        count = max(int(np.floor(len(forecast) * 0.2)), 1)
        ordered = forecast.sort_values()
        short_assets = ordered.index[:count]
        long_assets = ordered.index[-count:]
        targets.loc[date, long_assets] = (gross_limit / 2) / count
        targets.loc[date, short_assets] = -(gross_limit / 2) / count
    return targets


def build_equal_weight_targets(
    prices: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Create a fully invested long-only benchmark."""
    targets = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    targets.loc[signal_dates] = 1.0 / prices.shape[1]
    return targets


def purge_incomplete_holds(
    targets: pd.DataFrame,
    sample_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Remove signals that cannot complete lag plus five return-days."""
    clean = targets.copy()
    unavailable = EXECUTION_LAG_DAYS + HOLDING_DAYS
    if len(sample_dates) <= unavailable:
        clean.loc[sample_dates] = 0.0
    else:
        clean.loc[sample_dates[-unavailable:]] = 0.0
    return clean


def backtest_cohorts(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    transaction_cost_bps: float = 5.0,
    annual_short_borrow_bps: float = 25.0,
    initial_capital: float = 1.0,
) -> pd.DataFrame:
    """Backtest five fixed-share cohorts with explicit trading and borrow costs."""
    if not prices.index.equals(targets.index):
        raise ValueError("Price and target indexes differ")
    if list(prices.columns) != list(targets.columns):
        raise ValueError("Price and target columns differ")

    transaction_rate = transaction_cost_bps / 10_000
    borrow_rate = annual_short_borrow_bps / 10_000 / PERIODS_PER_YEAR
    active: list[tuple[int, pd.Series]] = []
    value = float(initial_capital)
    rows = []

    for i, date in enumerate(prices.index):
        today = prices.iloc[i]
        prior_value = value
        gross_pnl = 0.0
        borrow_cost = 0.0
        if i > 0 and active:
            yesterday = prices.iloc[i - 1]
            gross_pnl = float(
                sum((shares * (today - yesterday)).sum() for _, shares in active)
            )
            short_notional = float(
                sum((shares.clip(upper=0.0) * yesterday).abs().sum() for _, shares in active)
            )
            borrow_cost = short_notional * borrow_rate
            value += gross_pnl - borrow_cost

        aggregate_before_trade = sum(
            (shares for _, shares in active),
            start=pd.Series(0.0, index=prices.columns),
        )
        closing = [
            (entry_index, shares)
            for entry_index, shares in active
            if i - entry_index >= HOLDING_DAYS
        ]
        active = [
            (entry_index, shares)
            for entry_index, shares in active
            if i - entry_index < HOLDING_DAYS
        ]
        if i > 0 and value > 0:
            target = targets.iloc[i - 1].fillna(0.0)
            cohort_capital = value / HOLDING_DAYS
            shares = cohort_capital * target / today
            if shares.ne(0.0).any():
                active.append((i, shares))

        aggregate = sum(
            (shares for _, shares in active),
            start=pd.Series(0.0, index=prices.columns),
        )
        transaction_notional = float(
            ((aggregate - aggregate_before_trade) * today).abs().sum()
        )
        transaction_cost = transaction_notional * transaction_rate
        value -= transaction_cost
        if value <= 0:
            raise RuntimeError("Portfolio value became non-positive")

        current_weights = aggregate * today / value
        rows.append(
            {
                "date": date,
                "gross_return_before_costs": gross_pnl / prior_value if prior_value else np.nan,
                "transaction_cost": transaction_cost / prior_value if prior_value else np.nan,
                "borrow_cost": borrow_cost / prior_value if prior_value else np.nan,
                "portfolio_return": (value - prior_value) / prior_value if prior_value else np.nan,
                "portfolio_value": value,
                "net_exposure": float(current_weights.sum()),
                "gross_exposure": float(current_weights.abs().sum()),
                "turnover": transaction_notional / value,
            }
        )
    return pd.DataFrame(rows).set_index("date")


def performance_metrics(result: pd.DataFrame, strategy: str, sample: str) -> dict[str, object]:
    """Calculate standard net-of-cost portfolio statistics."""
    returns = result["portfolio_return"].fillna(0.0)
    volatility = float(returns.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
    sharpe = float(returns.mean() * PERIODS_PER_YEAR / volatility) if volatility > 0 else np.nan
    value = result["portfolio_value"]
    years = max((len(value) - 1) / PERIODS_PER_YEAR, 1 / PERIODS_PER_YEAR)
    drawdown = value.div(value.cummax()).sub(1.0)
    downside = returns.loc[returns.lt(0)].std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    return {
        "sample": sample,
        "strategy": strategy,
        "start": value.index.min(),
        "end": value.index.max(),
        "trading_days": len(value),
        "ending_value": float(value.iloc[-1]),
        "annualized_return": float(value.iloc[-1] ** (1 / years) - 1),
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "sortino": float(returns.mean() * PERIODS_PER_YEAR / downside) if downside > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "calmar": float((value.iloc[-1] ** (1 / years) - 1) / abs(drawdown.min()))
        if drawdown.min() < 0 else np.nan,
        "average_gross_exposure": float(result["gross_exposure"].mean()),
        "average_net_exposure": float(result["net_exposure"].mean()),
        "average_daily_turnover": float(result["turnover"].mean()),
        "total_transaction_cost": float(result["transaction_cost"].sum()),
        "total_borrow_cost": float(result["borrow_cost"].sum()),
    }

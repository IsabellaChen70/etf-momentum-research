"""Unit tests for the research pipeline components."""

from __future__ import annotations

import numpy as np
import pandas as pd

from etf_research.features import (
    execution_aligned_returns,
    preprocess_feature,
)
from etf_research.models import fold_rows
from etf_research.portfolio import (
    backtest_cohorts,
    optimize_market_neutral_weights,
    purge_incomplete_holds,
)
from etf_research.risk_overlay import (
    apply_volatility_regime_overlay,
    realized_volatility_regime,
)


def test_execution_target_begins_after_one_day_and_holds_five_returns() -> None:
    dates = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame(
        {
            "A": np.arange(10, 20, dtype=float),
            "B": np.arange(20, 30, dtype=float),
        },
        index=dates,
    )
    absolute, relative, entry, label_end = execution_aligned_returns(prices)
    assert absolute.loc[dates[0], "A"] == prices.loc[dates[6], "A"] / prices.loc[dates[1], "A"] - 1
    assert entry.loc[dates[0]] == dates[1]
    assert label_end.loc[dates[0]] == dates[6]
    assert abs(relative.loc[dates[0]].mean()) < 1e-15


def test_preprocessing_does_not_use_future_values() -> None:
    dates = pd.bdate_range("2023-01-01", periods=160)
    raw = pd.DataFrame(
        {
            "A": np.linspace(-1, 1, len(dates)),
            "B": np.linspace(1, -1, len(dates)),
        },
        index=dates,
    )
    original = preprocess_feature(raw, lookback=40, min_periods=20).normalized
    changed = raw.copy()
    changed.iloc[-1] = [10_000.0, -10_000.0]
    revised = preprocess_feature(changed, lookback=40, min_periods=20).normalized
    pd.testing.assert_frame_equal(original.iloc[:-1], revised.iloc[:-1])


def test_identical_forecasts_move_to_cash() -> None:
    symbols = ["A", "B", "C", "D"]
    forecast = pd.Series(0.01, index=symbols)
    covariance = pd.DataFrame(np.eye(4) * 0.01, index=symbols, columns=symbols)
    weights, diagnostics = optimize_market_neutral_weights(
        forecast,
        covariance,
        risk_aversion=10.0,
    )
    assert np.allclose(weights, 0.0)
    assert diagnostics["status"] == "cash_no_signal"


def test_optimizer_respects_net_gross_and_position_limits() -> None:
    symbols = ["A", "B", "C", "D", "E", "F", "G", "H"]
    forecast = pd.Series(np.linspace(-0.02, 0.02, len(symbols)), index=symbols)
    covariance = pd.DataFrame(np.eye(len(symbols)) * 0.002, index=symbols, columns=symbols)
    weights, diagnostics = optimize_market_neutral_weights(
        forecast,
        covariance,
        risk_aversion=10.0,
        gross_limit=1.0,
        position_limit=0.20,
    )
    assert abs(weights.sum()) < 1e-7
    assert weights.abs().sum() <= 1.0 + 1e-7
    assert weights.abs().max() <= 0.20 + 1e-7
    assert diagnostics["status"] == "optimized"


def test_costs_reduce_the_same_fixed_share_backtest() -> None:
    dates = pd.bdate_range("2024-01-01", periods=15)
    prices = pd.DataFrame(
        {
            "A": np.linspace(100, 115, len(dates)),
            "B": np.linspace(100, 95, len(dates)),
        },
        index=dates,
    )
    targets = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    targets.loc[dates[:8], "A"] = 1.0
    targets.loc[dates[:8], "B"] = -1.0
    free = backtest_cohorts(prices, targets, transaction_cost_bps=0, annual_short_borrow_bps=0)
    costed = backtest_cohorts(prices, targets, transaction_cost_bps=5, annual_short_borrow_bps=25)
    assert costed["portfolio_value"].iloc[-1] < free["portfolio_value"].iloc[-1]
    assert costed["transaction_cost"].sum() > 0
    assert costed["borrow_cost"].sum() > 0


def test_cohort_orders_are_netted_before_transaction_costs() -> None:
    dates = pd.bdate_range("2024-01-01", periods=15)
    prices = pd.DataFrame(100.0, index=dates, columns=["A", "B"])
    targets = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    targets.loc[:, "A"] = 1.0
    targets.loc[:, "B"] = -1.0
    result = backtest_cohorts(
        prices,
        targets,
        transaction_cost_bps=5,
        annual_short_borrow_bps=0,
    )
    # Once five identical cohorts are active, replacing one cohort requires
    # almost no net market trade because closing and opening shares offset.
    assert result["turnover"].iloc[6:10].max() < 1e-3


def test_incomplete_boundary_holds_are_removed() -> None:
    dates = pd.bdate_range("2024-01-01", periods=12)
    targets = pd.DataFrame(1.0, index=dates, columns=["A", "B"])
    clean = purge_incomplete_holds(targets, dates)
    assert clean.loc[dates[-6:]].abs().to_numpy().max() == 0.0
    assert clean.loc[dates[:-6]].abs().to_numpy().min() == 1.0


def test_walk_forward_labels_are_purged_at_month_boundaries() -> None:
    dates = pd.bdate_range("2024-01-01", "2024-03-29")
    panel = pd.DataFrame(
        {
            "date": dates,
            "symbol": "A",
            "feature": np.linspace(-1.0, 1.0, len(dates)),
            "future_5d_relative_return": 0.01,
            "label_end_date": dates + pd.offsets.BDay(6),
        }
    )
    train, validation, prediction = fold_rows(
        panel,
        ["feature"],
        pd.Period("2024-03", freq="M"),
        training_months=1,
        validation_months=1,
    )
    assert train["label_end_date"].max() <= pd.Timestamp("2024-01-31")
    assert validation["label_end_date"].max() <= pd.Timestamp("2024-02-29")
    assert prediction["date"].min() >= pd.Timestamp("2024-03-01")


def test_volatility_regime_threshold_does_not_use_future_prices() -> None:
    dates = pd.bdate_range("2022-01-03", periods=420)
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(len(dates), 6)),
        index=dates,
        columns=list("ABCDEF"),
    )
    prices = 100.0 * (1.0 + returns).cumprod()
    original = realized_volatility_regime(
        prices,
        regime_history=126,
        regime_min_periods=63,
    )
    changed = prices.copy()
    changed.iloc[-1] *= 3.0
    revised = realized_volatility_regime(
        changed,
        regime_history=126,
        regime_min_periods=63,
    )
    pd.testing.assert_series_equal(
        original["high_volatility_threshold"],
        revised["high_volatility_threshold"],
    )
    pd.testing.assert_frame_equal(original.iloc[:-1], revised.iloc[:-1])


def test_risk_overlay_preserves_neutrality_and_only_reduces_exposure() -> None:
    dates = pd.bdate_range("2022-01-03", periods=420)
    symbols = list("ABCDEFGH")
    rng = np.random.default_rng(11)
    returns = pd.DataFrame(
        rng.normal(0.0001, 0.012, size=(len(dates), len(symbols))),
        index=dates,
        columns=symbols,
    )
    prices = 100.0 * (1.0 + returns).cumprod()
    targets = pd.DataFrame(0.0, index=dates, columns=symbols)
    signal_dates = dates[-20:]
    targets.loc[signal_dates, symbols[:4]] = 0.25
    targets.loc[signal_dates, symbols[4:]] = -0.25

    managed, diagnostics = apply_volatility_regime_overlay(
        prices,
        targets,
        signal_dates,
        target_annualized_volatility=0.08,
        gross_limit=2.0,
        regime_history=126,
        regime_min_periods=63,
    )
    active = managed.loc[signal_dates]
    assert np.allclose(active.sum(axis=1), 0.0, atol=1e-10)
    assert (active.abs().sum(axis=1) <= 2.0 + 1e-10).all()
    assert diagnostics["final_scale"].between(0.0, 1.0).all()
    assert (
        diagnostics["final_predicted_annualized_volatility"]
        <= diagnostics["base_predicted_annualized_volatility"] + 1e-12
    ).all()

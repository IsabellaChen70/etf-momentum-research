"""Causal volatility targeting and realized-volatility regime controls."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .portfolio import HOLDING_DAYS, PERIODS_PER_YEAR, covariance_for_date


def realized_volatility_regime(
    prices: pd.DataFrame,
    volatility_lookback: int = 20,
    regime_history: int = 252,
    regime_min_periods: int = 126,
    high_volatility_quantile: float = 0.80,
) -> pd.DataFrame:
    """Classify high-volatility dates using a threshold known before each date."""
    if not 0.0 < high_volatility_quantile < 1.0:
        raise ValueError("high_volatility_quantile must lie between zero and one")
    if regime_min_periods > regime_history:
        raise ValueError("regime_min_periods cannot exceed regime_history")

    returns = prices.pct_change(fill_method=None)
    asset_volatility = (
        returns.rolling(
            volatility_lookback,
            min_periods=volatility_lookback,
        ).std(ddof=1)
        * np.sqrt(PERIODS_PER_YEAR)
    )
    market_volatility = asset_volatility.median(axis=1)
    threshold = (
        market_volatility.rolling(
            regime_history,
            min_periods=regime_min_periods,
        )
        .quantile(high_volatility_quantile)
        .shift(1)
    )
    regime = pd.Series("normal", index=prices.index, dtype="string")
    regime.loc[threshold.isna()] = "unavailable"
    regime.loc[threshold.notna() & market_volatility.gt(threshold)] = "high"
    return pd.DataFrame(
        {
            "market_realized_volatility": market_volatility,
            "high_volatility_threshold": threshold,
            "volatility_regime": regime,
        }
    )


def apply_volatility_regime_overlay(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    signal_dates: pd.DatetimeIndex,
    target_annualized_volatility: float = 0.10,
    maximum_scale: float = 1.0,
    high_volatility_scale: float = 0.50,
    gross_limit: float = 2.0,
    covariance_lookback: int = 126,
    covariance_min_periods: int = 63,
    regime_volatility_lookback: int = 20,
    regime_history: int = 252,
    regime_min_periods: int = 126,
    high_volatility_quantile: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale target weights with ex-ante volatility and realized-volatility state."""
    if target_annualized_volatility <= 0:
        raise ValueError("target_annualized_volatility must be positive")
    if not 0 < maximum_scale <= 1:
        raise ValueError("maximum_scale must lie in (0, 1]")
    if not 0 < high_volatility_scale <= 1:
        raise ValueError("high_volatility_scale must lie in (0, 1]")

    adjusted = pd.DataFrame(0.0, index=targets.index, columns=targets.columns)
    returns = prices.pct_change(fill_method=None)
    regime = realized_volatility_regime(
        prices,
        volatility_lookback=regime_volatility_lookback,
        regime_history=regime_history,
        regime_min_periods=regime_min_periods,
        high_volatility_quantile=high_volatility_quantile,
    )
    rows: list[dict[str, object]] = []

    for date in signal_dates:
        base = targets.loc[date].reindex(prices.columns).fillna(0.0)
        base_gross = float(base.abs().sum())
        if base_gross <= 0:
            rows.append(
                {
                    "date": date,
                    "status": "cash_no_signal",
                    "volatility_regime": regime.loc[date, "volatility_regime"],
                    "base_predicted_annualized_volatility": 0.0,
                    "volatility_scale": 0.0,
                    "regime_scale": 0.0,
                    "final_scale": 0.0,
                    "final_predicted_annualized_volatility": 0.0,
                    "final_gross_exposure": 0.0,
                }
            )
            continue

        covariance = covariance_for_date(
            returns,
            date,
            lookback=covariance_lookback,
            min_periods=covariance_min_periods,
        )
        vector = base.to_numpy(dtype=float)
        five_day_variance = float(vector @ covariance.to_numpy(dtype=float) @ vector)
        predicted_volatility = float(
            np.sqrt(max(five_day_variance, 0.0))
            * np.sqrt(PERIODS_PER_YEAR / HOLDING_DAYS)
        )
        volatility_scale = (
            min(target_annualized_volatility / predicted_volatility, maximum_scale)
            if predicted_volatility > 0
            else 0.0
        )
        state = str(regime.loc[date, "volatility_regime"])
        regime_scale = high_volatility_scale if state == "high" else 1.0
        constraint_scale = min(gross_limit / base_gross, 1.0)
        final_scale = min(volatility_scale * regime_scale, constraint_scale)
        adjusted.loc[date] = base * final_scale

        rows.append(
            {
                "date": date,
                "status": "scaled",
                "volatility_regime": state,
                "market_realized_volatility": regime.loc[
                    date, "market_realized_volatility"
                ],
                "high_volatility_threshold": regime.loc[
                    date, "high_volatility_threshold"
                ],
                "base_predicted_annualized_volatility": predicted_volatility,
                "volatility_scale": volatility_scale,
                "regime_scale": regime_scale,
                "final_scale": final_scale,
                "final_predicted_annualized_volatility": (
                    predicted_volatility * final_scale
                ),
                "final_gross_exposure": float(adjusted.loc[date].abs().sum()),
            }
        )

    return adjusted, pd.DataFrame(rows)

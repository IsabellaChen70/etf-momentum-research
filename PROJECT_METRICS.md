# Verified project metrics

This file separates reproducible project facts from claims that the current data do not support. Values come from the saved configuration and report tables.

## Scope

- 37 exchange-traded funds across six dataset categories.
- The categories cover broad US equities, US sectors, international equities, fixed income, commodities, and alternatives.
- 14 causal features derived from momentum, risk-adjusted momentum, cross-sectional ranks, and normalized SMA or EWMA MACD.
- Four model families in the final walk-forward comparison: LASSO, Ridge, Elastic Net, and histogram gradient boosting.

## Portfolio controls

- One-day execution lag with five return-day overlapping cohorts.
- Market-neutral target weights with a 200% gross-exposure limit.
- Absolute position size limited to 15% per ETF.
- Final backtests include 5 basis points of transaction cost and 25 basis points of annual short-borrow cost.

## Validation-only risk overlay

A pre-specified overlay targets 10% annualized volatility and cuts exposure by half when cross-asset realized volatility exceeds its causal 80th-percentile threshold.

| Validation metric | Base MOMRA reversal | Risk overlay | Change |
| --- | ---: | ---: | ---: |
| Annualized volatility | 15.7% | 8.4% | 46.5% reduction |
| Maximum drawdown magnitude | 7.5% | 6.7% | 11.7% reduction |
| Annualized return | 10.6% | 2.2% | Lower after scaling |
| Sharpe | 0.716 | 0.302 | Lower after scaling |

The overlay reduced validation risk, but it also reduced return and Sharpe. No overlay result is reported for the previously inspected historical test period. A later locked holdout is required to evaluate this extension.

## Historical evidence

The final validation-selected MOMRA reversal strategy returned -7.0% annualized in the June 2025 through June 2026 historical test, with a -0.328 Sharpe ratio and -20.1% maximum drawdown. This period has already been inspected and is not a fresh holdout.

## Scope boundaries

The current instruments are ETFs rather than futures contracts. The feature set does not include VIX-implied volatility. Claims about futures implementation, VIX data, or positive historical out-of-sample performance would require additional evidence.

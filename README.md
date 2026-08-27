# ETF Momentum Research

This repository documents a personal research exercise on cross-sectional momentum across 37 exchange-traded funds. The analysis begins with a review of the price panel, then compares two portfolio allocation rules over the period from January 2020 through June 2026.

## Research design

Momentum is the mean of the prior 20 close-to-close daily returns. A signal observed at close `t` enters at close `t+1`. Each position cohort earns the following five daily returns. Five equal-capital cohorts overlap once the backtest reaches steady state.

Both portfolios target 150% long exposure and 50% short exposure. Transaction costs, financing costs, taxes, and distributions are excluded. The estimates therefore describe this sample and implementation rather than an expected live return.

| Portfolio | Allocation rule |
| --- | --- |
| Equal sign groups | Assets with positive momentum share the long side equally. Assets with negative momentum share the short side equally. |
| Relative rank groups | The upper half of the daily momentum ranking shares the long side equally. The lower half shares the short side equally. |

## Data treatment

The source panel contains daily OHLCV observations. Supplied adjusted files replace USO and UNG. Documented 2-for-1 splits in XLB, XLE, XLK, XLU, and XLY are corrected in the scripts. Returns remain price returns because distributions are outside the dataset.

The input files are excluded from GitHub because their redistribution terms were not established. See [data/README.md](data/README.md) for the expected local paths.

## Backtest results

| Portfolio | Ending value | Annualized return | Annualized volatility | Return / volatility | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Equal sign groups | 1.545 | 7.0% | 17.7% | 0.39 | -32.8% |
| Relative rank groups | 1.892 | 10.4% | 15.4% | 0.67 | -22.8% |

The relative-rank allocation had the stronger result in this sample. It produced a higher ending value and a smaller drawdown. The comparison is in-sample and does not account for trading costs, so it provides limited evidence about performance outside this period.

![Momentum portfolio comparison](results/momentum/portfolio_comparison.png)

## Reproduction

Create an environment with Python 3.11 or later and install the dependencies:

```bash
python3 -m pip install -r requirements.txt
```

After placing the local inputs under `data/`, run:

```bash
python3 src/explore_etf_data.py
python3 src/momentum_backtest.py
```

Derived tables and figures are written under `results/`.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/explore_etf_data.py` | Data checks and descriptive plots |
| `src/momentum_backtest.py` | Signal construction and cohort backtester |
| `results/exploration/` | Descriptive tables and figures |
| `results/momentum/` | Portfolio comparison and performance table |

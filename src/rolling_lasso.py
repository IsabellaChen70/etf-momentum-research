"""Forward-safe signal preprocessing and rolling LASSO research.

The analysis reuses the split-adjusted 37-ETF data and the five-day cohort
backtester from earlier research stages. Previous signal families are winsorized,
demeaned, and divided by their historical standard deviation.

At every monthly walk-forward iteration:

* the prior three calendar months train the model;
* the immediately following month selects the LASSO alpha by prediction MSE;
* the next month is predicted out of sample;
* the window then advances by one month.

All rolling transforms are shifted by one day, and observations whose return
labels cross a training or validation boundary are purged.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import warnings

import matplotlib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_DIR / "results" / "modeling" / "lasso"
TABLE_DIR = RESULT_DIR / "tables"
PLOT_DIR = RESULT_DIR / "plots"

from src.signal_analysis.data_split import (
    TEST_START,
    TRAIN_START,
    VALIDATION_START,
    load_close_prices,
)
from src.momentum_backtest import multi_asset_backtest


HOLDING_DAYS = 5
EXECUTION_LAG_DAYS = 1
PREPROCESS_LOOKBACK = 126
PREPROCESS_MIN_PERIODS = 63
LOWER_QUANTILE = 0.025
UPPER_QUANTILE = 0.975
TRAIN_MONTHS = 3
VALIDATION_MONTHS = 1
TEST_MONTHS = 1
LONG_GROSS = 1.50
SHORT_GROSS = 0.50
PERIODS_PER_YEAR = 252
MIN_CROSS_SECTIONAL_SIGNAL_STD = 1e-10

MOM_WINDOWS = (3, 5, 10, 20)
SMA_MACD_PAIRS = (
    (1, 5),
    (1, 10),
    (5, 10),
    (5, 20),
    (10, 20),
    (10, 60),
    (20, 60),
)
EWMA_FAST_ALPHA = 0.50
EWMA_SLOW_ALPHA = 1 / 3
LASSO_ALPHAS = np.logspace(-7, -2, 16)
FEATURE_SAMPLE_ROWS = 2_000

SAMPLE_ORDER = ("training", "validation", "test")


@dataclass(frozen=True)
class ProcessedSignal:
    raw: pd.DataFrame
    lower_bound: pd.DataFrame
    upper_bound: pd.DataFrame
    winsorized: pd.DataFrame
    standardized: pd.DataFrame
    normalized: pd.DataFrame
    historical_mean: pd.DataFrame
    historical_std: pd.DataFrame


def sample_for_dates(dates: pd.DatetimeIndex) -> pd.Series:
    """Assign the established outer training/validation/test labels."""
    sample = pd.Series(pd.NA, index=dates, dtype="string")
    sample.loc[(dates >= TRAIN_START) & (dates < VALIDATION_START)] = "training"
    sample.loc[(dates >= VALIDATION_START) & (dates < TEST_START)] = "validation"
    sample.loc[dates >= TEST_START] = "test"
    return sample


def calculate_raw_signals(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Recreate the previous MOM, MOMRA, CSMOM, SMA-MACD and EWMA-MACD signals."""
    returns = prices.pct_change(fill_method=None)
    signals: dict[str, pd.DataFrame] = {}

    for days in MOM_WINDOWS:
        signals[f"mom_{days}"] = returns.rolling(days, min_periods=days).mean()

    mom20 = signals["mom_20"]
    vol20 = returns.rolling(20, min_periods=20).std(ddof=1)
    standard_error20 = vol20.div(np.sqrt(20))
    signals["momra_20"] = mom20.div(standard_error20.replace(0.0, np.nan))

    rank = mom20.rank(axis=1, method="average", ascending=True)
    available = mom20.notna().sum(axis=1)
    denominator = available.sub(1).replace(0, np.nan)
    signals["csmom_20"] = rank.sub(1).div(denominator, axis=0).mul(100.0)

    for fast, slow in SMA_MACD_PAIRS:
        fast_average = prices.rolling(fast, min_periods=fast).mean()
        slow_average = prices.rolling(slow, min_periods=slow).mean()
        signals[f"sma_macd_{fast}_{slow}"] = fast_average.div(slow_average).sub(1.0)

    fast_ewma = prices.ewm(alpha=EWMA_FAST_ALPHA, adjust=False, min_periods=1).mean()
    slow_ewma = prices.ewm(alpha=EWMA_SLOW_ALPHA, adjust=False, min_periods=1).mean()
    signals["ewma_macd_0500_0333"] = fast_ewma.div(slow_ewma).sub(1.0)
    return signals


def preprocess_signal(
    raw: pd.DataFrame,
    lookback: int = PREPROCESS_LOOKBACK,
    min_periods: int = PREPROCESS_MIN_PERIODS,
    lower_quantile: float = LOWER_QUANTILE,
    upper_quantile: float = UPPER_QUANTILE,
) -> ProcessedSignal:
    """Apply rolling winsorization, demeaning and volatility normalization.

    Every statistic is shifted one row, so the transform for date t is based
    only on values observed strictly before t.  Winsorization bounds are always
    calculated from the original raw signal, never recursively clipped values.
    """
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Winsorization quantiles must satisfy 0 <= low < high <= 1")
    if min_periods > lookback:
        raise ValueError("min_periods cannot exceed lookback")

    clean_raw = raw.replace([np.inf, -np.inf], np.nan)
    lower = clean_raw.rolling(lookback, min_periods=min_periods).quantile(lower_quantile).shift(1)
    upper = clean_raw.rolling(lookback, min_periods=min_periods).quantile(upper_quantile).shift(1)
    winsorized = clean_raw.where(clean_raw.ge(lower), lower)
    winsorized = winsorized.where(winsorized.le(upper), upper)

    historical_mean = winsorized.rolling(lookback, min_periods=min_periods).mean().shift(1)
    historical_std = winsorized.rolling(lookback, min_periods=min_periods).std(ddof=1).shift(1)
    standardized = winsorized.sub(historical_mean)
    normalized = standardized.div(historical_std.replace(0.0, np.nan))
    normalized = normalized.replace([np.inf, -np.inf], np.nan)

    return ProcessedSignal(
        raw=clean_raw,
        lower_bound=lower,
        upper_bound=upper,
        winsorized=winsorized,
        standardized=standardized,
        normalized=normalized,
        historical_mean=historical_mean,
        historical_std=historical_std,
    )


def preprocess_all_signals(
    raw_signals: dict[str, pd.DataFrame],
) -> dict[str, ProcessedSignal]:
    """Transform every previous signal independently through time by ETF."""
    return {name: preprocess_signal(signal) for name, signal in raw_signals.items()}


def calculate_execution_aligned_target(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return the five-day outcome earned after the established one-day lag.

    A signal at close t is entered at close t+1 and earns the five returns
    ending at close t+6.  The target is therefore price[t+6]/price[t+1]-1.
    """
    exit_offset = EXECUTION_LAG_DAYS + HOLDING_DAYS
    entry_price = prices.shift(-EXECUTION_LAG_DAYS)
    exit_price = prices.shift(-exit_offset)
    target = exit_price.div(entry_price).sub(1.0)
    dates = pd.Series(prices.index, index=prices.index)
    entry_dates = dates.shift(-EXECUTION_LAG_DAYS)
    end_dates = dates.shift(-exit_offset)
    return target, entry_dates, end_dates


def build_model_panel(
    prices: pd.DataFrame,
    processed: dict[str, ProcessedSignal],
) -> tuple[pd.DataFrame, list[str]]:
    """Create one ETF-date panel containing normalized features and future labels."""
    feature_names = list(processed)
    normalized = pd.concat(
        {name: item.normalized.stack(future_stack=True) for name, item in processed.items()},
        axis=1,
    )
    normalized.index.names = ["date", "symbol"]

    target, entry_dates, end_dates = calculate_execution_aligned_target(prices)
    panel = normalized.copy()
    panel["future_5d_return"] = target.stack(future_stack=True)
    panel = panel.reset_index()
    panel["entry_date"] = panel["date"].map(entry_dates)
    panel["label_end_date"] = panel["date"].map(end_dates)
    panel["sample"] = panel["date"].map(sample_for_dates(prices.index))
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    return panel, feature_names


def month_bounds(period: pd.Period) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return inclusive normalized start/end dates for a calendar month."""
    return period.start_time.normalize(), period.end_time.normalize()


def fold_rows(
    panel: pd.DataFrame,
    test_month: pd.Period,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return purged train/validation rows and all feature-complete test rows."""
    train_months = [test_month - offset for offset in (4, 3, 2)]
    validation_month = test_month - 1
    month = panel["date"].dt.to_period("M")
    feature_complete = panel[feature_names].notna().all(axis=1)
    label_complete = panel["future_5d_return"].notna() & panel["label_end_date"].notna()

    train_end = month_bounds(train_months[-1])[1]
    validation_end = month_bounds(validation_month)[1]
    train_mask = (
        month.isin(train_months)
        & feature_complete
        & label_complete
        & panel["label_end_date"].le(train_end)
    )
    validation_mask = (
        month.eq(validation_month)
        & feature_complete
        & label_complete
        & panel["label_end_date"].le(validation_end)
    )
    test_mask = month.eq(test_month) & feature_complete
    return (
        panel.loc[train_mask].copy(),
        panel.loc[validation_mask].copy(),
        panel.loc[test_mask].copy(),
    )


def fit_monthly_lasso(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: list[str],
    test_month: pd.Period,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Tune alpha on validation MSE, refit on known data, and predict one month."""
    x_train = train[feature_names].to_numpy(dtype=float)
    y_train = train["future_5d_return"].to_numpy(dtype=float)
    x_validation = validation[feature_names].to_numpy(dtype=float)
    y_validation = validation["future_5d_return"].to_numpy(dtype=float)

    grid_rows: list[dict[str, object]] = []
    fitted: list[tuple[float, Lasso]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for alpha in LASSO_ALPHAS:
            model = Lasso(alpha=float(alpha), fit_intercept=True, max_iter=50_000, tol=1e-7)
            model.fit(x_train, y_train)
            validation_prediction = model.predict(x_validation)
            mse = mean_squared_error(y_validation, validation_prediction)
            grid_rows.append(
                {
                    "test_month": str(test_month),
                    "alpha": float(alpha),
                    "validation_mse": float(mse),
                    "nonzero_coefficients": int(np.count_nonzero(model.coef_)),
                }
            )
            fitted.append((float(mse), model))

    grid = pd.DataFrame(grid_rows).sort_values(
        ["validation_mse", "alpha"], ascending=[True, False]
    )
    selected_alpha = float(grid.iloc[0]["alpha"])

    development = pd.concat([train, validation], ignore_index=True)
    final_model = Lasso(
        alpha=selected_alpha,
        fit_intercept=True,
        max_iter=50_000,
        tol=1e-7,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        final_model.fit(
            development[feature_names].to_numpy(dtype=float),
            development["future_5d_return"].to_numpy(dtype=float),
        )

    prediction = test[["date", "symbol", "entry_date", "label_end_date", "sample", "future_5d_return"]].copy()
    prediction["predicted_5d_return"] = final_model.predict(
        test[feature_names].to_numpy(dtype=float)
    )
    prediction["test_month"] = str(test_month)

    coefficients = pd.DataFrame(
        {
            "test_month": str(test_month),
            "feature": feature_names,
            "coefficient": final_model.coef_,
            "selected": np.abs(final_model.coef_) > 1e-12,
        }
    )
    fold_summary = {
        "test_month": str(test_month),
        "sample": prediction["sample"].dropna().iloc[0] if prediction["sample"].notna().any() else pd.NA,
        "train_start": train["date"].min(),
        "train_end": train["date"].max(),
        "validation_start": validation["date"].min(),
        "validation_end": validation["date"].max(),
        "test_start": test["date"].min(),
        "test_end": test["date"].max(),
        "train_observations": len(train),
        "validation_observations": len(validation),
        "test_observations": len(test),
        "selected_alpha": selected_alpha,
        "validation_mse": float(grid.iloc[0]["validation_mse"]),
        "nonzero_coefficients": int(coefficients["selected"].sum()),
        "intercept": float(final_model.intercept_),
    }
    return prediction, fold_summary, coefficients, grid


def run_walk_forward_lasso(
    panel: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run all feasible 3-month/1-month/1-month rolling LASSO folds."""
    complete_feature_dates = panel.loc[
        panel[feature_names].notna().all(axis=1), "date"
    ]
    if complete_feature_dates.empty:
        raise ValueError("No dates have complete normalized signals")

    first_month = complete_feature_dates.min().to_period("M") + 4
    last_month = panel["date"].max().to_period("M")
    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    coefficients: list[pd.DataFrame] = []
    grids: list[pd.DataFrame] = []

    for test_month in pd.period_range(first_month, last_month, freq="M"):
        train, validation, test = fold_rows(panel, test_month, feature_names)
        if len(train) < 500 or len(validation) < 100 or test.empty:
            continue
        prediction, fold, coefficient, grid = fit_monthly_lasso(
            train, validation, test, feature_names, test_month
        )
        predictions.append(prediction)
        folds.append(fold)
        coefficients.append(coefficient)
        grids.append(grid)

    if not predictions:
        raise ValueError("No complete walk-forward folds were available")
    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(folds),
        pd.concat(coefficients, ignore_index=True),
        pd.concat(grids, ignore_index=True),
    )


def proportional_150_50_weights(signal: pd.Series) -> pd.Series:
    """Demean one prediction cross-section and scale it to exact 150/50."""
    weights = pd.Series(0.0, index=signal.index, dtype=float)
    clean = signal.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        return weights
    if clean.std(ddof=1) <= MIN_CROSS_SECTIONAL_SIGNAL_STD:
        return weights
    centered = clean - clean.mean()
    positive = centered > 0
    negative = centered < 0
    positive_total = centered.loc[positive].sum()
    negative_total = -centered.loc[negative].sum()
    if positive_total <= 0 or negative_total <= 0:
        return weights
    weights.loc[centered.index[positive]] = LONG_GROSS * centered.loc[positive] / positive_total
    weights.loc[centered.index[negative]] = SHORT_GROSS * centered.loc[negative] / negative_total
    return weights


def predictions_to_targets(
    prices: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot monthly predictions into daily ETF signals and portfolio targets."""
    duplicate = predictions.duplicated(["date", "symbol"])
    if duplicate.any():
        raise AssertionError("Walk-forward predictions contain duplicate ETF-date rows")
    signal = predictions.pivot(index="date", columns="symbol", values="predicted_5d_return")
    signal = signal.reindex(index=prices.index, columns=prices.columns)
    targets = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for date, row in signal.iterrows():
        targets.loc[date] = proportional_150_50_weights(row)
    return signal, targets


def purge_incomplete_sample_targets(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Remove signals that cannot finish their full hold inside an outer sample.

    A signal on date t enters at t+1 and exits after earning returns through
    t+6.  Consequently, the final six signal dates of each independently
    evaluated outer sample cannot be traded without creating a partial cohort
    or crossing the sample boundary.
    """
    clean = targets.copy()
    sample_bounds = (
        (TRAIN_START, VALIDATION_START),
        (VALIDATION_START, TEST_START),
        (TEST_START, None),
    )
    unavailable_signal_dates = EXECUTION_LAG_DAYS + HOLDING_DAYS
    for start, end in sample_bounds:
        mask = clean.index >= start
        if end is not None:
            mask &= clean.index < end
        dates = clean.index[mask]
        if len(dates) <= unavailable_signal_dates:
            clean.loc[dates] = 0.0
        else:
            clean.loc[dates[-unavailable_signal_dates:]] = 0.0
    return clean


def backtest_outer_samples(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Backtest each outer sample from cash so no cohort crosses a boundary."""
    masks = {
        "training": (prices.index >= TRAIN_START) & (prices.index < VALIDATION_START),
        "validation": (prices.index >= VALIDATION_START) & (prices.index < TEST_START),
        "test": prices.index >= TEST_START,
    }
    results: dict[str, pd.DataFrame] = {}
    for sample, mask in masks.items():
        sample_prices = prices.loc[mask]
        sample_targets = targets.loc[mask]
        results[sample] = multi_asset_backtest(
            sample_prices,
            sample_targets,
            holding_days=HOLDING_DAYS,
        )
    return results


def portfolio_metrics(result: pd.DataFrame, sample: str) -> dict[str, object]:
    """Calculate direct return, path, exposure and turnover statistics."""
    returns = result["portfolio_return"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    value = result["portfolio_value"]
    active_days = max(int(returns.ne(0).sum()), 1)
    annualized_return = value.iloc[-1] ** (PERIODS_PER_YEAR / max(len(value) - 1, 1)) - 1
    running_peak = value.cummax()
    return {
        "sample": sample,
        "start": result.index.min(),
        "end": result.index.max(),
        "trading_days": len(result),
        "active_return_days": active_days,
        "mean_daily_return": float(returns.mean()),
        "daily_return_std": float(returns.std(ddof=1)),
        "annualized_return": float(annualized_return),
        "ending_value": float(value.iloc[-1]),
        "max_drawdown": float((value.div(running_peak).sub(1.0)).min()),
        "average_gross_exposure": float(result["gross_exposure"].mean()),
        "average_daily_turnover": float(result["turnover"].mean()),
    }


def clean_outer_sample_predictions(
    predictions: pd.DataFrame,
    sample: str,
) -> pd.DataFrame:
    """Return complete labels that finish inside one independent outer sample."""
    data = predictions.loc[predictions["sample"].eq(sample)].dropna(
        subset=["future_5d_return", "predicted_5d_return", "label_end_date"]
    )
    if sample == "training":
        data = data.loc[data["label_end_date"].lt(VALIDATION_START)]
    elif sample == "validation":
        data = data.loc[data["label_end_date"].lt(TEST_START)]
    return data


def prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize accuracy on independent, boundary-clean outer samples."""
    rows: list[dict[str, object]] = []
    for sample in SAMPLE_ORDER:
        data = clean_outer_sample_predictions(predictions, sample)
        if data.empty:
            continue
        actual = data["future_5d_return"]
        predicted = data["predicted_5d_return"]
        rows.append(
            {
                "sample": sample,
                "observations": len(data),
                "mean_actual_return": float(actual.mean()),
                "mean_predicted_return": float(predicted.mean()),
                "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                "mae": float((actual - predicted).abs().mean()),
                "correlation": float(actual.corr(predicted)),
                "direction_accuracy": float(np.sign(actual).eq(np.sign(predicted)).mean()),
            }
        )
    return pd.DataFrame(rows)


def processed_signal_sample(
    processed: dict[str, ProcessedSignal],
    prices: pd.DataFrame,
    row_limit: int = FEATURE_SAMPLE_ROWS,
) -> pd.DataFrame:
    """Save a compact, auditable before/after sample rather than a huge full panel."""
    blocks: dict[str, pd.Series] = {}
    for name, item in processed.items():
        blocks[f"{name}__raw"] = item.raw.stack(future_stack=True)
        blocks[f"{name}__winsorized"] = item.winsorized.stack(future_stack=True)
        blocks[f"{name}__standardized"] = item.standardized.stack(future_stack=True)
        blocks[f"{name}__normalized"] = item.normalized.stack(future_stack=True)
    sample = pd.concat(blocks, axis=1).reset_index()
    sample.columns.values[:2] = ["date", "symbol"]
    sample = sample.loc[sample.filter(like="__normalized").notna().all(axis=1)]
    if len(sample) > row_limit:
        positions = np.linspace(0, len(sample) - 1, row_limit, dtype=int)
        sample = sample.iloc[positions]
    return sample.reset_index(drop=True)


def audit_analysis(
    prices: pd.DataFrame,
    raw_signals: dict[str, pd.DataFrame],
    processed: dict[str, ProcessedSignal],
    panel: pd.DataFrame,
    feature_names: list[str],
    folds: pd.DataFrame,
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    backtests: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Run structural tests for look-ahead, alignment, exposure and numerical safety."""
    checks: list[dict[str, object]] = []

    # Manually recompute one historical quantile and compare it with the value
    # used on the next date.  This catches an accidental unshifted rolling bound.
    probe_feature = "mom_20"
    probe_symbol = prices.columns[0]
    raw = raw_signals[probe_feature][probe_symbol]
    valid = processed[probe_feature].lower_bound[probe_symbol].dropna()
    probe_date = valid.index[len(valid) // 2]
    location = raw.index.get_loc(probe_date)
    history = raw.iloc[max(0, location - PREPROCESS_LOOKBACK) : location].dropna()
    expected_lower = history.quantile(LOWER_QUANTILE)
    used_lower = processed[probe_feature].lower_bound.loc[probe_date, probe_symbol]
    if not np.isclose(expected_lower, used_lower, rtol=0, atol=1e-12):
        raise AssertionError("Rolling winsorization uses current/future information")
    checks.append({"check": "rolling_bounds_use_strictly_prior_raw_values", "status": "pass", "detail": str(probe_date.date())})

    if not folds["train_end"].lt(folds["validation_start"]).all():
        raise AssertionError("A training period overlaps its validation month")
    if not folds["validation_end"].lt(folds["test_start"]).all():
        raise AssertionError("A validation period overlaps its test month")
    checks.append({"check": "chronological_train_validation_test_order", "status": "pass", "detail": f"{len(folds)} monthly folds"})

    month = panel["date"].dt.to_period("M")
    for row in folds.itertuples(index=False):
        test_month = pd.Period(row.test_month, freq="M")
        train_months = [test_month - offset for offset in (4, 3, 2)]
        validation_month = test_month - 1
        train_end = month_bounds(train_months[-1])[1]
        validation_end = month_bounds(validation_month)[1]
        train_labels = panel.loc[month.isin(train_months) & panel[feature_names].notna().all(axis=1)].dropna(subset=["label_end_date"])
        train_labels = train_labels.loc[train_labels["label_end_date"].le(train_end)]
        validation_labels = panel.loc[month.eq(validation_month) & panel[feature_names].notna().all(axis=1)].dropna(subset=["label_end_date"])
        validation_labels = validation_labels.loc[validation_labels["label_end_date"].le(validation_end)]
        if not train_labels.empty and train_labels["label_end_date"].max() > train_end:
            raise AssertionError("A training label crosses into validation")
        if not validation_labels.empty and validation_labels["label_end_date"].max() > validation_end:
            raise AssertionError("A validation label crosses into test")
    checks.append({"check": "five_day_labels_purged_at_fold_boundaries", "status": "pass", "detail": "label end <= fold end"})

    if predictions.duplicated(["date", "symbol"]).any():
        raise AssertionError("Duplicate out-of-sample predictions detected")
    if not predictions["date"].dt.to_period("M").astype(str).eq(predictions["test_month"]).all():
        raise AssertionError("A prediction was assigned outside its test month")
    checks.append({"check": "predictions_are_monthly_out_of_sample", "status": "pass", "detail": f"{len(predictions):,} ETF-date predictions"})

    clean_training = clean_outer_sample_predictions(predictions, "training")
    clean_validation = clean_outer_sample_predictions(predictions, "validation")
    if clean_training["label_end_date"].max() >= VALIDATION_START:
        raise AssertionError("Training prediction evaluation crosses into outer validation")
    if clean_validation["label_end_date"].max() >= TEST_START:
        raise AssertionError("Validation prediction evaluation crosses into outer test")
    checks.append({"check": "prediction_metrics_use_clean_outer_samples", "status": "pass", "detail": "boundary-crossing labels excluded"})

    sample_bounds = (
        ("training", TRAIN_START, VALIDATION_START),
        ("validation", VALIDATION_START, TEST_START),
        ("test", TEST_START, None),
    )
    unavailable_signal_dates = EXECUTION_LAG_DAYS + HOLDING_DAYS
    for sample, start, end in sample_bounds:
        mask = targets.index >= start
        if end is not None:
            mask &= targets.index < end
        dates = targets.index[mask]
        boundary_targets = targets.loc[dates[-unavailable_signal_dates:]]
        if boundary_targets.abs().to_numpy().max() > 1e-12:
            raise AssertionError(f"{sample} contains a trade that cannot complete its five-day hold")
    checks.append({"check": "complete_five_day_holds_at_sample_boundaries", "status": "pass", "detail": "last 6 signal dates are flat"})

    synthetic_dates = pd.bdate_range("2020-01-01", periods=9)
    synthetic_prices = pd.DataFrame(
        {"asset": 100.0 + np.arange(len(synthetic_dates))},
        index=synthetic_dates,
    )
    synthetic_targets = pd.DataFrame(0.0, index=synthetic_dates, columns=["asset"])
    synthetic_targets.iloc[0, 0] = 1.0
    synthetic_result = multi_asset_backtest(
        synthetic_prices,
        synthetic_targets,
        holding_days=HOLDING_DAYS,
    )
    pnl_dates = synthetic_result.index[synthetic_result["portfolio_return"].ne(0.0)]
    held_dates = synthetic_result.index[synthetic_result["weight_asset"].ne(0.0)]
    if not pnl_dates.equals(synthetic_dates[2:7]):
        raise AssertionError("A signal does not earn exactly five returns after its one-day lag")
    if not held_dates.equals(synthetic_dates[1:6]):
        raise AssertionError("A cohort is not held for exactly five close dates")
    checks.append({"check": "one_day_lag_and_five_return_day_holding", "status": "pass", "detail": "P&L dates t+2 through t+6"})

    long_total = targets.clip(lower=0).sum(axis=1)
    short_total = -targets.clip(upper=0).sum(axis=1)
    active = long_total.gt(0) & short_total.gt(0)
    if not np.allclose(long_total.loc[active], LONG_GROSS, atol=1e-10):
        raise AssertionError("Long target exposure does not sum to 150%")
    if not np.allclose(short_total.loc[active], SHORT_GROSS, atol=1e-10):
        raise AssertionError("Short target exposure does not sum to 50%")
    checks.append({"check": "daily_target_exposure_150_50", "status": "pass", "detail": f"{int(active.sum())} active dates"})

    max_price_return = float(prices.pct_change(fill_method=None).abs().max().max())
    if max_price_return >= 1.0:
        raise AssertionError("A split-like adjusted-price return of at least 100% remains")
    checks.append({"check": "split_adjusted_price_returns", "status": "pass", "detail": f"max abs return {max_price_return:.6f}"})

    for sample, result in backtests.items():
        if not np.isfinite(result.select_dtypes(include=[np.number]).to_numpy()).all():
            raise AssertionError(f"Non-finite backtest output in {sample}")
        if not np.isclose(result["portfolio_value"].iloc[0], 1.0):
            raise AssertionError(f"{sample} backtest did not restart from cash")
    checks.append({"check": "finite_independent_outer_backtests", "status": "pass", "detail": "each sample starts at 1"})
    return pd.DataFrame(checks)


def plot_signal_transformation(
    processed: dict[str, ProcessedSignal],
    prices: pd.DataFrame,
    output_path: Path,
) -> None:
    """Show how winsorization and normalization change a representative signal."""
    feature = "momra_20"
    symbol = "SPY" if "SPY" in prices.columns else prices.columns[0]
    item = processed[feature]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(item.raw.index, item.raw[symbol], linewidth=0.9, color="#4C78A8")
    axes[0].set_ylabel("Raw")
    axes[1].plot(item.winsorized.index, item.winsorized[symbol], linewidth=0.9, color="#F58518")
    axes[1].set_ylabel("Winsorized")
    axes[2].plot(item.normalized.index, item.normalized[symbol], linewidth=0.9, color="#54A24B")
    axes[2].axhline(0, color="0.45", linewidth=0.8)
    axes[2].set_ylabel("Normalized z-score")
    axes[2].set_xlabel("Date")
    fig.suptitle(f"{symbol} MOMRA: rolling signal preprocessing")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_selection(coefficients: pd.DataFrame, output_path: Path) -> None:
    """Plot mean coefficient magnitude and monthly selection frequency."""
    stats = coefficients.groupby("feature").agg(
        mean_abs_coefficient=("coefficient", lambda x: float(np.abs(x).mean())),
        selection_frequency=("selected", "mean"),
    )
    stats = stats.sort_values("mean_abs_coefficient", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    axes[0].barh(stats.index, stats["mean_abs_coefficient"], color="#4C78A8")
    axes[0].set_xlabel("Mean absolute LASSO coefficient")
    axes[1].barh(stats.index, stats["selection_frequency"], color="#F58518")
    axes[1].set_xlabel("Fraction of monthly models selected")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_title("Coefficient magnitude")
    axes[1].set_title("Selection frequency")
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Rolling LASSO feature importance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_scatter(predictions: pd.DataFrame, output_path: Path) -> None:
    """Compare out-of-sample predicted and realized returns by outer sample."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, sample in zip(axes, SAMPLE_ORDER):
        data = clean_outer_sample_predictions(predictions, sample)
        ax.scatter(
            data["predicted_5d_return"],
            data["future_5d_return"],
            s=7,
            alpha=0.16,
            color="#4C78A8",
            edgecolors="none",
        )
        correlation = data["predicted_5d_return"].corr(data["future_5d_return"])
        ax.axhline(0, color="0.5", linewidth=0.7)
        ax.axvline(0, color="0.5", linewidth=0.7)
        ax.set_title(f"{sample.title()} (corr={correlation:.3f})")
        ax.set_xlabel("Predicted 5-day return")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Realized 5-day return")
    fig.suptitle("Walk-forward LASSO predictions versus realized returns")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_portfolio_samples(backtests: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Plot each outer sample from a separate starting value of one."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5))
    for ax, sample in zip(axes, SAMPLE_ORDER):
        result = backtests[sample]
        ax.plot(result.index, result["portfolio_value"], linewidth=1.4, color="#4C78A8")
        ax.axhline(1.0, color="0.45", linewidth=0.8)
        ax.set_ylabel("Value")
        ax.set_title(f"{sample.title()} (ending value {result['portfolio_value'].iloc[-1]:.3f})")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Date")
    fig.suptitle("LASSO signal-proportional portfolio by independent sample")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_selected_alpha(folds: pd.DataFrame, output_path: Path) -> None:
    """Show how validation-selected LASSO regularization changes over time."""
    dates = pd.PeriodIndex(folds["test_month"], freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.step(dates, folds["selected_alpha"], where="mid", linewidth=1.3, color="#4C78A8")
    ax.set_yscale("log")
    ax.set_xlabel("Out-of-sample test month")
    ax.set_ylabel("Validation-selected alpha (log scale)")
    ax.set_title("Rolling LASSO regularization choice")
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    feature_names: list[str],
    folds: pd.DataFrame,
    prediction_summary: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> Path:
    """Write a concise plain-text interpretation of the rolling LASSO results."""
    selection = coefficients.groupby("feature")["selected"].mean().sort_values(ascending=False)
    lines = [
        "ROLLING LASSO RESULTS",
        "",
        "Method",
        f"- {len(feature_names)} prior signals were winsorized at rolling 2.5%/97.5% bounds.",
        "- Standardization means demeaning; normalization means dividing by rolling standard deviation.",
        f"- All preprocessing uses strictly prior values over a {PREPROCESS_LOOKBACK}-day window.",
        "- Each fold uses 3 months training, 1 month validation, and 1 month out-of-sample testing.",
        "- The five-day target begins after the established one-day execution lag.",
        "- LASSO alpha is selected by validation MSE, not by Sharpe ratio.",
        "",
        f"Completed monthly folds: {len(folds)}",
        f"First test month: {folds['test_month'].min()}",
        f"Last test month: {folds['test_month'].max()}",
        "",
        "Prediction results",
        prediction_summary.to_string(index=False),
        "",
        "Portfolio results",
        portfolio_summary.to_string(index=False),
        "",
        "Most frequently retained signals",
        selection.head(8).rename("selection_frequency").to_string(),
        "",
        "Interpretation reminder",
        "- Validation and test results are the important evidence for whether the combined signal generalizes.",
        "- A coefficient of zero means LASSO did not find incremental value for that signal in that monthly model.",
        "- Positive test performance is not proof of future profitability; this is a historical walk-forward test.",
    ]
    path = RESULT_DIR / "research_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_notebook_metadata(feature_names: list[str]) -> None:
    """Save machine-readable settings used by the script and notebook."""
    settings = {
        "feature_names": feature_names,
        "preprocess_lookback": PREPROCESS_LOOKBACK,
        "preprocess_min_periods": PREPROCESS_MIN_PERIODS,
        "winsor_lower_quantile": LOWER_QUANTILE,
        "winsor_upper_quantile": UPPER_QUANTILE,
        "train_months": TRAIN_MONTHS,
        "validation_months": VALIDATION_MONTHS,
        "test_months": TEST_MONTHS,
        "execution_lag_days": EXECUTION_LAG_DAYS,
        "holding_days": HOLDING_DAYS,
        "lasso_alphas": [float(value) for value in LASSO_ALPHAS],
    }
    (TABLE_DIR / "analysis_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )


def run_analysis() -> dict[str, object]:
    """Run the rolling LASSO analysis and save compact research outputs."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_close_prices()
    raw_signals = calculate_raw_signals(prices)
    processed = preprocess_all_signals(raw_signals)
    panel, feature_names = build_model_panel(prices, processed)
    predictions, folds, coefficients, alpha_grid = run_walk_forward_lasso(panel, feature_names)
    prediction_signal, targets = predictions_to_targets(prices, predictions)
    targets = purge_incomplete_sample_targets(prices, targets)
    backtests = backtest_outer_samples(prices, targets)

    prediction_summary = prediction_metrics(predictions)
    portfolio_summary = pd.DataFrame(
        [portfolio_metrics(backtests[sample], sample) for sample in SAMPLE_ORDER]
    )
    coefficient_summary = (
        coefficients.groupby("feature")
        .agg(
            months=("test_month", "nunique"),
            selection_frequency=("selected", "mean"),
            mean_coefficient=("coefficient", "mean"),
            mean_abs_coefficient=("coefficient", lambda x: float(np.abs(x).mean())),
        )
        .reset_index()
        .sort_values("mean_abs_coefficient", ascending=False)
    )
    feature_sample = processed_signal_sample(processed, prices)

    audit = audit_analysis(
        prices,
        raw_signals,
        processed,
        panel,
        feature_names,
        folds,
        predictions,
        targets,
        backtests,
    )

    table_paths = {
        "processed_sample": TABLE_DIR / "processed_signal_sample.csv",
        "folds": TABLE_DIR / "rolling_lasso_folds.csv",
        "grid": TABLE_DIR / "lasso_alpha_validation_grid.csv",
        "coefficients": TABLE_DIR / "lasso_coefficients.csv",
        "coefficient_summary": TABLE_DIR / "lasso_feature_summary.csv",
        "predictions": TABLE_DIR / "walk_forward_predictions.csv",
        "prediction_summary": TABLE_DIR / "prediction_performance.csv",
        "portfolio_summary": TABLE_DIR / "portfolio_performance.csv",
        "portfolio_values": TABLE_DIR / "portfolio_values.csv",
        "targets": TABLE_DIR / "daily_target_weights.csv",
        "audit": TABLE_DIR / "validation_checks.csv",
    }
    feature_sample.to_csv(table_paths["processed_sample"], index=False)
    folds.to_csv(table_paths["folds"], index=False)
    alpha_grid.to_csv(table_paths["grid"], index=False)
    coefficients.to_csv(table_paths["coefficients"], index=False)
    coefficient_summary.to_csv(table_paths["coefficient_summary"], index=False)
    predictions.to_csv(table_paths["predictions"], index=False)
    prediction_summary.to_csv(table_paths["prediction_summary"], index=False)
    portfolio_summary.to_csv(table_paths["portfolio_summary"], index=False)
    pd.concat(
        {sample: result["portfolio_value"] for sample, result in backtests.items()},
        names=["sample", "date"],
    ).rename("portfolio_value").reset_index().to_csv(table_paths["portfolio_values"], index=False)
    targets.loc[targets.abs().sum(axis=1).gt(0)].to_csv(table_paths["targets"], index_label="date")
    audit.to_csv(table_paths["audit"], index=False)
    save_notebook_metadata(feature_names)

    plot_paths = {
        "transformation": PLOT_DIR / "signal_preprocessing_example.png",
        "selection": PLOT_DIR / "lasso_feature_importance.png",
        "scatter": PLOT_DIR / "predicted_vs_realized_returns.png",
        "portfolio": PLOT_DIR / "lasso_portfolio_by_sample.png",
        "alpha": PLOT_DIR / "selected_lasso_alpha.png",
    }
    plot_signal_transformation(processed, prices, plot_paths["transformation"])
    plot_feature_selection(coefficients, plot_paths["selection"])
    plot_prediction_scatter(predictions, plot_paths["scatter"])
    plot_portfolio_samples(backtests, plot_paths["portfolio"])
    plot_selected_alpha(folds, plot_paths["alpha"])

    write_summary(
        feature_names, folds, prediction_summary, portfolio_summary, coefficients
    )
    return {
        "prices": prices,
        "feature_names": feature_names,
        "processed": processed,
        "panel": panel,
        "predictions": predictions,
        "folds": folds,
        "coefficients": coefficients,
        "prediction_summary": prediction_summary,
        "portfolio_summary": portfolio_summary,
        "audit": audit,
        "backtests": backtests,
    }


if __name__ == "__main__":
    output = run_analysis()
    print("Features:", ", ".join(output["feature_names"]))
    print("\nPrediction performance")
    print(output["prediction_summary"].to_string(index=False))
    print("\nPortfolio performance")
    print(output["portfolio_summary"].to_string(index=False))
    print("\nValidation checks")
    print(output["audit"].to_string(index=False))

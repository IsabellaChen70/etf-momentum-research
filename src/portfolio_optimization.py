"""Validation-selected mean-variance optimization with historical evaluation.

The expected-return signal comes from the preceding walk-forward model comparison.
Risk aversion is selected using validation Sharpe only. The selected value is
then frozen before the historical test signal and returns are evaluated.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_DIR / "results" / "optimization"
PLOT_DIR = RESULT_DIR / "plots"
TABLE_DIR = RESULT_DIR / "tables"

from src.momentum_backtest import multi_asset_backtest
from src.signal_analysis.data_split import TEST_START, VALIDATION_START, load_close_prices
from src import rolling_lasso as lasso_framework


PREDICTION_PATH = (
    PROJECT_DIR
    / "results"
    / "modeling"
    / "comparison"
    / "tables"
    / "walk_forward_model_predictions.csv"
)
MODEL_RANKING_PATH = (
    PROJECT_DIR
    / "results"
    / "modeling"
    / "comparison"
    / "tables"
    / "validation_model_ranking.csv"
)

HOLDING_DAYS = 5
EXECUTION_LAG_DAYS = 1
PERIODS_PER_YEAR = 252
COVARIANCE_LOOKBACK = 60
COVARIANCE_MIN_PERIODS = 40
COVARIANCE_SHRINKAGE = 0.10
NET_EXPOSURE = 1.00
GROSS_LIMIT = 3.00
RISK_FREE_RATE = 0.00
RISK_AVERSION_GRID = np.array(
    [0.10, 0.30, 1.00, 3.00, 10.00, 30.00, 100.00, 300.00, 1000.00],
    dtype=float,
)


def selected_model_from_validation() -> str:
    """Read the model selected using outer validation MSE."""
    ranking = pd.read_csv(MODEL_RANKING_PATH)
    selected_flag = ranking["selected_model"].astype(str).str.lower().eq("true")
    selected = ranking.loc[selected_flag]
    if len(selected) != 1:
        raise AssertionError("Expected exactly one validation-selected model")
    if not selected["sample"].eq("validation").all():
        raise AssertionError("The model winner was not selected on validation")
    return str(selected.iloc[0]["model"])


def load_prediction_signal(
    prices: pd.DataFrame,
    model: str,
    sample: str,
) -> pd.DataFrame:
    """Load one outer sample of predictions for the selected model."""
    columns = ["date", "symbol", "sample", "model", "predicted_5d_return"]
    predictions = pd.read_csv(PREDICTION_PATH, usecols=columns, parse_dates=["date"])
    predictions = predictions.loc[
        predictions["model"].eq(model) & predictions["sample"].eq(sample)
    ].copy()
    if predictions.duplicated(["date", "symbol"]).any():
        raise AssertionError(f"Duplicate {sample} ETF-date predictions")
    signal = predictions.pivot(
        index="date",
        columns="symbol",
        values="predicted_5d_return",
    )
    return signal.reindex(index=prices.index, columns=prices.columns)


def sample_dates(prices: pd.DataFrame, sample: str) -> pd.DatetimeIndex:
    """Return dates for validation or test."""
    if sample == "validation":
        return prices.index[
            (prices.index >= VALIDATION_START) & (prices.index < TEST_START)
        ]
    if sample == "test":
        return prices.index[prices.index >= TEST_START]
    raise ValueError(f"Unsupported sample: {sample}")


def covariance_for_date(
    returns: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.DataFrame:
    """Estimate a five-day covariance matrix from returns known by date."""
    window = returns.loc[:date].tail(COVARIANCE_LOOKBACK)
    if len(window) < COVARIANCE_MIN_PERIODS:
        raise ValueError(f"Insufficient covariance history at {date.date()}")
    sample_covariance = window.cov(min_periods=COVARIANCE_MIN_PERIODS)
    if sample_covariance.isna().to_numpy().any():
        raise ValueError(f"Incomplete covariance matrix at {date.date()}")

    covariance = sample_covariance.to_numpy(dtype=float) * HOLDING_DAYS
    diagonal = np.diag(np.diag(covariance))
    covariance = (
        (1.0 - COVARIANCE_SHRINKAGE) * covariance
        + COVARIANCE_SHRINKAGE * diagonal
    )
    ridge = max(float(np.median(np.diag(covariance))) * 1e-8, 1e-12)
    covariance += np.eye(len(covariance)) * ridge
    return pd.DataFrame(
        covariance,
        index=sample_covariance.index,
        columns=sample_covariance.columns,
    )


def covariance_cache(
    prices: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """Precompute historical covariance matrices for a set of signal dates."""
    returns = prices.pct_change(fill_method=None)
    return {date: covariance_for_date(returns, date) for date in dates}


def optimize_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float,
) -> tuple[pd.Series, dict[str, float]]:
    """Maximize expected return minus a risk-aversion variance penalty."""
    expected = expected_returns.reindex(covariance.index).replace(
        [np.inf, -np.inf], np.nan
    )
    if expected.isna().any():
        raise ValueError("Expected returns are incomplete")
    mu = expected.to_numpy(dtype=float)
    sigma = covariance.to_numpy(dtype=float)
    n_assets = len(mu)

    # x contains portfolio weights followed by auxiliary absolute-weight bounds.
    initial_weights = np.full(n_assets, NET_EXPOSURE / n_assets)
    initial = np.concatenate([initial_weights, np.abs(initial_weights)])

    net_row = np.concatenate([np.ones(n_assets), np.zeros(n_assets)])
    gross_row = np.concatenate([np.zeros(n_assets), np.ones(n_assets)])
    z_minus_w = np.concatenate([-np.eye(n_assets), np.eye(n_assets)], axis=1)
    z_plus_w = np.concatenate([np.eye(n_assets), np.eye(n_assets)], axis=1)
    matrix = np.vstack([net_row, gross_row, z_minus_w, z_plus_w])
    lower = np.concatenate(
        [
            [NET_EXPOSURE, -np.inf],
            np.zeros(n_assets),
            np.zeros(n_assets),
        ]
    )
    upper = np.concatenate(
        [
            [NET_EXPOSURE, GROSS_LIMIT],
            np.full(n_assets, np.inf),
            np.full(n_assets, np.inf),
        ]
    )
    net_constraint = LinearConstraint(
        matrix[:1],
        lower[:1],
        upper[:1],
    )
    inequality_constraint = LinearConstraint(
        matrix[1:],
        lower[1:],
        upper[1:],
    )
    bounds = Bounds(
        np.concatenate(
            [np.full(n_assets, -1.0), np.zeros(n_assets)]
        ),
        np.concatenate(
            [np.full(n_assets, 2.0), np.full(n_assets, 2.0)]
        ),
    )

    def objective(x: np.ndarray) -> float:
        weights = x[:n_assets]
        variance = float(weights @ sigma @ weights)
        return 0.5 * risk_aversion * variance - float(mu @ weights)

    def gradient(x: np.ndarray) -> np.ndarray:
        weights = x[:n_assets]
        weight_gradient = risk_aversion * sigma @ weights - mu
        return np.concatenate([weight_gradient, np.zeros(n_assets)])

    solution = minimize(
        objective,
        initial,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=[net_constraint, inequality_constraint],
        options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
    )
    if not solution.success:
        raise RuntimeError(f"Optimization failed: {solution.message}")

    weights = pd.Series(solution.x[:n_assets], index=expected.index, dtype=float)
    weights.loc[weights.abs() < 1e-10] = 0.0
    net = float(weights.sum())
    gross = float(weights.abs().sum())
    if not np.isclose(net, NET_EXPOSURE, atol=1e-7):
        raise AssertionError(f"Net exposure is {net:.10f}, expected {NET_EXPOSURE}")
    if gross > GROSS_LIMIT + 1e-7:
        raise AssertionError(f"Gross exposure is {gross:.10f}, limit {GROSS_LIMIT}")

    diagnostics = {
        "expected_5d_return": float(mu @ weights.to_numpy()),
        "expected_5d_volatility": float(
            np.sqrt(max(weights.to_numpy() @ sigma @ weights.to_numpy(), 0.0))
        ),
        "target_net_exposure": net,
        "target_gross_exposure": gross,
        "largest_absolute_weight": float(weights.abs().max()),
    }
    return weights, diagnostics


def build_optimized_targets(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    dates: pd.DatetimeIndex,
    covariances: dict[pd.Timestamp, pd.DataFrame],
    risk_aversion: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct daily optimized targets for one risk-aversion setting."""
    targets = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    diagnostics: list[dict[str, object]] = []
    for date in dates:
        expected = signal.loc[date]
        if expected.isna().any():
            continue
        weights, row = optimize_weights(
            expected,
            covariances[date].reindex(index=prices.columns, columns=prices.columns),
            risk_aversion,
        )
        targets.loc[date] = weights.reindex(prices.columns)
        diagnostics.append(
            {"date": date, "risk_aversion": risk_aversion, **row}
        )
    return targets, pd.DataFrame(diagnostics)


def baseline_targets(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Recreate the prior 150/50 signal-proportional target as a reference."""
    targets = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    for date in dates:
        row = signal.loc[date]
        if row.notna().sum() >= 2:
            targets.loc[date] = lasso_framework.proportional_150_50_weights(row)
    return targets


def purge_incomplete_holds(
    targets: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Remove signals that cannot complete lag plus five return-days."""
    clean = targets.copy()
    unavailable = EXECUTION_LAG_DAYS + HOLDING_DAYS
    if len(dates) <= unavailable:
        clean.loc[dates] = 0.0
    else:
        clean.loc[dates[-unavailable:]] = 0.0
    return clean


def backtest_sample(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Backtest one independent sample from starting capital of one."""
    return multi_asset_backtest(
        prices.loc[dates],
        targets.loc[dates],
        holding_days=HOLDING_DAYS,
    )


def performance_metrics(
    result: pd.DataFrame,
    sample: str,
    strategy: str,
    risk_aversion: float | None,
) -> dict[str, object]:
    """Calculate return and risk statistics from one backtest."""
    daily = result["portfolio_return"].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    volatility = float(daily.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
    sharpe = (
        float(
            (daily.mean() * PERIODS_PER_YEAR - RISK_FREE_RATE)
            / volatility
        )
        if volatility > 0
        else np.nan
    )
    value = result["portfolio_value"]
    annualized_return = float(
        value.iloc[-1] ** (PERIODS_PER_YEAR / max(len(value) - 1, 1)) - 1.0
    )
    drawdown = value.div(value.cummax()).sub(1.0)
    return {
        "sample": sample,
        "strategy": strategy,
        "risk_aversion": risk_aversion,
        "start": result.index.min(),
        "end": result.index.max(),
        "trading_days": len(result),
        "annualized_return": annualized_return,
        "annualized_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "ending_value": float(value.iloc[-1]),
        "average_gross_exposure": float(result["gross_exposure"].mean()),
        "average_net_exposure": float(result["net_exposure"].mean()),
        "average_daily_turnover": float(result["turnover"].mean()),
    }


def plot_validation_selection(
    search: pd.DataFrame,
    selected_risk_aversion: float,
    output: Path,
) -> None:
    """Plot the validation evidence used to select risk aversion."""
    ordered = search.sort_values("risk_aversion")
    selected = ordered["risk_aversion"].eq(selected_risk_aversion)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].plot(
        ordered["risk_aversion"],
        ordered["sharpe"],
        marker="o",
        linewidth=1.5,
    )
    axes[0].scatter(
        ordered.loc[selected, "risk_aversion"],
        ordered.loc[selected, "sharpe"],
        color="#D62728",
        s=90,
        zorder=3,
        label="Selected",
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Risk-aversion score, log scale")
    axes[0].set_ylabel("Validation Sharpe")
    axes[0].set_title("Selection criterion")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    points = axes[1].scatter(
        ordered["annualized_volatility"],
        ordered["annualized_return"],
        c=np.log10(ordered["risk_aversion"]),
        cmap="viridis",
        s=65,
    )
    axes[1].scatter(
        ordered.loc[selected, "annualized_volatility"],
        ordered.loc[selected, "annualized_return"],
        color="#D62728",
        marker="*",
        s=180,
        zorder=3,
    )
    for row in ordered.itertuples(index=False):
        axes[1].annotate(
            f"{row.risk_aversion:g}",
            (row.annualized_volatility, row.annualized_return),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set_xlabel("Validation annualized volatility")
    axes[1].set_ylabel("Validation annualized return")
    axes[1].set_title("Risk-return tradeoff")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].grid(alpha=0.25)
    fig.colorbar(points, ax=axes[1], label="log10 risk aversion")

    fig.suptitle("Validation-only risk-aversion selection")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_test_results(
    optimized: pd.DataFrame,
    baseline: pd.DataFrame,
    selected_risk_aversion: float,
    output: Path,
) -> None:
    """Plot the final out-of-sample comparison after selection is frozen."""
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(
        optimized.index,
        optimized["portfolio_value"],
        linewidth=1.8,
        label=f"Mean-variance optimized, risk aversion {selected_risk_aversion:g}",
    )
    ax.plot(
        baseline.index,
        baseline["portfolio_value"],
        linewidth=1.5,
        label="Prior signal-proportional 150/50",
    )
    ax.axhline(1.0, color="0.45", linewidth=0.8)
    ax.set_title("Historical out-of-sample portfolio evaluation")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value, starting value = 1")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def validation_audits(
    selected_model: str,
    validation_search: pd.DataFrame,
    selected_risk_aversion: float,
    selected_validation_targets: pd.DataFrame,
    test_targets: pd.DataFrame,
    validation_dates: pd.DatetimeIndex,
    test_dates: pd.DatetimeIndex,
    test_results: dict[str, pd.DataFrame],
    solver_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Check selection isolation, target constraints, and saved P&L."""
    checks: list[dict[str, str]] = []

    if not validation_search["sample"].eq("validation").all():
        raise AssertionError("Risk-aversion search contains non-validation rows")
    expected = validation_search.sort_values(
        ["sharpe", "annualized_return", "risk_aversion"],
        ascending=[False, False, False],
    ).iloc[0]["risk_aversion"]
    if not np.isclose(expected, selected_risk_aversion):
        raise AssertionError("Selected risk aversion is not the validation winner")
    checks.append(
        {
            "check": "risk_aversion_selected_on_validation_only",
            "status": "pass",
            "detail": f"{selected_risk_aversion:g}",
        }
    )

    ranking = pd.read_csv(MODEL_RANKING_PATH)
    selected_flag = ranking["selected_model"].astype(str).str.lower().eq("true")
    selected = ranking.loc[selected_flag].iloc[0]
    if selected["model"] != selected_model or selected["sample"] != "validation":
        raise AssertionError("Expected-return model was not the validation-selected winner")
    checks.append(
        {
            "check": "expected_return_model_selected_on_validation",
            "status": "pass",
            "detail": selected_model,
        }
    )

    for label, targets, dates in (
        ("validation", selected_validation_targets, validation_dates),
        ("test", test_targets, test_dates),
    ):
        active = targets.loc[dates].abs().sum(axis=1).gt(0.0)
        active_targets = targets.loc[dates].loc[active]
        net = active_targets.sum(axis=1)
        gross = active_targets.abs().sum(axis=1)
        if not np.allclose(net, NET_EXPOSURE, atol=1e-7):
            raise AssertionError(f"{label} targets violate net exposure")
        if (gross > GROSS_LIMIT + 1e-7).any():
            raise AssertionError(f"{label} targets violate gross exposure")
        unavailable = EXECUTION_LAG_DAYS + HOLDING_DAYS
        if targets.loc[dates[-unavailable:]].abs().to_numpy().max() > 1e-12:
            raise AssertionError(f"{label} has an incomplete boundary hold")
    checks.append(
        {
            "check": "target_constraints_and_complete_holds",
            "status": "pass",
            "detail": "100% net, gross no greater than 300%",
        }
    )

    if solver_diagnostics.empty or not np.isfinite(
        solver_diagnostics.select_dtypes(include=[np.number]).to_numpy()
    ).all():
        raise AssertionError("Optimization diagnostics are incomplete")
    checks.append(
        {
            "check": "finite_optimizer_outputs",
            "status": "pass",
            "detail": f"{len(solver_diagnostics)} historical test dates",
        }
    )

    for label, result in test_results.items():
        independent = (1.0 + result["portfolio_return"].fillna(0.0)).cumprod()
        if not np.allclose(
            independent,
            result["portfolio_value"],
            atol=1e-12,
        ):
            raise AssertionError(f"{label} portfolio path does not match returns")
        if not np.isclose(result["portfolio_value"].iloc[0], 1.0):
            raise AssertionError(f"{label} test backtest did not start from one")
    checks.append(
        {
            "check": "independent_test_portfolio_recalculation",
            "status": "pass",
            "detail": "optimized and baseline",
        }
    )
    return pd.DataFrame(checks)


def write_summary(
    selected_model: str,
    selected_risk_aversion: float,
    validation_search: pd.DataFrame,
    test_performance: pd.DataFrame,
    identical_test_signal_dates: int,
    total_test_signal_dates: int,
) -> Path:
    """Write a concise interpretation of the optimization exercise."""
    selected_validation = validation_search.loc[
        validation_search["risk_aversion"].eq(selected_risk_aversion)
    ].iloc[0]
    optimized_test = test_performance.loc[
        test_performance["strategy"].eq("mean_variance_optimized")
    ].iloc[0]
    baseline_test = test_performance.loc[
        test_performance["strategy"].eq("signal_proportional_150_50")
    ].iloc[0]
    lines = [
        "PORTFOLIO OPTIMIZATION RESULTS",
        "",
        "Method",
        f"- Expected-return model: {selected_model}, selected previously by validation MSE.",
        f"- Risk estimate: trailing {COVARIANCE_LOOKBACK}-day covariance with {COVARIANCE_SHRINKAGE:.0%} diagonal shrinkage.",
        "- Constraints: 100% net exposure and no more than 300% gross exposure.",
        "- Risk aversion was selected by validation Sharpe before test evaluation.",
        "",
        f"Selected risk-aversion score: {selected_risk_aversion:g}",
        f"Selected validation Sharpe: {selected_validation['sharpe']:.3f}",
        "",
        "Historical out-of-sample results",
        f"- Optimized ending value: {optimized_test['ending_value']:.3f}",
        f"- Optimized annualized return: {optimized_test['annualized_return']:.1%}",
        f"- Optimized Sharpe: {optimized_test['sharpe']:.3f}",
        f"- Optimized maximum drawdown: {optimized_test['max_drawdown']:.1%}",
        f"- Baseline ending value: {baseline_test['ending_value']:.3f}",
        f"- Baseline Sharpe: {baseline_test['sharpe']:.3f}",
        "",
        "Interpretation",
        f"- LASSO forecasts were identical across ETFs on {identical_test_signal_dates} of {total_test_signal_dates} historical test dates.",
        "- With a required 100% net exposure, those dates reduce to minimum-variance allocation.",
        "The historical test result is reported after the risk-aversion parameter was frozen.",
        "It is one historical out-of-sample period and does not establish future performance.",
    ]
    path = RESULT_DIR / "research_summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_analysis() -> dict[str, object]:
    """Run validation selection, freeze risk aversion, and evaluate the test period."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    prices = load_close_prices()
    selected_model = selected_model_from_validation()

    # Selection stage. Only validation predictions and validation returns enter.
    validation_dates = sample_dates(prices, "validation")
    validation_signal = load_prediction_signal(
        prices,
        selected_model,
        "validation",
    )
    validation_covariances = covariance_cache(prices, validation_dates)
    validation_rows: list[dict[str, object]] = []
    validation_results: dict[float, pd.DataFrame] = {}
    validation_targets: dict[float, pd.DataFrame] = {}

    for risk_aversion in RISK_AVERSION_GRID:
        targets, _ = build_optimized_targets(
            prices,
            validation_signal,
            validation_dates,
            validation_covariances,
            float(risk_aversion),
        )
        targets = purge_incomplete_holds(targets, validation_dates)
        result = backtest_sample(
            prices,
            targets,
            validation_dates,
        )
        validation_targets[float(risk_aversion)] = targets
        validation_results[float(risk_aversion)] = result
        validation_rows.append(
            performance_metrics(
                result,
                sample="validation",
                strategy="mean_variance_optimized",
                risk_aversion=float(risk_aversion),
            )
        )

    validation_search = pd.DataFrame(validation_rows)
    validation_search = validation_search.sort_values(
        ["sharpe", "annualized_return", "risk_aversion"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    validation_search.insert(
        0,
        "validation_rank",
        np.arange(1, len(validation_search) + 1),
    )
    validation_search["selected"] = False
    validation_search.loc[0, "selected"] = True
    selected_risk_aversion = float(validation_search.loc[0, "risk_aversion"])
    selected_validation_targets = validation_targets[selected_risk_aversion]

    # Final evaluation stage. Test predictions are loaded after selection freezes.
    test_dates = sample_dates(prices, "test")
    test_signal = load_prediction_signal(prices, selected_model, "test")
    test_covariances = covariance_cache(prices, test_dates)
    test_targets, test_diagnostics = build_optimized_targets(
        prices,
        test_signal,
        test_dates,
        test_covariances,
        selected_risk_aversion,
    )
    test_targets = purge_incomplete_holds(test_targets, test_dates)
    optimized_test = backtest_sample(prices, test_targets, test_dates)

    reference_targets = baseline_targets(prices, test_signal, test_dates)
    reference_targets = purge_incomplete_holds(reference_targets, test_dates)
    baseline_test = backtest_sample(prices, reference_targets, test_dates)
    test_results = {
        "mean_variance_optimized": optimized_test,
        "signal_proportional_150_50": baseline_test,
    }
    test_performance = pd.DataFrame(
        [
            performance_metrics(
                optimized_test,
                sample="test",
                strategy="mean_variance_optimized",
                risk_aversion=selected_risk_aversion,
            ),
            performance_metrics(
                baseline_test,
                sample="test",
                strategy="signal_proportional_150_50",
                risk_aversion=None,
            ),
        ]
    )

    audits = validation_audits(
        selected_model,
        validation_search,
        selected_risk_aversion,
        selected_validation_targets,
        test_targets,
        validation_dates,
        test_dates,
        test_results,
        test_diagnostics,
    )

    table_paths = {
        "validation": TABLE_DIR / "validation_risk_aversion_search.csv",
        "test": TABLE_DIR / "historical_test_performance.csv",
        "weights": TABLE_DIR / "historical_test_target_weights.csv",
        "diagnostics": TABLE_DIR / "historical_test_optimizer_diagnostics.csv",
        "values": TABLE_DIR / "historical_test_portfolio_values.csv",
        "audits": TABLE_DIR / "validation_checks.csv",
        "settings": TABLE_DIR / "analysis_settings.json",
    }
    validation_search.to_csv(table_paths["validation"], index=False)
    test_performance.to_csv(table_paths["test"], index=False)
    test_targets.loc[test_targets.abs().sum(axis=1).gt(0)].to_csv(
        table_paths["weights"],
        index_label="date",
    )
    test_diagnostics.to_csv(table_paths["diagnostics"], index=False)
    pd.DataFrame(
        {
            "mean_variance_optimized": optimized_test["portfolio_value"],
            "signal_proportional_150_50": baseline_test["portfolio_value"],
        }
    ).to_csv(table_paths["values"], index_label="date")
    audits.to_csv(table_paths["audits"], index=False)
    settings = {
        "expected_return_model": selected_model,
        "model_selection_source": "walk-forward outer validation MSE",
        "risk_aversion_selection_metric": "validation Sharpe",
        "risk_aversion_grid": RISK_AVERSION_GRID.tolist(),
        "selected_risk_aversion": selected_risk_aversion,
        "covariance_lookback": COVARIANCE_LOOKBACK,
        "covariance_min_periods": COVARIANCE_MIN_PERIODS,
        "covariance_shrinkage": COVARIANCE_SHRINKAGE,
        "net_exposure": NET_EXPOSURE,
        "gross_limit": GROSS_LIMIT,
        "holding_days": HOLDING_DAYS,
        "execution_lag_days": EXECUTION_LAG_DAYS,
        "risk_free_rate": RISK_FREE_RATE,
    }
    table_paths["settings"].write_text(
        json.dumps(settings, indent=2) + "\n",
        encoding="utf-8",
    )

    plot_paths = {
        "selection": PLOT_DIR / "validation_risk_aversion_selection.png",
        "test": PLOT_DIR / "historical_out_of_sample_portfolio.png",
    }
    plot_validation_selection(
        validation_search,
        selected_risk_aversion,
        plot_paths["selection"],
    )
    plot_test_results(
        optimized_test,
        baseline_test,
        selected_risk_aversion,
        plot_paths["test"],
    )
    summary_path = write_summary(
        selected_model,
        selected_risk_aversion,
        validation_search,
        test_performance,
        int(test_signal.std(axis=1).fillna(0.0).le(1e-15).loc[test_dates].sum()),
        len(test_dates),
    )
    return {
        "selected_model": selected_model,
        "selected_risk_aversion": selected_risk_aversion,
        "validation_search": validation_search,
        "test_performance": test_performance,
        "audits": audits,
        "summary_path": summary_path,
    }


if __name__ == "__main__":
    output = run_analysis()
    print(f"Selected model: {output['selected_model']}")
    print(f"Selected risk aversion: {output['selected_risk_aversion']:g}")
    print("\nValidation search")
    print(
        output["validation_search"][
            [
                "validation_rank",
                "risk_aversion",
                "annualized_return",
                "annualized_volatility",
                "sharpe",
                "max_drawdown",
                "selected",
            ]
        ].to_string(index=False)
    )
    print("\nHistorical test performance")
    print(output["test_performance"].to_string(index=False))
    print("\nValidation checks")
    print(output["audits"].to_string(index=False))

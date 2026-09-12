"""End-to-end ETF research pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .data import data_quality_report, load_adjusted_prices, source_manifest
from .features import build_cross_sectional_panel
from .models import (
    MIN_FORECAST_DISPERSION,
    MODEL_LABELS,
    monthly_walk_forward,
    prediction_performance,
    select_model,
)
from .portfolio import (
    build_optimized_targets,
    build_rank_targets,
    backtest_cohorts,
    performance_metrics,
    purge_incomplete_holds,
)

STRATEGY_LABELS = {
    "optimized_relative_return": "Model-based mean-variance portfolio",
    "momra_reversal_rank": "MOMRA reversal rank portfolio",
    "cash_no_deployment": "Cash",
}


def load_config(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_optional_csv(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        path.unlink(missing_ok=True)
    else:
        frame.to_csv(path, index=False)


def _dates_between(
    index: pd.DatetimeIndex,
    start: str,
    end: str | None,
) -> pd.DatetimeIndex:
    mask = index >= pd.Timestamp(start)
    if end is not None:
        mask &= index <= pd.Timestamp(end)
    return index[mask]


def _prediction_signal(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    if predictions.duplicated(["date", "symbol"]).any():
        raise AssertionError("Duplicate ETF-date forecasts")
    return predictions.pivot(
        index="date",
        columns="symbol",
        values="predicted_5d_relative_return",
    ).reindex(index=prices.index, columns=prices.columns)


def _backtest_one_sample(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    dates: pd.DatetimeIndex,
    config: dict[str, object],
) -> pd.DataFrame:
    return backtest_cohorts(
        prices.loc[dates],
        targets.loc[dates],
        transaction_cost_bps=float(config["transaction_cost_bps"]),
        annual_short_borrow_bps=float(config["annual_short_borrow_bps"]),
    )


def plot_model_selection(ranking: pd.DataFrame, output: Path) -> None:
    ordered = ranking.sort_values("mean_daily_rank_ic")
    colors = ["#D62728" if selected else "#4C78A8" for selected in ordered["selected"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].barh(ordered["model_label"], ordered["mean_daily_rank_ic"], color=colors)
    axes[0].axvline(0.0, color="0.4", linewidth=0.8)
    axes[0].set_xlabel("Mean daily Spearman rank IC")
    axes[0].set_title("Validation ranking signal")
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].barh(ordered["model_label"], ordered["active_forecast_share"], color=colors)
    axes[1].set_xlabel("Dates with cross-sectional forecast dispersion")
    axes[1].set_title("Signal availability")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].grid(axis="x", alpha=0.2)
    fig.suptitle("Cross-sectional model selection on validation")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_risk_selection(search: pd.DataFrame, selected: float, output: Path) -> None:
    ordered = search.sort_values("risk_aversion")
    chosen = ordered["risk_aversion"].eq(selected)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ordered["risk_aversion"], ordered["sharpe"], marker="o")
    ax.scatter(
        ordered.loc[chosen, "risk_aversion"],
        ordered.loc[chosen, "sharpe"],
        color="#D62728",
        s=90,
        label="Selected on validation",
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Risk-aversion score, log scale")
    ax.set_ylabel("Validation Sharpe after costs")
    ax.set_title("Risk-aversion parameter selected before historical test evaluation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_strategy_selection(selection: pd.DataFrame, output: Path) -> None:
    ordered = selection.sort_values("sharpe").copy()
    ordered["display_label"] = ordered["strategy"].map(STRATEGY_LABELS)
    colors = ["#D62728" if selected else "#4C78A8" for selected in ordered["selected"]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(ordered["display_label"], ordered["sharpe"], color=colors)
    ax.axvline(0.0, color="0.4", linewidth=0.8)
    ax.set_xlabel("Validation Sharpe after modeled costs")
    ax.set_title("Deployment decision made before historical test evaluation")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_historical_evaluation(
    results: dict[str, pd.DataFrame],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for name, result in results.items():
        label = STRATEGY_LABELS.get(name, name.replace("_", " ").title())
        axes[0].plot(result.index, result["portfolio_value"], label=label, linewidth=1.5)
        drawdown = result["portfolio_value"].div(result["portfolio_value"].cummax()).sub(1.0)
        axes[1].plot(result.index, drawdown, label=label, linewidth=1.3)
    axes[0].axhline(1.0, color="0.45", linewidth=0.8)
    axes[0].set_ylabel("Portfolio value")
    axes[0].set_title("Historical out-of-sample evaluation, net of modeled costs")
    axes[0].legend(frameon=False)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def audit_pipeline(
    prices: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    selected_model: str,
    test_predictions: pd.DataFrame,
    validation_ranking: pd.DataFrame,
    risk_search: pd.DataFrame,
    selected_risk_aversion: float,
    strategy_selection: pd.DataFrame,
    selected_strategy: str,
    test_targets: pd.DataFrame,
    test_dates: pd.DatetimeIndex,
    test_results: dict[str, pd.DataFrame],
    config: dict[str, object],
) -> pd.DataFrame:
    checks: list[dict[str, str]] = []
    expected_model = validation_ranking.sort_values(
        ["mean_daily_rank_ic", "mse", "active_forecast_share"],
        ascending=[False, True, False],
    ).iloc[0]["model"]
    if expected_model != selected_model:
        raise AssertionError("Model selection does not reproduce from validation")
    if selected_strategy == "optimized_relative_return":
        if test_predictions.empty or not test_predictions["model"].eq(selected_model).all():
            raise AssertionError("Historical test contains an unfrozen model family")
    elif not test_predictions.empty:
        raise AssertionError("Model test predictions were generated for an undeployed model")
    checks.append(
        {
            "check": "model_family_frozen_before_historical_test",
            "status": "pass",
            "detail": selected_model,
        }
    )

    expected_strategy = strategy_selection.sort_values(
        ["sharpe", "annualized_return", "strategy"],
        ascending=[False, False, True],
    ).iloc[0]["strategy"]
    if expected_strategy != selected_strategy:
        raise AssertionError("Final strategy does not reproduce from validation")
    checks.append(
        {
            "check": "deployment_strategy_selected_on_validation",
            "status": "pass",
            "detail": selected_strategy,
        }
    )

    expected_risk = risk_search.sort_values(
        ["sharpe", "annualized_return", "risk_aversion"],
        ascending=[False, False, False],
    ).iloc[0]["risk_aversion"]
    if not np.isclose(expected_risk, selected_risk_aversion):
        raise AssertionError("Risk aversion does not reproduce from validation")
    checks.append(
        {
            "check": "risk_aversion_selected_on_validation",
            "status": "pass",
            "detail": f"{selected_risk_aversion:g}",
        }
    )

    if validation_predictions["date"].max() >= pd.Timestamp(config["historical_test_start"]):
        raise AssertionError("Validation model ranking includes historical test dates")
    checks.append(
        {
            "check": "validation_ranking_excludes_historical_test",
            "status": "pass",
            "detail": str(validation_predictions["date"].max().date()),
        }
    )

    active = test_targets.loc[test_dates].abs().sum(axis=1).gt(0)
    active_targets = test_targets.loc[test_dates].loc[active]
    if not np.allclose(active_targets.sum(axis=1), 0.0, atol=1e-7):
        raise AssertionError("Selected portfolio is not market neutral")
    if (active_targets.abs().sum(axis=1) > float(config["gross_limit"]) + 1e-7).any():
        raise AssertionError("Selected portfolio exceeds gross limit")
    if (
        active_targets.abs().max(axis=1)
        > float(config["position_limit"]) + 1e-7
    ).any():
        raise AssertionError("Selected portfolio exceeds ETF position limit")
    checks.append(
        {
            "check": "portfolio_constraints",
            "status": "pass",
            "detail": "zero net, bounded gross and ETF weights",
        }
    )

    for name, result in test_results.items():
        if not np.isfinite(result.select_dtypes(include=[np.number])).all().all():
            raise AssertionError(f"{name} contains invalid backtest values")
        if (result[["transaction_cost", "borrow_cost"]] < -1e-15).any().any():
            raise AssertionError(f"{name} contains negative modeled costs")
    checks.append(
        {
            "check": "finite_net_of_cost_backtests",
            "status": "pass",
            "detail": f"{len(test_results)} strategies",
        }
    )

    max_return = prices.pct_change(fill_method=None).abs().max().max()
    if max_return >= 1.0:
        raise AssertionError("A split-like daily price return remains")
    checks.append(
        {
            "check": "adjusted_price_discontinuities",
            "status": "pass",
            "detail": f"max absolute daily return {max_return:.3f}",
        }
    )
    return pd.DataFrame(checks)


def run_pipeline(
    project_root: str | Path,
    config_path: str | Path,
) -> dict[str, object]:
    """Run selection stages, freeze choices, and evaluate the inspected test period."""
    root = Path(project_root)
    config = load_config(config_path)
    report_dir = root / "reports"
    figure_dir = report_dir / "figures"
    table_dir = report_dir / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    prices = load_adjusted_prices(root)
    quality = data_quality_report(prices)
    panel, feature_names, processed = build_cross_sectional_panel(prices)

    validation_dates = _dates_between(
        prices.index,
        str(config["validation_start"]),
        str(config["validation_end"]),
    )
    validation_predictions = []
    validation_folds = []
    validation_grids = []
    for model_name in config["models"]:
        prediction, folds, grid = monthly_walk_forward(
            panel,
            feature_names,
            str(model_name),
            validation_dates.min(),
            validation_dates.max(),
            training_months=int(config["training_months"]),
            validation_months=int(config["inner_validation_months"]),
        )
        validation_predictions.append(prediction)
        validation_folds.append(folds)
        validation_grids.append(grid)
    validation_predictions_frame = pd.concat(validation_predictions, ignore_index=True)
    validation_performance = prediction_performance(validation_predictions_frame)
    selected_model, model_ranking = select_model(validation_performance)

    selected_validation_predictions = validation_predictions_frame.loc[
        validation_predictions_frame["model"].eq(selected_model)
    ]
    validation_signal = _prediction_signal(selected_validation_predictions, prices)
    risk_rows = []
    validation_results: dict[float, pd.DataFrame] = {}
    for risk_aversion in config["risk_aversion_grid"]:
        targets, _ = build_optimized_targets(
            prices,
            validation_signal,
            validation_dates,
            float(risk_aversion),
            gross_limit=float(config["gross_limit"]),
            position_limit=float(config["position_limit"]),
            turnover_penalty=float(config["turnover_penalty"]),
            covariance_lookback=int(config["covariance_lookback"]),
            covariance_min_periods=int(config["covariance_min_periods"]),
        )
        targets = purge_incomplete_holds(targets, validation_dates)
        result = _backtest_one_sample(prices, targets, validation_dates, config)
        validation_results[float(risk_aversion)] = result
        row = performance_metrics(result, "optimized_relative_return", "validation")
        row["risk_aversion"] = float(risk_aversion)
        risk_rows.append(row)
    risk_search = pd.DataFrame(risk_rows).sort_values(
        ["sharpe", "annualized_return", "risk_aversion"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    risk_search.insert(0, "validation_rank", np.arange(1, len(risk_search) + 1))
    risk_search["selected"] = False
    risk_search.loc[0, "selected"] = True
    selected_risk_aversion = float(risk_search.loc[0, "risk_aversion"])

    model_validation = validation_results[selected_risk_aversion]
    momra_validation_targets = purge_incomplete_holds(
        build_rank_targets(
            prices,
            -processed["momra_20"].raw,
            validation_dates,
            gross_limit=float(config["gross_limit"]),
        ),
        validation_dates,
    )
    momra_validation = _backtest_one_sample(
        prices,
        momra_validation_targets,
        validation_dates,
        config,
    )
    cash_validation_targets = pd.DataFrame(
        0.0,
        index=prices.index,
        columns=prices.columns,
    )
    cash_validation = _backtest_one_sample(
        prices,
        cash_validation_targets,
        validation_dates,
        config,
    )
    strategy_selection = pd.DataFrame(
        [
            performance_metrics(
                model_validation,
                "optimized_relative_return",
                "validation",
            ),
            performance_metrics(
                momra_validation,
                "momra_reversal_rank",
                "validation",
            ),
            performance_metrics(
                cash_validation,
                "cash_no_deployment",
                "validation",
            ),
        ]
    )
    strategy_selection.loc[
        strategy_selection["strategy"].eq("cash_no_deployment"),
        ["sharpe", "sortino", "calmar"],
    ] = 0.0
    strategy_selection = strategy_selection.sort_values(
        ["sharpe", "annualized_return", "strategy"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    strategy_selection.insert(
        0,
        "validation_rank",
        np.arange(1, len(strategy_selection) + 1),
    )
    strategy_selection["selected"] = False
    strategy_selection.loc[0, "selected"] = True
    selected_strategy = str(strategy_selection.loc[0, "strategy"])

    # Model, risk aversion, and deployment choice are frozen above. Only the
    # selected strategy is evaluated on the previously inspected test period.
    historical_test_dates = _dates_between(
        prices.index,
        str(config["historical_test_start"]),
        config.get("historical_test_end"),
    )
    test_predictions = pd.DataFrame()
    test_folds = pd.DataFrame()
    test_grid = pd.DataFrame()
    optimizer_diagnostics = pd.DataFrame()
    selected_targets = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if selected_strategy == "optimized_relative_return":
        test_predictions, test_folds, test_grid = monthly_walk_forward(
            panel,
            feature_names,
            selected_model,
            historical_test_dates.min(),
            historical_test_dates.max(),
            training_months=int(config["training_months"]),
            validation_months=int(config["inner_validation_months"]),
        )
        test_signal = _prediction_signal(test_predictions, prices)
        selected_targets, optimizer_diagnostics = build_optimized_targets(
            prices,
            test_signal,
            historical_test_dates,
            selected_risk_aversion,
            gross_limit=float(config["gross_limit"]),
            position_limit=float(config["position_limit"]),
            turnover_penalty=float(config["turnover_penalty"]),
            covariance_lookback=int(config["covariance_lookback"]),
            covariance_min_periods=int(config["covariance_min_periods"]),
        )
    elif selected_strategy == "momra_reversal_rank":
        selected_targets = build_rank_targets(
            prices,
            -processed["momra_20"].raw,
            historical_test_dates,
            gross_limit=float(config["gross_limit"]),
        )
    selected_targets = purge_incomplete_holds(selected_targets, historical_test_dates)
    selected_result = _backtest_one_sample(
        prices,
        selected_targets,
        historical_test_dates,
        config,
    )
    cash_test = _backtest_one_sample(
        prices,
        pd.DataFrame(0.0, index=prices.index, columns=prices.columns),
        historical_test_dates,
        config,
    )
    test_results = {
        selected_strategy: selected_result,
        "cash_no_deployment": cash_test,
    }
    test_performance = pd.DataFrame(
        [
            performance_metrics(result, name, "historical_test")
            for name, result in test_results.items()
        ]
    )
    test_prediction_performance = (
        prediction_performance(test_predictions)
        if not test_predictions.empty
        else pd.DataFrame()
    )

    checks = audit_pipeline(
        prices,
        validation_predictions_frame,
        selected_model,
        test_predictions,
        model_ranking,
        risk_search,
        selected_risk_aversion,
        strategy_selection,
        selected_strategy,
        selected_targets,
        historical_test_dates,
        test_results,
        config,
    )

    quality.to_csv(table_dir / "data_quality.csv", index=False)
    source_manifest(root).to_csv(table_dir / "data_source_manifest.csv", index=False)
    model_ranking.to_csv(table_dir / "validation_model_ranking.csv", index=False)
    risk_search.to_csv(table_dir / "validation_risk_aversion.csv", index=False)
    strategy_selection.to_csv(
        table_dir / "validation_strategy_selection.csv",
        index=False,
    )
    _write_optional_csv(
        test_prediction_performance,
        table_dir / "historical_test_prediction_performance.csv",
    )
    test_performance.to_csv(
        table_dir / "historical_test_portfolio_performance.csv",
        index=False,
    )
    pd.concat(validation_folds, ignore_index=True).to_csv(
        table_dir / "validation_walk_forward_folds.csv",
        index=False,
    )
    pd.concat(validation_grids, ignore_index=True).to_csv(
        table_dir / "validation_hyperparameter_search.csv",
        index=False,
    )
    _write_optional_csv(
        test_folds,
        table_dir / "historical_test_walk_forward_folds.csv",
    )
    _write_optional_csv(
        test_grid,
        table_dir / "historical_test_inner_parameter_search.csv",
    )
    _write_optional_csv(
        optimizer_diagnostics,
        table_dir / "optimizer_diagnostics.csv",
    )
    selected_targets.loc[historical_test_dates].to_csv(
        table_dir / "historical_test_target_weights.csv",
        index_label="date",
    )
    checks.to_csv(table_dir / "pipeline_checks.csv", index=False)
    pd.DataFrame(
        {
            name: result["portfolio_value"]
            for name, result in test_results.items()
        }
    ).to_csv(table_dir / "historical_test_portfolio_values.csv", index_label="date")

    plot_model_selection(model_ranking, figure_dir / "validation_model_selection.png")
    plot_risk_selection(
        risk_search,
        selected_risk_aversion,
        figure_dir / "validation_risk_selection.png",
    )
    plot_strategy_selection(
        strategy_selection,
        figure_dir / "validation_strategy_selection.png",
    )
    plot_historical_evaluation(
        test_results,
        figure_dir / "historical_out_of_sample_evaluation.png",
    )

    settings = {
        **config,
        "selected_model": selected_model,
        "selected_risk_aversion": selected_risk_aversion,
        "selected_strategy": selected_strategy,
        "feature_names": feature_names,
        "historical_test_status": "previously inspected, not a fresh blind holdout",
        "return_type": "split-adjusted price return, dividends not fully incorporated",
    }
    (table_dir / "pipeline_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_test = test_performance.loc[
        test_performance["strategy"].eq(selected_strategy)
    ].iloc[0]
    selected_validation = strategy_selection.loc[
        strategy_selection["strategy"].eq(selected_strategy)
    ].iloc[0]
    summary_lines = [
        "CROSS-ASSET ETF RESEARCH: FINAL EVALUATION",
        "",
        "Selection",
        f"- Best cross-sectional model on validation rank IC: {MODEL_LABELS[selected_model]}.",
        f"- Final validation-selected strategy: {selected_strategy}.",
        f"- Validation Sharpe after costs: {selected_validation['sharpe']:.3f}.",
        "",
        "Historical out-of-sample evaluation",
        f"- Ending value: {selected_test['ending_value']:.3f}.",
        f"- Annualized return: {selected_test['annualized_return']:.1%}.",
        f"- Annualized volatility: {selected_test['annualized_volatility']:.1%}.",
        f"- Sharpe: {selected_test['sharpe']:.3f}.",
        f"- Maximum drawdown: {selected_test['max_drawdown']:.1%}.",
        "",
        "Research controls",
        "- Signals use cross-sectional relative-return targets.",
        "- Collapsed forecasts produce cash rather than forced risky exposure.",
        "- Results include 5 bps transaction costs and 25 bps annual short borrow cost.",
        "- Risk uses a trailing Ledoit-Wolf shrinkage covariance estimate.",
        "",
        "Limitations",
        "- The historical test period was inspected in earlier revisions and is not a fresh blind holdout.",
        "- The source data represent split-adjusted price returns rather than complete ETF total returns.",
    ]
    (report_dir / "research_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )
    return {
        "selected_model": selected_model,
        "selected_risk_aversion": selected_risk_aversion,
        "selected_strategy": selected_strategy,
        "strategy_selection": strategy_selection,
        "model_ranking": model_ranking,
        "risk_search": risk_search,
        "test_prediction_performance": test_prediction_performance,
        "test_performance": test_performance,
        "checks": checks,
    }

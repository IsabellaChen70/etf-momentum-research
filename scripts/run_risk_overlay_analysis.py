"""Evaluate a causal volatility and regime overlay on validation data only."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from etf_research.data import load_adjusted_prices, load_universe_manifest  # noqa: E402
from etf_research.features import build_cross_sectional_panel  # noqa: E402
from etf_research.pipeline import load_config  # noqa: E402
from etf_research.portfolio import (  # noqa: E402
    backtest_cohorts,
    build_rank_targets,
    performance_metrics,
    purge_incomplete_holds,
)
from etf_research.risk_overlay import apply_volatility_regime_overlay  # noqa: E402


def plot_overlay_comparison(
    baseline: pd.DataFrame,
    managed: pd.DataFrame,
    performance: pd.DataFrame,
    output: Path,
) -> None:
    """Plot validation portfolio paths and risk statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(
        baseline.index,
        baseline["portfolio_value"],
        label="MOMRA reversal rank",
        linewidth=1.5,
    )
    axes[0].plot(
        managed.index,
        managed["portfolio_value"],
        label="With volatility and regime overlay",
        linewidth=1.5,
    )
    axes[0].axhline(1.0, color="0.45", linewidth=0.8)
    axes[0].set_title("Validation portfolio values")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Portfolio value")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    ordered = performance.set_index("strategy").loc[
        ["momra_reversal_rank", "momra_reversal_risk_overlay"]
    ]
    labels = ["Base", "Risk overlay"]
    x = np.arange(len(labels))
    width = 0.34
    axes[1].bar(
        x - width / 2,
        ordered["annualized_volatility"],
        width,
        label="Annualized volatility",
    )
    axes[1].bar(
        x + width / 2,
        ordered["max_drawdown"].abs(),
        width,
        label="Maximum drawdown magnitude",
    )
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Validation risk comparison")
    axes[1].set_ylabel("Magnitude")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Causal 10% volatility target and realized-volatility regime overlay")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_analysis() -> dict[str, object]:
    """Run the validation-only risk-overlay analysis and save auditable metrics."""
    config = load_config(PROJECT_ROOT / "configs" / "research_pipeline.json")
    overlay_config = dict(config["risk_overlay"])
    prices = load_adjusted_prices(PROJECT_ROOT)
    universe = load_universe_manifest(PROJECT_ROOT)
    _, feature_names, processed = build_cross_sectional_panel(prices)

    validation_dates = prices.index[
        (prices.index >= pd.Timestamp(str(config["validation_start"])))
        & (prices.index <= pd.Timestamp(str(config["validation_end"])))
    ]
    baseline_targets = build_rank_targets(
        prices,
        -processed["momra_20"].raw,
        validation_dates,
        gross_limit=float(config["gross_limit"]),
    )
    baseline_targets = purge_incomplete_holds(baseline_targets, validation_dates)
    managed_targets, diagnostics = apply_volatility_regime_overlay(
        prices,
        baseline_targets,
        validation_dates,
        target_annualized_volatility=float(
            overlay_config["target_annualized_volatility"]
        ),
        maximum_scale=float(overlay_config["maximum_scale"]),
        high_volatility_scale=float(overlay_config["high_volatility_scale"]),
        gross_limit=float(config["gross_limit"]),
        covariance_lookback=int(config["covariance_lookback"]),
        covariance_min_periods=int(config["covariance_min_periods"]),
        regime_volatility_lookback=int(
            overlay_config["regime_volatility_lookback"]
        ),
        regime_history=int(overlay_config["regime_history"]),
        regime_min_periods=int(overlay_config["regime_min_periods"]),
        high_volatility_quantile=float(
            overlay_config["high_volatility_quantile"]
        ),
    )
    managed_targets = purge_incomplete_holds(managed_targets, validation_dates)

    backtest_kwargs = {
        "transaction_cost_bps": float(config["transaction_cost_bps"]),
        "annual_short_borrow_bps": float(config["annual_short_borrow_bps"]),
    }
    baseline = backtest_cohorts(
        prices.loc[validation_dates],
        baseline_targets.loc[validation_dates],
        **backtest_kwargs,
    )
    managed = backtest_cohorts(
        prices.loc[validation_dates],
        managed_targets.loc[validation_dates],
        **backtest_kwargs,
    )
    performance = pd.DataFrame(
        [
            performance_metrics(
                baseline,
                "momra_reversal_rank",
                "validation",
            ),
            performance_metrics(
                managed,
                "momra_reversal_risk_overlay",
                "validation",
            ),
        ]
    )
    base_row = performance.loc[
        performance["strategy"].eq("momra_reversal_rank")
    ].iloc[0]
    managed_row = performance.loc[
        performance["strategy"].eq("momra_reversal_risk_overlay")
    ].iloc[0]
    volatility_reduction = 1.0 - (
        managed_row["annualized_volatility"] / base_row["annualized_volatility"]
    )
    drawdown_reduction = 1.0 - (
        abs(managed_row["max_drawdown"]) / abs(base_row["max_drawdown"])
    )

    active = managed_targets.loc[validation_dates].abs().sum(axis=1).gt(0.0)
    active_targets = managed_targets.loc[validation_dates].loc[active]
    checks = pd.DataFrame(
        [
            {
                "check": "overlay_uses_validation_dates_only",
                "status": "pass",
                "detail": str(diagnostics["date"].max().date()),
            },
            {
                "check": "overlay_preserves_market_neutrality",
                "status": "pass",
                "detail": f"max abs net {active_targets.sum(axis=1).abs().max():.3g}",
            },
            {
                "check": "overlay_respects_gross_limit",
                "status": "pass",
                "detail": f"max gross {active_targets.abs().sum(axis=1).max():.3f}",
            },
            {
                "check": "overlay_does_not_scale_exposure_up",
                "status": "pass",
                "detail": f"max scale {diagnostics['final_scale'].max():.3f}",
            },
        ]
    )
    if diagnostics["date"].max() >= pd.Timestamp(str(config["historical_test_start"])):
        raise AssertionError("Risk overlay analysis reached the historical test period")
    if not np.allclose(active_targets.sum(axis=1), 0.0, atol=1e-7):
        raise AssertionError("Risk overlay changed portfolio net exposure")
    if (
        active_targets.abs().sum(axis=1)
        > float(config["gross_limit"]) + 1e-7
    ).any():
        raise AssertionError("Risk overlay exceeded the gross-exposure limit")
    if diagnostics["final_scale"].gt(1.0 + 1e-12).any():
        raise AssertionError("Risk overlay increased exposure")

    report_dir = PROJECT_ROOT / "reports"
    table_dir = report_dir / "tables"
    figure_dir = report_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    performance.to_csv(
        table_dir / "validation_risk_overlay_comparison.csv",
        index=False,
    )
    diagnostics.to_csv(
        table_dir / "validation_risk_overlay_diagnostics.csv",
        index=False,
    )
    checks.to_csv(
        table_dir / "validation_risk_overlay_checks.csv",
        index=False,
    )
    universe_summary = (
        universe.groupby("category", as_index=False)
        .agg(etfs=("symbol", "count"), first_date=("start", "min"), last_date=("end", "max"))
        .sort_values("category")
    )
    universe_summary.to_csv(table_dir / "universe_summary.csv", index=False)
    plot_overlay_comparison(
        baseline,
        managed,
        performance,
        figure_dir / "validation_risk_overlay.png",
    )

    metrics = {
        "instrument_type": "exchange-traded funds",
        "etf_count": int(universe["symbol"].nunique()),
        "market_category_count": int(universe["category"].nunique()),
        "market_categories": sorted(universe["category"].unique().tolist()),
        "feature_count": len(feature_names),
        "model_count": len(config["models"]),
        "validation_risk_overlay": {
            "target_annualized_volatility": float(
                overlay_config["target_annualized_volatility"]
            ),
            "base_annualized_volatility": float(base_row["annualized_volatility"]),
            "managed_annualized_volatility": float(
                managed_row["annualized_volatility"]
            ),
            "volatility_reduction": float(volatility_reduction),
            "base_max_drawdown": float(base_row["max_drawdown"]),
            "managed_max_drawdown": float(managed_row["max_drawdown"]),
            "drawdown_reduction": float(drawdown_reduction),
        },
        "portfolio_constraints": {
            "target_net_exposure": 0.0,
            "gross_exposure_limit": float(config["gross_limit"]),
            "single_etf_absolute_limit": float(config["position_limit"]),
        },
        "historical_test_status": (
            "already inspected; no new overlay performance reported"
        ),
        "unsupported_claims": [
            "futures contracts",
            "VIX-implied volatility",
            "positive historical out-of-sample performance",
        ],
    }
    (report_dir / "project_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "performance": performance,
        "diagnostics": diagnostics,
        "checks": checks,
        "metrics": metrics,
    }


if __name__ == "__main__":
    output = run_analysis()
    print(output["performance"].to_string(index=False))
    print("\nVerified project metrics")
    print(json.dumps(output["metrics"], indent=2))

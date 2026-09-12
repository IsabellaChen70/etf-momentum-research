"""Run the end-to-end ETF research pipeline."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from etf_research.pipeline import run_pipeline  # noqa: E402


if __name__ == "__main__":
    output = run_pipeline(
        PROJECT_ROOT,
        PROJECT_ROOT / "configs" / "research_pipeline.json",
    )
    print(f"Selected model: {output['selected_model']}")
    print(f"Selected risk aversion: {output['selected_risk_aversion']:g}")
    print(f"Selected strategy: {output['selected_strategy']}")
    print("\nValidation model ranking")
    print(output["model_ranking"].to_string(index=False))
    print("\nValidation risk aversion")
    print(
        output["risk_search"][
            ["validation_rank", "risk_aversion", "annualized_return", "sharpe", "selected"]
        ].to_string(index=False)
    )
    print("\nValidation strategy selection")
    print(output["strategy_selection"].to_string(index=False))
    print("\nHistorical test performance")
    print(output["test_performance"].to_string(index=False))
    print("\nPipeline checks")
    print(output["checks"].to_string(index=False))

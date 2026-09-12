# Evaluation policy

## Current evidence

The period from June 2025 through June 2026 has already been inspected during
model and portfolio development. It remains useful as a historical
out-of-sample evaluation, but it should not be described as a fresh blind test
for later revisions.

## Selection rules

- Model families are ranked using the June 2024 through May 2025 validation period.
- Hyperparameters use purged rolling windows that end before each prediction month.
- Portfolio risk aversion is selected on validation returns after modeled costs.
- The deployment strategy, including the option to remain in cash, is selected on validation.
- A model-based historical test would use only the frozen model family and risk-aversion setting.

## Future holdout

The next available observation after June 29, 2026 begins a new locked holdout.
No parameter, feature, or portfolio rule should be changed in response to that
holdout. Changes require a new version and a later evaluation period.

Until enough future observations accumulate, paper-trading results should be
reported separately from the historical research.

## Return definition

The current bundle supports split-adjusted price returns. It does not provide a
complete dividend-adjusted total-return series for every ETF. Results are
therefore labeled as price-return research.

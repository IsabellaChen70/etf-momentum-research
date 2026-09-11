"""Forward-safe comparison of equal-weight, regression, and tree models.

This analysis extends the verified rolling LASSO framework. All candidates
use the same 14 transformed signals, the same five-day execution-aligned
return label, and the same monthly 3-month train / 1-month validation /
1-month test folds.  Hyperparameters are selected within each fold using
validation MSE.  The final model family is selected only from the established
overall backtest validation period; the outer test period is reporting only.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
import json
import warnings

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_DIR / "results" / "modeling" / "comparison"
TABLE_DIR = RESULT_DIR / "tables"
PLOT_DIR = RESULT_DIR / "plots"

from src import rolling_lasso as lasso_framework


RANDOM_STATE = 20260826
MODEL_ORDER = (
    "equal_weight",
    "lasso",
    "ridge",
    "elastic_net",
    "random_forest",
    "gradient_boosting",
)
MODEL_LABELS = {
    "equal_weight": "Equal Weight",
    "lasso": "LASSO",
    "ridge": "Ridge",
    "elastic_net": "Elastic Net",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}


def parameter_grid(model_name: str) -> list[dict[str, object]]:
    """Return a compact hyperparameter grid for each model family."""
    if model_name == "equal_weight":
        return [{}]
    if model_name == "lasso":
        # Preserve the rolling LASSO candidate set so the inherited model is a
        # true baseline rather than a slightly different model.
        return [{"alpha": float(x)} for x in lasso_framework.LASSO_ALPHAS]
    if model_name == "ridge":
        return [{"alpha": float(x)} for x in np.logspace(-4, 3, 8)]
    if model_name == "elastic_net":
        return [
            {"alpha": float(alpha), "l1_ratio": float(l1_ratio)}
            for alpha, l1_ratio in product(
                np.logspace(-6, -2, 5), (0.2, 0.5, 0.8)
            )
        ]
    if model_name == "random_forest":
        return [
            {
                "n_estimators": 100,
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "max_features": "sqrt",
            }
            for depth, leaf in product((4, 8), (10, 30))
        ]
    if model_name == "gradient_boosting":
        return [
            {
                "n_estimators": estimators,
                "learning_rate": rate,
                "max_depth": depth,
                "min_samples_leaf": 20,
            }
            for estimators, rate, depth in (
                (100, 0.03, 2),
                (100, 0.05, 2),
                (100, 0.05, 3),
                (150, 0.03, 2),
                (150, 0.05, 2),
                (150, 0.05, 3),
            )
        ]
    raise KeyError(model_name)


def make_model(model_name: str, params: dict[str, object]):
    """Construct one deterministic candidate model."""
    if model_name == "lasso":
        return Lasso(
            fit_intercept=True,
            max_iter=50_000,
            tol=1e-7,
            **params,
        )
    if model_name == "ridge":
        return Ridge(fit_intercept=True, **params)
    if model_name == "elastic_net":
        return ElasticNet(
            fit_intercept=True,
            max_iter=50_000,
            tol=1e-7,
            **params,
        )
    if model_name == "random_forest":
        return RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )
    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            loss="squared_error",
            **params,
        )
    raise KeyError(model_name)


def equal_weight_score(x: np.ndarray) -> np.ndarray:
    """Give each transformed signal exactly the same relative contribution."""
    return np.mean(x, axis=1, keepdims=True)


def fit_and_predict(
    model_name: str,
    params: dict[str, object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
) -> tuple[np.ndarray, object]:
    """Fit one model, including return-scale calibration for equal weights."""
    if model_name == "equal_weight":
        model = LinearRegression(fit_intercept=True)
        model.fit(equal_weight_score(x_train), y_train)
        return model.predict(equal_weight_score(x_predict)), model

    model = make_model(model_name, params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x_train, y_train)
    return model.predict(x_predict), model


def feature_strength(
    model_name: str,
    model: object,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract a comparable nonnegative feature-strength summary."""
    if model_name == "equal_weight":
        common_slope = float(np.ravel(model.coef_)[0]) / len(feature_names)
        values = np.repeat(abs(common_slope), len(feature_names))
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    else:
        values = np.repeat(np.nan, len(feature_names))
    return pd.DataFrame({"feature": feature_names, "feature_strength": values})


def fit_monthly_model(
    model_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_names: list[str],
    test_month: pd.Period,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Tune one model on the fold validation month and predict its test month."""
    x_train = train[feature_names].to_numpy(dtype=float)
    y_train = train["future_5d_return"].to_numpy(dtype=float)
    x_validation = validation[feature_names].to_numpy(dtype=float)
    y_validation = validation["future_5d_return"].to_numpy(dtype=float)

    grid_rows: list[dict[str, object]] = []
    for candidate_id, params in enumerate(parameter_grid(model_name)):
        validation_prediction, _ = fit_and_predict(
            model_name,
            params,
            x_train,
            y_train,
            x_validation,
        )
        grid_rows.append(
            {
                "test_month": str(test_month),
                "model": model_name,
                "candidate_id": candidate_id,
                "parameters": json.dumps(params, sort_keys=True),
                "validation_mse": float(
                    mean_squared_error(y_validation, validation_prediction)
                ),
                "validation_mae": float(
                    mean_absolute_error(y_validation, validation_prediction)
                ),
            }
        )

    grid = pd.DataFrame(grid_rows)
    if model_name == "lasso":
        # Prefer the more regularized alpha when validation MSE ties exactly.
        # This keeps the inherited baseline numerically reproducible.
        grid["regularization_tie_break"] = grid["parameters"].map(
            lambda value: float(json.loads(value)["alpha"])
        )
        grid = grid.sort_values(
            ["validation_mse", "regularization_tie_break"],
            ascending=[True, False],
        )
    else:
        grid["regularization_tie_break"] = np.nan
        grid = grid.sort_values(
            ["validation_mse", "validation_mae", "candidate_id"],
            ascending=[True, True, True],
        )
    best = grid.iloc[0]
    selected_params = json.loads(best["parameters"])

    development = pd.concat([train, validation], ignore_index=True)
    x_development = development[feature_names].to_numpy(dtype=float)
    y_development = development["future_5d_return"].to_numpy(dtype=float)
    test_prediction, final_model = fit_and_predict(
        model_name,
        selected_params,
        x_development,
        y_development,
        test[feature_names].to_numpy(dtype=float),
    )

    prediction = test[
        [
            "date",
            "symbol",
            "entry_date",
            "label_end_date",
            "sample",
            "future_5d_return",
        ]
    ].copy()
    prediction["model"] = model_name
    prediction["predicted_5d_return"] = test_prediction
    prediction["test_month"] = str(test_month)

    importance = feature_strength(model_name, final_model, feature_names)
    importance["model"] = model_name
    importance["test_month"] = str(test_month)

    fold = {
        "test_month": str(test_month),
        "model": model_name,
        "sample": prediction["sample"].dropna().iloc[0]
        if prediction["sample"].notna().any()
        else pd.NA,
        "train_start": train["date"].min(),
        "train_end": train["date"].max(),
        "validation_start": validation["date"].min(),
        "validation_end": validation["date"].max(),
        "test_start": test["date"].min(),
        "test_end": test["date"].max(),
        "train_observations": len(train),
        "validation_observations": len(validation),
        "test_observations": len(test),
        "selected_parameters": best["parameters"],
        "selected_validation_mse": float(best["validation_mse"]),
        "selected_validation_mae": float(best["validation_mae"]),
    }
    return prediction, importance, fold, grid


def run_all_models(
    panel: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run every model over the exact same feasible monthly folds."""
    complete_dates = panel.loc[panel[feature_names].notna().all(axis=1), "date"]
    if complete_dates.empty:
        raise ValueError("No dates have complete normalized features")
    first_month = complete_dates.min().to_period("M") + 4
    last_month = panel["date"].max().to_period("M")

    predictions: list[pd.DataFrame] = []
    importance: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    grids: list[pd.DataFrame] = []

    for test_month in pd.period_range(first_month, last_month, freq="M"):
        train, validation, test = lasso_framework.fold_rows(panel, test_month, feature_names)
        if len(train) < 500 or len(validation) < 100 or test.empty:
            continue
        for model_name in MODEL_ORDER:
            prediction, strength, fold, grid = fit_monthly_model(
                model_name,
                train,
                validation,
                test,
                feature_names,
                test_month,
            )
            predictions.append(prediction)
            importance.append(strength)
            folds.append(fold)
            grids.append(grid)

    if not predictions:
        raise ValueError("No complete walk-forward folds were available")
    return (
        pd.concat(predictions, ignore_index=True),
        pd.concat(importance, ignore_index=True),
        pd.DataFrame(folds),
        pd.concat(grids, ignore_index=True),
    )


def prediction_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calculate pooled prediction metrics for every model and outer sample."""
    rows: list[dict[str, object]] = []
    for model_name in MODEL_ORDER:
        for sample in lasso_framework.SAMPLE_ORDER:
            model_predictions = predictions.loc[predictions["model"].eq(model_name)]
            data = lasso_framework.clean_outer_sample_predictions(model_predictions, sample)
            if data.empty:
                continue
            actual = data["future_5d_return"].to_numpy(dtype=float)
            predicted = data["predicted_5d_return"].to_numpy(dtype=float)
            rows.append(
                {
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "sample": sample,
                    "observations": len(data),
                    "mse": float(mean_squared_error(actual, predicted)),
                    "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                    "mae": float(mean_absolute_error(actual, predicted)),
                    "correlation": float(pd.Series(actual).corr(pd.Series(predicted))),
                    "direction_accuracy": float(
                        np.mean(np.sign(actual) == np.sign(predicted))
                    ),
                    "mean_actual_return": float(np.mean(actual)),
                    "mean_predicted_return": float(np.mean(predicted)),
                }
            )
    return pd.DataFrame(rows)


def validation_ranking(performance: pd.DataFrame) -> pd.DataFrame:
    """Rank model families only on the established overall validation period."""
    ranking = performance.loc[performance["sample"].eq("validation")].copy()
    ranking = ranking.sort_values(["mse", "mae", "model"], ascending=True)
    ranking.insert(0, "validation_rank", np.arange(1, len(ranking) + 1))
    ranking["selected_model"] = False
    ranking.loc[ranking.index[0], "selected_model"] = True
    return ranking.reset_index(drop=True)


def model_backtests(
    prices: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[
    dict[str, dict[str, pd.DataFrame]],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    """Backtest every model with the same 150/50 five-cohort implementation."""
    all_results: dict[str, dict[str, pd.DataFrame]] = {}
    metrics: list[dict[str, object]] = []
    active_targets: list[pd.DataFrame] = []
    all_targets: dict[str, pd.DataFrame] = {}

    for model_name in MODEL_ORDER:
        model_predictions = predictions.loc[predictions["model"].eq(model_name)]
        _, targets = lasso_framework.predictions_to_targets(prices, model_predictions)
        targets = lasso_framework.purge_incomplete_sample_targets(prices, targets)
        all_targets[model_name] = targets
        results = lasso_framework.backtest_outer_samples(prices, targets)
        all_results[model_name] = results

        active = targets.loc[targets.abs().sum(axis=1).gt(0)].copy()
        active.insert(0, "model", model_name)
        active.insert(1, "date", active.index)
        active_targets.append(active.reset_index(drop=True))

        for sample in lasso_framework.SAMPLE_ORDER:
            row = lasso_framework.portfolio_metrics(results[sample], sample)
            daily = results[sample]["portfolio_return"].fillna(0.0)
            annualized_volatility = float(
                daily.std(ddof=1) * np.sqrt(lasso_framework.PERIODS_PER_YEAR)
            )
            row.update(
                {
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "annualized_volatility": annualized_volatility,
                    "sharpe": float(
                        daily.mean()
                        / daily.std(ddof=1)
                        * np.sqrt(lasso_framework.PERIODS_PER_YEAR)
                    )
                    if daily.std(ddof=1) > 0
                    else np.nan,
                }
            )
            metrics.append(row)

    return (
        all_results,
        pd.DataFrame(metrics),
        pd.concat(active_targets, ignore_index=True),
        all_targets,
    )


def plot_validation_metrics(ranking: pd.DataFrame, output: Path) -> None:
    """Show the primary model-selection evidence."""
    ordered = ranking.sort_values("mse", ascending=False)
    colors = ["#54A24B" if selected else "#4C78A8" for selected in ordered["selected_model"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].barh(ordered["model_label"], ordered["mse"], color=colors)
    axes[0].set_xlabel("Overall validation MSE (lower is better)")
    axes[0].set_title("Primary selection metric")
    axes[1].barh(ordered["model_label"], ordered["mae"], color=colors)
    axes[1].set_xlabel("Overall validation MAE (lower is better)")
    axes[1].set_title("Secondary metric")
    for ax, metric in zip(axes, ("mse", "mae")):
        maximum = float(ordered[metric].max())
        ax.set_xlim(0.0, maximum * 1.16)
        for position, value in enumerate(ordered[metric]):
            ax.text(
                float(value) + maximum * 0.012,
                position,
                f"{value:.6f}",
                va="center",
                fontsize=8,
            )
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Walk-forward model comparison on the validation set")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all_portfolios(
    backtests: dict[str, dict[str, pd.DataFrame]],
    output: Path,
) -> None:
    """Compare all candidate portfolios in validation and historical test samples."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharey=False)
    for ax, sample in zip(axes, ("validation", "test")):
        for model_name in MODEL_ORDER:
            result = backtests[model_name][sample]
            ax.plot(
                result.index,
                result["portfolio_value"],
                linewidth=1.25,
                label=MODEL_LABELS[model_name],
            )
        ax.axhline(1.0, color="0.45", linewidth=0.8)
        title = "Historical test, descriptive only" if sample == "test" else "Validation"
        ax.set_title(title)
        ax.set_ylabel("Portfolio value")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Date")
    axes[0].legend(ncol=3, fontsize=8, loc="best")
    fig.suptitle("Portfolio comparison by independent outer sample")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_winner_portfolio(
    winner: str,
    backtests: dict[str, dict[str, pd.DataFrame]],
    output: Path,
) -> None:
    """Plot the validation-selected winner in all three outer samples."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5))
    for ax, sample in zip(axes, lasso_framework.SAMPLE_ORDER):
        result = backtests[winner][sample]
        ax.plot(result.index, result["portfolio_value"], color="#4C78A8", linewidth=1.4)
        ax.axhline(1.0, color="0.45", linewidth=0.8)
        ax.set_ylabel("Value")
        ax.set_title(
            f"{sample.title()} (ending value {result['portfolio_value'].iloc[-1]:.3f})"
        )
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Date")
    fig.suptitle(f"Validation-selected winner: {MODEL_LABELS[winner]}")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_winner_feature_strength(
    winner: str,
    importance: pd.DataFrame,
    output: Path,
) -> None:
    """Summarize which transformed signals matter most to the winning family."""
    data = (
        importance.loc[importance["model"].eq(winner)]
        .groupby("feature")["feature_strength"]
        .mean()
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(data.index, data.values, color="#4C78A8")
    ax.set_xlabel("Mean monthly feature strength")
    ax.set_title(f"{MODEL_LABELS[winner]} feature importance")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def audit_results(
    prices: pd.DataFrame,
    panel: pd.DataFrame,
    feature_names: list[str],
    predictions: pd.DataFrame,
    folds: pd.DataFrame,
    grids: pd.DataFrame,
    performance: pd.DataFrame,
    ranking: pd.DataFrame,
    backtests: dict[str, dict[str, pd.DataFrame]],
    model_targets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Independently enforce fairness, chronology, selection, and P&L rules."""
    checks: list[dict[str, str]] = []

    expected_models = set(MODEL_ORDER)
    if set(predictions["model"].unique()) != expected_models:
        raise AssertionError("A required model is missing predictions")
    counts = predictions.groupby("model").size()
    if counts.nunique() != 1:
        raise AssertionError("Models were not evaluated on identical rows")
    checks.append({"check": "identical_prediction_universe", "status": "pass", "detail": f"{counts.iloc[0]:,} rows/model"})

    keys = ["date", "symbol", "test_month"]
    reference = predictions.loc[predictions["model"].eq(MODEL_ORDER[0]), keys].reset_index(drop=True)
    for model_name in MODEL_ORDER[1:]:
        candidate = predictions.loc[predictions["model"].eq(model_name), keys].reset_index(drop=True)
        if not reference.equals(candidate):
            raise AssertionError(f"{model_name} used a different prediction universe")
    checks.append({"check": "same_dates_symbols_and_targets", "status": "pass", "detail": "all six models"})

    if not folds["train_end"].lt(folds["validation_start"]).all():
        raise AssertionError("Training overlaps validation")
    if not folds["validation_end"].lt(folds["test_start"]).all():
        raise AssertionError("Validation overlaps test")
    checks.append({"check": "chronological_monthly_folds", "status": "pass", "detail": "3m/1m/1m"})

    for (test_month, model_name), group in grids.groupby(["test_month", "model"]):
        selected = folds.loc[
            folds["test_month"].eq(test_month) & folds["model"].eq(model_name)
        ].iloc[0]
        if model_name == "lasso":
            best = group.sort_values(
                ["validation_mse", "regularization_tie_break"],
                ascending=[True, False],
            ).iloc[0]
        else:
            best = group.sort_values(
                ["validation_mse", "validation_mae", "candidate_id"]
            ).iloc[0]
        if not np.isclose(selected["selected_validation_mse"], best["validation_mse"], atol=1e-15):
            raise AssertionError("A fold did not select minimum validation MSE")
    checks.append({"check": "fold_hyperparameters_minimize_validation_mse", "status": "pass", "detail": f"{len(folds)} model-folds"})

    validation = performance.loc[performance["sample"].eq("validation")]
    expected_winner = validation.sort_values(["mse", "mae", "model"]).iloc[0]["model"]
    reported_winner = ranking.loc[ranking["selected_model"], "model"].iloc[0]
    if reported_winner != expected_winner:
        raise AssertionError("Overall winner was not selected by validation MSE")
    checks.append({"check": "winner_uses_overall_validation_only", "status": "pass", "detail": reported_winner})

    validation_rows = predictions.loc[
        predictions["sample"].eq("validation")
        & predictions["future_5d_return"].notna()
    ]
    expected_count = int(
        validation_rows.loc[
            validation_rows["model"].eq(MODEL_ORDER[0])
            & validation_rows["label_end_date"].lt(lasso_framework.TEST_START)
        ].shape[0]
    )
    used_counts = performance.loc[
        performance["sample"].eq("validation"), "observations"
    ]
    if not used_counts.eq(expected_count).all():
        raise AssertionError("Outer validation MSE includes labels ending in test")
    checks.append({"check": "overall_validation_labels_end_before_test", "status": "pass", "detail": "boundary labels excluded"})

    target, entry, label_end = lasso_framework.calculate_execution_aligned_target(prices)
    probe = predictions.dropna(subset=["future_5d_return"]).iloc[len(predictions) // 3]
    expected = target.loc[probe["date"], probe["symbol"]]
    if not np.isclose(expected, probe["future_5d_return"], atol=1e-15):
        raise AssertionError("Saved target is not price[t+6]/price[t+1]-1")
    if entry.loc[probe["date"]] != probe["entry_date"] or label_end.loc[probe["date"]] != probe["label_end_date"]:
        raise AssertionError("Execution-aligned dates are incorrect")
    checks.append({"check": "one_day_lag_five_day_target", "status": "pass", "detail": "price[t+6]/price[t+1]-1"})

    unavailable_dates = (
        lasso_framework.EXECUTION_LAG_DAYS + lasso_framework.HOLDING_DAYS
    )
    for model_name, targets in model_targets.items():
        long_total = targets.clip(lower=0).sum(axis=1)
        short_total = -targets.clip(upper=0).sum(axis=1)
        active = long_total.gt(0) & short_total.gt(0)
        if not np.allclose(
            long_total.loc[active], lasso_framework.LONG_GROSS, atol=1e-10
        ):
            raise AssertionError(f"{model_name} long targets do not total 150%")
        if not np.allclose(
            short_total.loc[active], lasso_framework.SHORT_GROSS, atol=1e-10
        ):
            raise AssertionError(f"{model_name} short targets do not total 50%")
        for start, end in (
            (lasso_framework.TRAIN_START, lasso_framework.VALIDATION_START),
            (lasso_framework.VALIDATION_START, lasso_framework.TEST_START),
            (lasso_framework.TEST_START, None),
        ):
            mask = targets.index >= start
            if end is not None:
                mask &= targets.index < end
            dates = targets.index[mask]
            if targets.loc[dates[-unavailable_dates:]].abs().to_numpy().max() > 1e-12:
                raise AssertionError(f"{model_name} has an incomplete boundary cohort")
    checks.append({"check": "all_models_exact_150_50_and_complete_holds", "status": "pass", "detail": "six models"})

    for model_name in MODEL_ORDER:
        for sample in lasso_framework.SAMPLE_ORDER:
            result = backtests[model_name][sample]
            independent_value = (1.0 + result["portfolio_return"].fillna(0.0)).cumprod()
            if not np.allclose(independent_value, result["portfolio_value"], atol=1e-12):
                raise AssertionError("Portfolio path does not match saved daily returns")
            if not np.isclose(result["portfolio_value"].iloc[0], 1.0):
                raise AssertionError("An outer sample did not restart from cash")
    checks.append({"check": "independent_portfolio_recalculation", "status": "pass", "detail": "18 model-samples"})

    numerical = predictions[["future_5d_return", "predicted_5d_return"]].dropna().to_numpy()
    if not np.isfinite(numerical).all():
        raise AssertionError("Predictions contain missing or infinite numerical values")
    checks.append({"check": "finite_predictions", "status": "pass", "detail": f"{len(numerical):,} rows"})

    return pd.DataFrame(checks)


def write_summary(
    ranking: pd.DataFrame,
    prediction: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> Path:
    """Write a short plain-text model-comparison summary."""
    winner = ranking.loc[ranking["selected_model"], "model"].iloc[0]
    winner_prediction = prediction.loc[prediction["model"].eq(winner)]
    winner_portfolio = portfolio.loc[portfolio["model"].eq(winner)]
    lines = [
        "MODEL COMPARISON RESULTS",
        "",
        "Method",
        "- Equal Weight averages all normalized signals, then calibrates one common slope on training data.",
        "- Regression candidates: LASSO, Ridge, Elastic Net.",
        "- Tree candidates: Random Forest, Gradient Boosting.",
        "- Every model uses identical forward-safe monthly folds and five-day targets.",
        "- Fold hyperparameters minimize prior validation MSE.",
        "- The final family is selected by overall validation MSE; MAE breaks ties.",
        "",
        f"Selected model: {MODEL_LABELS[winner]}",
        "",
        "Validation ranking",
        ranking[["validation_rank", "model_label", "mse", "mae", "correlation"]].to_string(index=False),
        "",
        "Selected model prediction performance",
        winner_prediction[["sample", "mse", "mae", "correlation", "direction_accuracy"]].to_string(index=False),
        "",
        "Selected model portfolio performance",
        winner_portfolio[["sample", "annualized_return", "annualized_volatility", "sharpe", "max_drawdown", "ending_value"]].to_string(index=False),
        "",
        "Reminder: model selection uses validation metrics only. The already-inspected test period is descriptive.",
    ]
    path = RESULT_DIR / "research_summary.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_analysis() -> dict[str, object]:
    """Run the model comparison and save its tables and figures."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    prices = lasso_framework.load_close_prices()
    raw_signals = lasso_framework.calculate_raw_signals(prices)
    processed = lasso_framework.preprocess_all_signals(raw_signals)
    panel, feature_names = lasso_framework.build_model_panel(prices, processed)

    predictions, importance, folds, grids = run_all_models(panel, feature_names)
    prediction_summary = prediction_performance(predictions)
    ranking = validation_ranking(prediction_summary)
    winner = ranking.loc[ranking["selected_model"], "model"].iloc[0]

    backtests, portfolio_summary, active_targets, model_targets = model_backtests(
        prices, predictions
    )
    audits = audit_results(
        prices,
        panel,
        feature_names,
        predictions,
        folds,
        grids,
        prediction_summary,
        ranking,
        backtests,
        model_targets,
    )

    table_paths = {
        "ranking": TABLE_DIR / "validation_model_ranking.csv",
        "prediction": TABLE_DIR / "model_prediction_performance.csv",
        "portfolio": TABLE_DIR / "model_portfolio_performance.csv",
        "folds": TABLE_DIR / "monthly_model_folds.csv",
        "grids": TABLE_DIR / "monthly_hyperparameter_grid.csv",
        "predictions": TABLE_DIR / "walk_forward_model_predictions.csv",
        "importance": TABLE_DIR / "monthly_feature_strength.csv",
        "targets": TABLE_DIR / "active_daily_target_weights.csv",
        "audits": TABLE_DIR / "validation_checks.csv",
    }
    ranking.to_csv(table_paths["ranking"], index=False)
    prediction_summary.to_csv(table_paths["prediction"], index=False)
    portfolio_summary.to_csv(table_paths["portfolio"], index=False)
    folds.to_csv(table_paths["folds"], index=False)
    grids.to_csv(table_paths["grids"], index=False)
    predictions.to_csv(table_paths["predictions"], index=False)
    importance.to_csv(table_paths["importance"], index=False)
    active_targets.to_csv(table_paths["targets"], index=False)
    audits.to_csv(table_paths["audits"], index=False)

    settings = {
        "models": list(MODEL_ORDER),
        "model_labels": MODEL_LABELS,
        "feature_names": feature_names,
        "selection_metric": "overall validation MSE",
        "tie_breaker": "overall validation MAE",
        "random_state": RANDOM_STATE,
        "train_months": lasso_framework.TRAIN_MONTHS,
        "validation_months": lasso_framework.VALIDATION_MONTHS,
        "prediction_months": lasso_framework.TEST_MONTHS,
        "holding_days": lasso_framework.HOLDING_DAYS,
        "execution_lag_days": lasso_framework.EXECUTION_LAG_DAYS,
    }
    settings_path = TABLE_DIR / "analysis_settings.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    plot_paths = {
        "metrics": PLOT_DIR / "validation_mse_mae_comparison.png",
        "portfolios": PLOT_DIR / "all_model_portfolio_comparison.png",
        "winner": PLOT_DIR / "selected_model_portfolio.png",
        "importance": PLOT_DIR / "selected_model_feature_importance.png",
    }
    plot_validation_metrics(ranking, plot_paths["metrics"])
    plot_all_portfolios(backtests, plot_paths["portfolios"])
    plot_winner_portfolio(winner, backtests, plot_paths["winner"])
    plot_winner_feature_strength(winner, importance, plot_paths["importance"])

    write_summary(ranking, prediction_summary, portfolio_summary)
    return {
        "winner": winner,
        "ranking": ranking,
        "prediction_summary": prediction_summary,
        "portfolio_summary": portfolio_summary,
        "audits": audits,
    }


if __name__ == "__main__":
    output = run_analysis()
    print("\nValidation model ranking")
    print(
        output["ranking"][["validation_rank", "model_label", "mse", "mae", "correlation"]]
        .to_string(index=False)
    )
    print(f"\nSelected model: {MODEL_LABELS[output['winner']]}")
    print("\nSelected model prediction performance")
    print(
        output["prediction_summary"]
        .loc[output["prediction_summary"]["model"].eq(output["winner"])]
        .to_string(index=False)
    )
    print("\nSelected model portfolio performance")
    print(
        output["portfolio_summary"]
        .loc[output["portfolio_summary"]["model"].eq(output["winner"])]
        .to_string(index=False)
    )
    print("\nValidation checks")
    print(output["audits"].to_string(index=False))

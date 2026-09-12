"""Purged monthly walk-forward models for cross-sectional ETF returns."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


TARGET = "future_5d_relative_return"
MIN_FORECAST_DISPERSION = 1e-10
MODEL_LABELS = {
    "lasso": "LASSO",
    "ridge": "Ridge",
    "elastic_net": "Elastic Net",
    "hist_gradient_boosting": "Histogram Gradient Boosting",
}


def parameter_grid(model_name: str) -> list[dict[str, float | int]]:
    """Return a compact grid suitable for repeated monthly estimation."""
    if model_name == "lasso":
        return [{"alpha": value} for value in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)]
    if model_name == "ridge":
        return [{"alpha": value} for value in (0.1, 1.0, 10.0, 100.0)]
    if model_name == "elastic_net":
        return [
            {"alpha": alpha, "l1_ratio": ratio}
            for alpha in (1e-6, 1e-5, 1e-4)
            for ratio in (0.1, 0.5)
        ]
    if model_name == "hist_gradient_boosting":
        return [
            {"max_leaf_nodes": 7, "l2_regularization": 1.0},
            {"max_leaf_nodes": 15, "l2_regularization": 10.0},
        ]
    raise ValueError(f"Unknown model: {model_name}")


def make_model(model_name: str, parameters: dict[str, float | int]) -> object:
    """Create one deterministic regression candidate."""
    if model_name == "lasso":
        return Lasso(
            alpha=float(parameters["alpha"]),
            fit_intercept=True,
            max_iter=50_000,
            tol=1e-7,
        )
    if model_name == "ridge":
        return Ridge(alpha=float(parameters["alpha"]), fit_intercept=True)
    if model_name == "elastic_net":
        return ElasticNet(
            alpha=float(parameters["alpha"]),
            l1_ratio=float(parameters["l1_ratio"]),
            fit_intercept=True,
            max_iter=50_000,
            tol=1e-7,
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            l2_regularization=float(parameters["l2_regularization"]),
            random_state=42,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _month_end(period: pd.Period) -> pd.Timestamp:
    return period.end_time.normalize()


def fold_rows(
    panel: pd.DataFrame,
    feature_names: list[str],
    prediction_month: pd.Period,
    training_months: int,
    validation_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return chronological rows with five-day labels purged at both boundaries."""
    month = panel["date"].dt.to_period("M")
    feature_complete = panel[feature_names].notna().all(axis=1)
    label_complete = panel[TARGET].notna() & panel["label_end_date"].notna()

    validation_periods = [
        prediction_month - offset
        for offset in range(validation_months, 0, -1)
    ]
    training_periods = [
        prediction_month - validation_months - offset
        for offset in range(training_months, 0, -1)
    ]
    train_end = _month_end(training_periods[-1])
    validation_end = _month_end(validation_periods[-1])

    train_mask = (
        month.isin(training_periods)
        & feature_complete
        & label_complete
        & panel["label_end_date"].le(train_end)
    )
    validation_mask = (
        month.isin(validation_periods)
        & feature_complete
        & label_complete
        & panel["label_end_date"].le(validation_end)
    )
    prediction_mask = month.eq(prediction_month) & feature_complete
    return (
        panel.loc[train_mask].copy(),
        panel.loc[validation_mask].copy(),
        panel.loc[prediction_mask].copy(),
    )


def _fit_predict(
    model_name: str,
    parameters: dict[str, float | int],
    train: pd.DataFrame,
    predict: pd.DataFrame,
    feature_names: list[str],
) -> tuple[np.ndarray, object]:
    model = make_model(model_name, parameters)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(train[feature_names], train[TARGET])
    return np.asarray(model.predict(predict[feature_names]), dtype=float), model


def monthly_walk_forward(
    panel: pd.DataFrame,
    feature_names: list[str],
    model_name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    training_months: int = 12,
    validation_months: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune on prior months, refit, and predict each requested month."""
    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    grids: list[pd.DataFrame] = []
    first_month = start.to_period("M")
    last_month = end.to_period("M")

    for prediction_month in pd.period_range(first_month, last_month, freq="M"):
        train, validation, prediction = fold_rows(
            panel,
            feature_names,
            prediction_month,
            training_months,
            validation_months,
        )
        prediction = prediction.loc[
            prediction["date"].between(start, end, inclusive="both")
        ]
        if train.empty or validation.empty or prediction.empty:
            continue

        rows = []
        for candidate_id, parameters in enumerate(parameter_grid(model_name)):
            forecast, _ = _fit_predict(
                model_name,
                parameters,
                train,
                validation,
                feature_names,
            )
            rows.append(
                {
                    "prediction_month": str(prediction_month),
                    "model": model_name,
                    "candidate_id": candidate_id,
                    "parameters": json.dumps(parameters, sort_keys=True),
                    "validation_mse": float(mean_squared_error(validation[TARGET], forecast)),
                    "validation_mae": float(mean_absolute_error(validation[TARGET], forecast)),
                }
            )
        grid = pd.DataFrame(rows).sort_values(
            ["validation_mse", "validation_mae", "candidate_id"],
            ascending=True,
        )
        best_parameters = json.loads(grid.iloc[0]["parameters"])
        development = pd.concat([train, validation], ignore_index=True)
        forecast, model = _fit_predict(
            model_name,
            best_parameters,
            development,
            prediction,
            feature_names,
        )

        output = prediction[
            [
                "date",
                "symbol",
                "entry_date",
                "label_end_date",
                "future_5d_return",
                TARGET,
            ]
        ].copy()
        output["model"] = model_name
        output["predicted_5d_relative_return"] = forecast
        output["prediction_month"] = str(prediction_month)
        predictions.append(output)
        grids.append(grid)

        nonzero = np.nan
        if hasattr(model, "coef_"):
            nonzero = int(np.count_nonzero(np.abs(np.asarray(model.coef_)) > 1e-12))
        folds.append(
            {
                "prediction_month": str(prediction_month),
                "model": model_name,
                "train_start": train["date"].min(),
                "train_end": train["date"].max(),
                "validation_start": validation["date"].min(),
                "validation_end": validation["date"].max(),
                "prediction_start": prediction["date"].min(),
                "prediction_end": prediction["date"].max(),
                "train_observations": len(train),
                "validation_observations": len(validation),
                "prediction_observations": len(prediction),
                "selected_parameters": json.dumps(best_parameters, sort_keys=True),
                "nonzero_coefficients": nonzero,
            }
        )

    if not predictions:
        raise ValueError(f"No walk-forward predictions were generated for {model_name}")
    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(folds),
        pd.concat(grids, ignore_index=True),
    )


def prediction_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize cross-sectional prediction quality for model selection."""
    rows = []
    for model_name, group in predictions.groupby("model", sort=False):
        clean = group.dropna(subset=[TARGET, "predicted_5d_relative_return"]).copy()
        def daily_rank_ic(frame: pd.DataFrame) -> float:
            actual = frame[TARGET]
            forecast = frame["predicted_5d_relative_return"]
            if actual.std(ddof=1) <= MIN_FORECAST_DISPERSION:
                return np.nan
            if forecast.std(ddof=1) <= MIN_FORECAST_DISPERSION:
                return np.nan
            return float(actual.corr(forecast, method="spearman"))

        daily_ic = clean.groupby("date").apply(
            daily_rank_ic,
            include_groups=False,
        )
        dispersion = clean.groupby("date")["predicted_5d_relative_return"].std(ddof=1)
        rows.append(
            {
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "observations": len(clean),
                "mse": float(mean_squared_error(clean[TARGET], clean["predicted_5d_relative_return"])),
                "mae": float(mean_absolute_error(clean[TARGET], clean["predicted_5d_relative_return"])),
                "mean_daily_rank_ic": float(daily_ic.mean()),
                "rank_ic_information_ratio": float(
                    daily_ic.mean() / daily_ic.std(ddof=1)
                ) if daily_ic.std(ddof=1) > 0 else np.nan,
                "identical_forecast_dates": int(dispersion.le(MIN_FORECAST_DISPERSION).sum()),
                "forecast_dates": int(len(dispersion)),
                "active_forecast_share": float(dispersion.gt(MIN_FORECAST_DISPERSION).mean()),
            }
        )
    return pd.DataFrame(rows)


def select_model(validation_performance: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Select the relative-return model using validation rank IC only."""
    ranking = validation_performance.copy()
    ranking = ranking.sort_values(
        ["mean_daily_rank_ic", "mse", "active_forecast_share"],
        ascending=[False, True, False],
    ).reset_index(drop=True)
    ranking.insert(0, "validation_rank", np.arange(1, len(ranking) + 1))
    ranking["selected"] = False
    ranking.loc[0, "selected"] = True
    return str(ranking.loc[0, "model"]), ranking

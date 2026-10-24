"""Chronological splits, canonical metrics, and model comparison."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from forecasting import MODEL_NAMES, fit_and_forecast


def clean_price_series(values: pd.Series | pd.DataFrame, min_obs: int = 40) -> pd.Series:
    """Clean a single equity-price series without filling missing trading dates."""
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("Expected exactly one price column.")
        values = values.iloc[:, 0]
    y = pd.to_numeric(pd.Series(values).copy(), errors="coerce")
    y = y.replace([np.inf, -np.inf], np.nan).dropna()
    if isinstance(y.index, pd.DatetimeIndex):
        idx = pd.to_datetime(y.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        y.index = idx
        y = y[~y.index.duplicated(keep="last")].sort_index()
    else:
        y = y.reset_index(drop=True)
    y = y[y > 0].astype(float)
    if len(y) < min_obs:
        raise ValueError(f"At least {min_obs} valid positive price observations are required; found {len(y)}.")
    return y.rename(values.name or "Price")


def chronological_split(prices: pd.Series, holdout_fraction: float = 0.15) -> dict[str, pd.Series]:
    """Split into train / validation / test with equal validation and test fractions."""
    y = clean_price_series(prices)
    fraction = float(holdout_fraction)
    if not 0.05 <= fraction <= 0.25:
        raise ValueError("Validation/test fraction must be between 5% and 25% each.")
    n_holdout = max(1, int(round(len(y) * fraction)))
    train_end = len(y) - 2 * n_holdout
    val_end = len(y) - n_holdout
    if train_end < 40:
        raise ValueError(
            f"Not enough observations for this split: {len(y)} total leaves {train_end} training points. "
            "Use a longer history or reduce the validation/test fraction."
        )
    return {
        "train": y.iloc[:train_end],
        "validation": y.iloc[train_end:val_end],
        "train_validation": y.iloc[:val_end],
        "test": y.iloc[val_end:],
        "all": y,
    }


def compute_metrics(actual: pd.Series, predicted: pd.Series, origin: float, mase_scale: float) -> dict[str, float]:
    """Canonical metric implementation for aligned actual/predicted price arrays."""
    if not actual.index.equals(predicted.index):
        raise ValueError("Actual and predicted observations must have identical timestamps in the same order.")
    a, p = actual.to_numpy(dtype=float), predicted.to_numpy(dtype=float)
    if len(a) == 0 or len(a) != len(p):
        raise ValueError("Actual and prediction arrays must be non-empty and have equal length.")
    if not (np.isfinite(a).all() and np.isfinite(p).all()):
        raise ValueError("Metrics require finite actual and predicted values.")

    error = a - p
    absolute = np.abs(error)
    mae = float(absolute.mean())
    rmse = float(np.sqrt(np.mean(error**2)))
    mape = float(np.mean(absolute / np.maximum(np.abs(a), 1e-12)) * 100)
    smape = float(np.mean(2 * absolute / np.maximum(np.abs(a) + np.abs(p), 1e-12)) * 100)
    actual_moves = np.diff(np.r_[float(origin), a])
    predicted_moves = np.diff(np.r_[float(origin), p])
    direction = float((np.sign(actual_moves) == np.sign(predicted_moves)).mean() * 100)
    mase = float(mae / mase_scale) if np.isfinite(mase_scale) and mase_scale > 0 else float("nan")
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "sMAPE (%)": smape,
        "MASE": mase,
        "Direction Accuracy (%)": direction,
    }


def mase_scale(training: pd.Series) -> float:
    """Mean absolute one-step naive error from training only."""
    values = training.to_numpy(dtype=float)
    if len(values) < 2:
        return float("nan")
    return float(np.mean(np.abs(np.diff(values))))


def run_model_comparison(
    prices: pd.Series,
    interval: str = "Daily",
    test_fraction: float = 0.15,
    seasonal_period: int = 5,
    sma_window: int = 20,
    included_models: Iterable[str] | None = None,
    auto_max_order: int = 3,
    auto_max_seasonal_order: int = 1,
) -> dict[str, Any]:
    """Fit on train, select on validation, then evaluate refits on an untouched test block."""
    splits = chronological_split(prices, test_fraction)
    train, validation = splits["train"], splits["validation"]
    train_validation, test = splits["train_validation"], splits["test"]
    models = list(included_models) if included_models is not None else MODEL_NAMES.copy()
    unknown = sorted(set(models) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown model name(s): {unknown}")
    if not models:
        raise ValueError("Choose at least one model to compare.")

    rows: list[dict[str, Any]] = []
    val_predictions: dict[str, pd.Series] = {"Actual": validation.copy()}
    test_predictions: dict[str, pd.Series] = {"Actual": test.copy()}
    residuals: dict[str, np.ndarray] = {}
    validation_fits: dict[str, dict[str, Any]] = {}
    test_fits: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}

    for name in models:
        try:
            val_fit = fit_and_forecast(train, len(validation), name, seasonal_period, sma_window, interval,
                                       auto_max_order, auto_max_seasonal_order)
            test_fit = fit_and_forecast(train_validation, len(test), name, seasonal_period, sma_window, interval,
                                        auto_max_order, auto_max_seasonal_order)
            val_pred = pd.Series(val_fit["forecast"].to_numpy(), index=validation.index, name=name)
            test_pred = pd.Series(test_fit["forecast"].to_numpy(), index=test.index, name=name)
            val_metrics = compute_metrics(validation, val_pred, float(train.iloc[-1]), mase_scale(train))
            test_metrics = compute_metrics(test, test_pred, float(train_validation.iloc[-1]), mase_scale(train_validation))

            row: dict[str, Any] = {
                "Model": name,
                "ARIMA order (p,d,q)": str(val_fit["arima_order"]) if val_fit["arima_order"] is not None else "—",
                "Seasonal order (P,D,Q,m)": str(val_fit["seasonal_order"]) if val_fit["seasonal_order"] is not None else "—",
                "Validation AIC": val_fit["aic"] if val_fit["aic"] is not None else np.nan,
                "Validation AICc": val_fit["aicc"] if val_fit["aicc"] is not None else np.nan,
                "Validation BIC": val_fit["bic"] if val_fit["bic"] is not None else np.nan,
                "Test-fit ARIMA order": str(test_fit["arima_order"]) if test_fit["arima_order"] is not None else "—",
                "Test-fit seasonal order": str(test_fit["seasonal_order"]) if test_fit["seasonal_order"] is not None else "—",
                "Test-fit AIC": test_fit["aic"] if test_fit["aic"] is not None else np.nan,
                "Test-fit AICc": test_fit["aicc"] if test_fit["aicc"] is not None else np.nan,
                "Test-fit BIC": test_fit["bic"] if test_fit["bic"] is not None else np.nan,
                "Val/test fit warning count": sum(len(fit.get("diagnostics", {}).get("fit_warnings", []))
                                          + len(fit.get("diagnostics", {}).get("failed_candidates", []))
                                          for fit in (val_fit, test_fit)),
                "Parameters": test_fit["details"],
            }
            row.update({f"Validation {k}": v for k, v in val_metrics.items()})
            row.update({f"Test {k}": v for k, v in test_metrics.items()})
            rows.append(row)
            val_predictions[name] = val_pred
            test_predictions[name] = test_pred
            validation_fits[name] = val_fit
            test_fits[name] = test_fit
            residuals[name] = test_fit["residuals"]
        except Exception as exc:
            skipped[name] = f"{type(exc).__name__}: {exc}"

    if not rows:
        summary = "; ".join(f"{key}: {reason}" for key, reason in skipped.items())
        raise RuntimeError(f"No models completed successfully. {summary}")

    return {
        "comparison": pd.DataFrame(rows),
        "validation_predictions": pd.DataFrame(val_predictions),
        "test_predictions": pd.DataFrame(test_predictions),
        "validation_fits": validation_fits,
        "test_fits": test_fits,
        "residuals": residuals,
        "skipped": skipped,
        **splits,
        "seasonal_period": int(seasonal_period),
        "sma_window": int(sma_window),
        "interval": interval,
        "holdout_fraction": float(test_fraction),
        "included_models": models,
        "auto_max_order": int(auto_max_order),
        "auto_max_seasonal_order": int(auto_max_seasonal_order),
    }

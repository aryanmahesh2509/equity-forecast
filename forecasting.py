"""Forecasting candidates for the ATSA Equity Forecast Lab.

All candidates consume price-level observations and return forecasts on the
same price scale. The AR/MA/ARMA and legacy ARIMA candidates are fitted to
percentage returns, then their predicted returns are compounded to prices.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt, SimpleExpSmoothing
from statsmodels.tsa.stattools import adfuller

MODEL_NAMES = [
    "Naive (random walk)",
    "Drift",
    "Simple Moving Average",
    "Weighted Moving Average",
    "Linear Trend Regression",
    "Simple Exponential Smoothing",
    "Holt Linear Trend",
    "Holt-Winters Additive",
    "Holt-Winters Multiplicative",
    "AR(p) on returns (AICc search)",
    "MA(q) on returns (AICc search)",
    "ARMA(p,q) on returns (AICc search)",
    "Legacy ARIMA(2,0,2) on returns",
    "Auto ARIMA",
    "Auto SARIMA",
]

MODEL_DESCRIPTIONS = {
    "Naive (random walk)": "Next forecast equals the latest observed price; essential baseline.",
    "Drift": "Extends the average historical change by one step at a time.",
    "Simple Moving Average": "Uses the latest k observations' average as a constant forecast; not the MA(q) stochastic model.",
    "Weighted Moving Average": "Uses fixed weights 0.40, 0.35, 0.20, 0.05 on the latest four prices; not the MA(q) stochastic model.",
    "Linear Trend Regression": "OLS straight-line trend against observation time.",
    "Simple Exponential Smoothing": "Level-only exponential smoothing; no explicit trend or seasonality.",
    "Holt Linear Trend": "Level plus damped additive trend.",
    "Holt-Winters Additive": "Level, damped trend and additive seasonality (constant seasonal amplitude).",
    "Holt-Winters Multiplicative": "Level, damped trend and multiplicative seasonality (seasonal amplitude proportional to level).",
    "AR(p) on returns (AICc search)": "Selects AR order p from a bounded grid by AICc on training returns; forecast returns are compounded to prices.",
    "MA(q) on returns (AICc search)": "Selects MA order q from a bounded grid by AICc on training returns; forecast returns are compounded to prices.",
    "ARMA(p,q) on returns (AICc search)": "Selects p and q from a bounded grid by AICc on training returns; forecast returns are compounded to prices.",
    "Legacy ARIMA(2,0,2) on returns": "Fixed-order legacy benchmark, adapted to fixed-origin multi-step forecasts on the common price scale.",
    "Auto ARIMA": "Bounded p,d,q search using ADF-based differencing and AICc within the search.",
    "Auto SARIMA": "Bounded non-seasonal and seasonal order search; seasonality is a hypothesis to test, not assumed to help.",
}


def _positive_price_series(values: pd.Series | pd.DataFrame, min_obs: int = 40) -> pd.Series:
    """Normalize one price series without inventing prices on missing dates."""
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("Expected exactly one price column.")
        values = values.iloc[:, 0]
    y = pd.Series(values).copy()
    y = pd.to_numeric(y, errors="coerce")
    y = y.replace([np.inf, -np.inf], np.nan).dropna()

    if isinstance(y.index, pd.DatetimeIndex):
        index = pd.to_datetime(y.index)
        if index.tz is not None:
            index = index.tz_localize(None)
        y.index = index
        y = y[~y.index.duplicated(keep="last")].sort_index()
    else:
        y = y.reset_index(drop=True)

    # Non-positive values are invalid for equity prices: drop them, never fill them.
    y = y[y > 0].astype(float)
    if len(y) < min_obs:
        raise ValueError(f"At least {min_obs} valid positive price observations are required; found {len(y)}.")
    y.name = values.name if isinstance(values, pd.Series) else "Price"
    return y


def future_index(last_index: Any, steps: int, interval: str) -> pd.Index:
    """Generate approximate future observation dates; exchange holidays are not modelled."""
    if steps < 1:
        raise ValueError("Forecast horizon must be at least one observation.")
    if not isinstance(last_index, pd.Timestamp):
        return pd.RangeIndex(int(last_index) + 1, int(last_index) + steps + 1)
    if interval == "Daily":
        return pd.bdate_range(last_index + pd.offsets.BDay(1), periods=steps)
    if interval == "Weekly":
        start = last_index + pd.Timedelta(days=1)
        return pd.date_range(start=start, periods=steps, freq="W-FRI")
    return pd.date_range(start=last_index + pd.offsets.MonthEnd(1), periods=steps, freq=pd.offsets.MonthEnd())


def _compound_returns(last_price: float, returns: np.ndarray) -> np.ndarray:
    """Convert forecast simple returns to a price path."""
    r = np.asarray(returns, dtype=float)
    # A simple return below -100% is invalid for a long-only equity price.
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=-0.999)
    r = np.maximum(r, -0.999)
    return float(last_price) * np.cumprod(1.0 + r)


def _require_stationary_returns(returns: pd.Series, label: str) -> float:
    """Screen AR/MA/ARMA-on-return fits with ADF at the 5% level."""
    try:
        pvalue = float(adfuller(np.asarray(returns, dtype=float), autolag="AIC")[1])
    except Exception as exc:
        raise ValueError(f"Could not assess return stationarity with ADF: {type(exc).__name__}: {exc}") from exc
    if pvalue >= 0.05:
        raise ValueError(
            f"{label} assumes stationary returns, but the training-return ADF p-value is {pvalue:.4g} (>= 0.05). "
            "This candidate was skipped rather than fitting an ARMA model to a series with unresolved unit-root evidence."
        )
    return pvalue


def _select_d_by_adf(values: np.ndarray, max_d: int = 2) -> tuple[int, list[str]]:
    """Select the first differencing order that rejects the ADF unit-root null.

    This explicit step avoids a pmdarima edge case where its internal ADF
    regression can become singular on strongly smooth or seasonally differenced
    input. The test is performed only on the fit's training observations.
    """
    x = np.asarray(values, dtype=float)
    notes: list[str] = []
    for d in range(max_d + 1):
        try:
            pvalue = float(adfuller(x, autolag="AIC")[1])
            notes.append(f"ADF p-value after d={d}: {pvalue:.4g}")
            if pvalue < 0.05:
                return d, notes
        except Exception as exc:
            notes.append(f"ADF test failed at d={d}: {type(exc).__name__}: {exc}")
            return d, notes
        if d < max_d:
            x = np.diff(x)
    return max_d, notes


def _select_D_by_ocsb(values: np.ndarray, period: int) -> tuple[int, str | None]:
    """Select seasonal differencing with OCSB; safely fall back to D=0 on test failure."""
    try:
        from pmdarima.arima.utils import nsdiffs
        return int(nsdiffs(np.asarray(values, dtype=float), m=int(period), test="ocsb", max_D=1)), None
    except Exception as exc:
        return 0, f"OCSB seasonal differencing test failed ({type(exc).__name__}); used D=0 fallback."


def _result(
    model_name: str,
    index: pd.Index,
    forecast: np.ndarray,
    details: str,
    residuals: np.ndarray | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    order: tuple[int, int, int] | None = None,
    seasonal_order: tuple[int, int, int, int] | None = None,
    aicc: float | None = None,
    aic: float | None = None,
    bic: float | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pred = np.asarray(forecast, dtype=float).reshape(-1)
    if len(pred) != len(index):
        raise ValueError(f"{model_name} returned {len(pred)} forecasts; expected {len(index)}.")
    interval = None
    if lower is not None and upper is not None:
        lo, hi = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if len(lo) == len(pred) and len(hi) == len(pred):
            interval = pd.DataFrame({"Lower 95%": lo, "Upper 95%": hi}, index=index)
    return {
        "forecast": pd.Series(pred, index=index, name=model_name),
        "interval": interval,
        "details": details,
        "residuals": np.asarray(residuals if residuals is not None else [], dtype=float),
        "arima_order": order,
        "seasonal_order": seasonal_order,
        "aicc": aicc,
        "aic": aic,
        "bic": bic,
        "diagnostics": diagnostics or {},
    }


def fit_and_forecast(
    values: pd.Series,
    steps: int,
    model_name: str,
    seasonal_period: int = 5,
    sma_window: int = 20,
    interval: str = "Daily",
    auto_max_order: int = 3,
    auto_max_seasonal_order: int = 1,
) -> dict[str, Any]:
    """Fit one candidate and return forecast, model metadata and residuals."""
    y = _positive_price_series(values)
    if steps < 1:
        raise ValueError("Forecast horizon must be at least one observation.")
    index = future_index(y.index[-1], steps, interval)
    prices = y.to_numpy(dtype=float)

    if model_name == "Naive (random walk)":
        return _result(model_name, index, np.repeat(prices[-1], steps), MODEL_DESCRIPTIONS[model_name], np.diff(prices))

    if model_name == "Drift":
        slope = (prices[-1] - prices[0]) / max(len(prices) - 1, 1)
        return _result(model_name, index, prices[-1] + slope * np.arange(1, steps + 1),
                       f"Random walk with drift; mean change={slope:.6g} per observation.", np.diff(prices) - slope)

    if model_name == "Simple Moving Average":
        k = min(max(1, int(sma_window)), len(prices))
        level = float(np.mean(prices[-k:]))
        residuals = prices[k - 1:] - pd.Series(prices).rolling(k).mean().dropna().to_numpy()
        return _result(model_name, index, np.repeat(level, steps), f"Constant forecast from latest {k}-observation rolling mean; not MA(q).", residuals)

    if model_name == "Weighted Moving Average":
        weights = np.array([0.05, 0.20, 0.35, 0.40])  # oldest to most recent; lecture example weights reversed into time order
        level = float(np.dot(prices[-4:], weights))
        fitted = np.array([np.dot(prices[i - 3:i + 1], weights) for i in range(3, len(prices))])
        residuals = prices[3:] - fitted
        return _result(model_name, index, np.repeat(level, steps),
                       "Constant forecast; weights on oldest→newest observations are 0.05, 0.20, 0.35, 0.40.", residuals)

    if model_name == "Linear Trend Regression":
        x = np.arange(len(prices), dtype=float)
        slope, intercept = np.polyfit(x, prices, 1)
        fitted = intercept + slope * x
        pred = intercept + slope * np.arange(len(prices), len(prices) + steps)
        return _result(model_name, index, pred, f"OLS trend; intercept={intercept:.5g}, slope={slope:.5g}/observation.", prices - fitted)

    if model_name == "Simple Exponential Smoothing":
        fit = SimpleExpSmoothing(y, initialization_method="estimated").fit(optimized=True)
        alpha = fit.params.get("smoothing_level", np.nan)
        return _result(model_name, index, fit.forecast(steps), f"SES; alpha={alpha:.4g}.", fit.resid)

    if model_name == "Holt Linear Trend":
        fit = Holt(y, damped_trend=True, initialization_method="estimated").fit(optimized=True)
        p = fit.params
        details = f"Holt damped trend; alpha={p.get('smoothing_level', np.nan):.4g}, beta={p.get('smoothing_trend', np.nan):.4g}, phi={p.get('damping_trend', np.nan):.4g}."
        return _result(model_name, index, fit.forecast(steps), details, fit.resid)

    if model_name in {"Holt-Winters Additive", "Holt-Winters Multiplicative"}:
        m = int(seasonal_period)
        if m < 2 or len(y) < max(3 * m, 40):
            raise ValueError(f"Need at least {max(3*m, 40)} observations for seasonal period {m}.")
        if model_name.endswith("Multiplicative") and (y <= 0).any():
            raise ValueError("Multiplicative seasonality requires positive values.")
        season = "mul" if model_name.endswith("Multiplicative") else "add"
        fit = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal=season,
                                   seasonal_periods=m, initialization_method="estimated").fit(optimized=True)
        p = fit.params
        details = f"Holt-Winters {season} seasonality; period={m}, alpha={p.get('smoothing_level', np.nan):.4g}, beta={p.get('smoothing_trend', np.nan):.4g}, gamma={p.get('smoothing_seasonal', np.nan):.4g}."
        return _result(model_name, index, fit.forecast(steps), details, fit.resid)

    return_families = {
        "AR(p) on returns (AICc search)": "ar",
        "MA(q) on returns (AICc search)": "ma",
        "ARMA(p,q) on returns (AICc search)": "arma",
    }
    if model_name in return_families:
        returns = y.pct_change().dropna()
        return_adf_p = _require_stationary_returns(returns, model_name)
        family = return_families[model_name]
        max_lag = max(1, min(5, int(auto_max_order)))
        if len(returns) < max(30, 5 * max_lag):
            raise ValueError("Not enough returns for the bounded AR/MA/ARMA order search.")
        orders = (
            [(p, 0, 0) for p in range(1, max_lag + 1)] if family == "ar" else
            [(0, 0, q) for q in range(1, max_lag + 1)] if family == "ma" else
            [(p, 0, q) for p in range(1, max_lag + 1) for q in range(1, max_lag + 1)]
        )
        fitted = []
        failures = []
        fit_warnings = []
        for candidate in orders:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    candidate_fit = ARIMA(returns, order=candidate, enforce_stationarity=False,
                                          enforce_invertibility=False).fit()
                fit_warnings.extend(f"{candidate}: {str(item.message).splitlines()[0][:160]}" for item in caught)
                converged = getattr(candidate_fit, "mle_retvals", {}).get("converged", True)
                score = float(candidate_fit.aicc)
                if converged and np.isfinite(score):
                    fitted.append((score, candidate, candidate_fit))
                else:
                    failures.append(f"{candidate}: optimizer did not converge or AICc was not finite")
            except Exception as exc:
                failures.append(f"{candidate}: {type(exc).__name__}: {exc}")
        if not fitted:
            raise RuntimeError("Every AR/MA/ARMA candidate failed: " + ", ".join(failures[:5]))
        aicc_value, order, fit = min(fitted, key=lambda item: item[0])
        ret_pred = np.asarray(fit.forecast(steps=steps), dtype=float)
        pred = _compound_returns(float(y.iloc[-1]), ret_pred)
        spec = f"Selected returns order {order} by minimum AICc among {len(orders)} bounded {family.upper()} candidates; return ADF p={return_adf_p:.4g}; predicted returns compounded to price."
        if failures:
            spec += f" Failed candidates: {'; '.join(failures)}."
        if fit_warnings:
            spec += f" Captured {len(fit_warnings)} convergence/numerical warning(s)."
        return _result(model_name, index, pred, spec, fit.resid, order=order, aicc=aicc_value,
                       aic=float(fit.aic), bic=float(fit.bic),
                       diagnostics={"candidate_orders": orders, "failed_candidates": failures,
                                    "fit_warnings": list(dict.fromkeys(fit_warnings))})

    if model_name == "Legacy ARIMA(2,0,2) on returns":
        returns = y.pct_change().dropna()
        return_adf_p = _require_stationary_returns(returns, model_name)
        order = (2, 0, 2)
        fit = ARIMA(returns, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
        ret_pred = np.asarray(fit.forecast(steps=steps), dtype=float)
        return _result(model_name, index, _compound_returns(float(y.iloc[-1]), ret_pred),
                       f"Legacy ARIMA(2,0,2) fitted to simple returns (ADF p={return_adf_p:.4g}); fixed-origin multi-step forecast, compounded to price.",
                       fit.resid, order=order, aicc=float(fit.aicc), aic=float(fit.aic), bic=float(fit.bic))

    if model_name in {"Auto ARIMA", "Auto SARIMA"}:
        try:
            import pmdarima as pm
        except ImportError as exc:
            raise RuntimeError("Auto ARIMA/SARIMA needs pmdarima. Install the requirements file.") from exc

        seasonal = model_name == "Auto SARIMA"
        m = int(seasonal_period)
        max_order = max(1, min(5, int(auto_max_order)))
        notes: list[str] = []
        seasonal_order = None
        if seasonal:
            if m < 2 or len(y) < max(3 * m, 60):
                raise ValueError(f"Auto SARIMA requires at least {max(3*m, 60)} observations for period {m}; found {len(y)}.")
            D, ocsb_note = _select_D_by_ocsb(prices, m)
            if ocsb_note:
                notes.append(ocsb_note)
            differencing_input = prices[m:] - prices[:-m] if D else prices
        else:
            D, differencing_input = 0, prices
        d, adf_notes = _select_d_by_adf(differencing_input, max_d=2)
        notes.extend(adf_notes)

        common = dict(start_p=0, start_q=0, max_p=max_order, max_q=max_order,
                      d=d, D=D if seasonal else None, max_d=2,
                      test="adf", information_criterion="aicc", stepwise=True,
                      suppress_warnings=True, error_action="trace", with_intercept="auto", maxiter=100)
        if seasonal:
            max_s = max(0, min(2, int(auto_max_seasonal_order)))
            common.update(seasonal=True, m=m, seasonal_test="ocsb", max_P=max_s, max_Q=max_s,
                          max_D=1, max_order=min(8, 2 * max_order + 2 * max_s))
        else:
            common["seasonal"] = False

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = pm.auto_arima(prices, **common)
        fit_warnings = list(dict.fromkeys(str(item.message).splitlines()[0][:200] for item in caught))
        order = tuple(map(int, fit.order))
        seasonal_order = tuple(map(int, fit.seasonal_order)) if seasonal else None
        pred, conf = fit.predict(n_periods=steps, return_conf_int=True, alpha=0.05)
        aicc = float(fit.aicc())
        aic, bic = float(fit.aic()), float(fit.bic())
        details = f"Auto ARIMA{order}; d={d} chosen via ADF on training data; AICc={aicc:.3f}."
        if seasonal:
            details = f"Auto SARIMA{order} x {seasonal_order}; m={m}, d={d} via ADF, D={D} via OCSB; AICc={aicc:.3f}."
        if notes:
            details += " Differencing notes: " + "; ".join(notes) + "."
        return _result(model_name, index, pred, details, fit.arima_res_.resid,
                       conf[:, 0], conf[:, 1], order, seasonal_order, aicc, aic, bic,
                       diagnostics={"search": "stepwise", "max_order": max_order, "seasonal": seasonal,
                                    "adf_notes": adf_notes, "d": d, "D": D,
                                    "ocbs_note": notes[:1] if seasonal else [], "fit_warnings": fit_warnings})

    raise ValueError(f"Unknown model: {model_name}")

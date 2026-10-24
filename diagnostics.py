"""Stationarity and residual diagnostics used by the ATSA dashboard."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


def adf_table(prices: pd.Series) -> pd.DataFrame:
    """ADF tests for price level, first difference, and simple returns."""
    p = pd.to_numeric(prices, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(p) < 20:
        raise ValueError("At least 20 values are required for the ADF diagnostic.")
    candidates = [
        ("Price level", p),
        ("First difference", p.diff().dropna()),
        ("Simple returns", p.pct_change().dropna()),
    ]
    rows: list[dict[str, Any]] = []
    for label, series in candidates:
        try:
            stat, pvalue, lags, nobs, critical, _ = adfuller(series, autolag="AIC")
            rows.append({
                "Series": label,
                "ADF statistic": float(stat),
                "p-value": float(pvalue),
                "Lags": int(lags),
                "Observations": int(nobs),
                "5% critical value": float(critical["5%"]),
                "Interpretation (5%)": (
                    "Reject unit-root null; evidence consistent with stationarity"
                    if pvalue < 0.05 else "Fail to reject unit-root null"
                ),
            })
        except Exception as exc:
            rows.append({"Series": label, "Interpretation (5%)": f"Test unavailable: {type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def acf_pacf_table(series: pd.Series, max_lags: int = 30) -> tuple[pd.DataFrame, float]:
    """Return lag-wise ACF/PACF estimates and an approximate 95% white-noise bound."""
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(x) < 20:
        raise ValueError("At least 20 values are required for ACF/PACF plots.")
    nlags = min(int(max_lags), max(1, len(x) // 2 - 1))
    av = acf(x, nlags=nlags, fft=True, missing="drop")
    pv = pacf(x, nlags=nlags, method="ywm")
    bound = 1.96 / np.sqrt(len(x))
    return pd.DataFrame({"Lag": np.arange(len(av)), "ACF": av, "PACF": pv}), float(bound)


def ljung_box_test(residuals: np.ndarray, model_df: int = 0, max_lag: int = 10) -> dict[str, Any]:
    """Test residual serial correlation; model_df adjusts ARMA parameter degrees of freedom."""
    r = np.asarray(residuals, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return {"Lag": np.nan, "Q statistic": np.nan, "p-value": np.nan, "Interpretation": "Too few residuals"}
    df = max(0, int(model_df))
    lag = min(int(max_lag), max(1, len(r) // 5), max(1, len(r) - 1))
    if lag <= df:
        lag = min(len(r) - 1, df + 1)
    try:
        result = acorr_ljungbox(r, lags=[lag], model_df=df, return_df=True)
        stat, pvalue = float(result["lb_stat"].iloc[0]), float(result["lb_pvalue"].iloc[0])
    except Exception as exc:
        return {"Lag": lag, "Q statistic": np.nan, "p-value": np.nan,
                "Interpretation": f"Test unavailable: {type(exc).__name__}: {exc}"}
    interp = ("No evidence of residual autocorrelation at the tested lag" if pvalue >= 0.05
              else "Residual autocorrelation remains; the model may be missing structure")
    return {"Lag": lag, "Q statistic": stat, "p-value": pvalue, "Interpretation": interp}


def classical_decomposition(prices: pd.Series, period: int, model: str = "additive") -> pd.DataFrame:
    """Classical trend/seasonal/remainder decomposition for exploratory diagnostics only."""
    from statsmodels.tsa.seasonal import seasonal_decompose

    p = pd.to_numeric(prices, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    m = int(period)
    if m < 2 or len(p) < 2 * m:
        raise ValueError(f"Classical decomposition needs at least {2 * m} observations for period {m}.")
    if model not in {"additive", "multiplicative"}:
        raise ValueError("Decomposition model must be additive or multiplicative.")
    if model == "multiplicative" and (p <= 0).any():
        raise ValueError("Multiplicative decomposition requires positive values.")
    result = seasonal_decompose(p, model=model, period=m, extrapolate_trend="freq")
    return pd.DataFrame({"Observed": p, "Trend": result.trend, "Seasonal": result.seasonal, "Remainder": result.resid}, index=p.index)

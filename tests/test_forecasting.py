import numpy as np
import pandas as pd
import pytest

from evaluation import clean_price_series, chronological_split, compute_metrics, run_model_comparison, mase_scale
from forecasting import MODEL_NAMES, fit_and_forecast


def series(n=220):
    idx = pd.bdate_range("2021-01-01", periods=n)
    t = np.arange(n, dtype=float)
    values = 100 + 0.08 * t + 1.7 * np.sin(2 * np.pi * t / 5) + 0.35 * np.sin(2 * np.pi * t / 29)
    return pd.Series(values, index=idx, name="SYNTH")


def stationary_return_prices(n=220, seed=123):
    """Positive price path with i.i.d. simple returns for AR-family smoke tests."""
    idx = pd.bdate_range("2021-01-01", periods=n)
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.012, n - 1)
    prices = 100.0 * np.cumprod(np.r_[1.0, 1.0 + returns])
    return pd.Series(prices, index=idx, name="STATIONARY_RETURNS")


def test_clean_price_series_drops_invalid_and_duplicates_without_filling():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-05"])
    raw = pd.Series([10.0, np.nan, 12.0, -1.0], index=dates, name="X")
    cleaned = clean_price_series(raw, min_obs=1)
    assert len(cleaned) == 2
    assert cleaned.index.tolist() == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    assert cleaned.loc[pd.Timestamp("2024-01-02")] == 12.0  # keep last valid duplicate
    assert cleaned.index.max() == pd.Timestamp("2024-01-02")  # no forward-fill to Jan 5


def test_chronological_split_counts_and_order():
    y = series(200)
    parts = chronological_split(y, 0.15)
    assert len(parts["train"]) == 140
    assert len(parts["validation"]) == 30
    assert len(parts["test"]) == 30
    assert parts["train"].index.max() < parts["validation"].index.min()
    assert parts["validation"].index.max() < parts["test"].index.min()
    assert parts["train_validation"].index.max() == parts["validation"].index.max()


def test_metrics_are_consistent_and_direction_has_one_definition():
    actual = pd.Series([101.0, 102.0, 100.0], index=pd.date_range("2024-01-01", periods=3))
    predicted = pd.Series([101.0, 101.0, 101.0], index=actual.index)
    got = compute_metrics(actual, predicted, origin=100.0, mase_scale=1.5)
    assert got["MAE"] == pytest.approx(2 / 3)
    assert got["RMSE"] == pytest.approx(np.sqrt(2 / 3))
    assert got["Direction Accuracy (%)"] == pytest.approx(100 / 3)
    assert got["MASE"] == pytest.approx((2 / 3) / 1.5)


def test_metrics_reject_misaligned_timestamps():
    actual = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2))
    predicted = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-02", periods=2))
    with pytest.raises(ValueError, match="identical timestamps"):
        compute_metrics(actual, predicted, origin=1.0, mase_scale=1.0)


def test_naive_baseline_and_horizon_are_correct():
    y = series(100)
    result = fit_and_forecast(y, 7, "Naive (random walk)")
    assert len(result["forecast"]) == 7
    assert np.allclose(result["forecast"].to_numpy(), y.iloc[-1])
    assert result["forecast"].index[0] > y.index[-1]


def test_weighted_moving_average_uses_documented_weights():
    y = pd.Series(np.arange(100.0, 140.0), index=pd.bdate_range("2023-01-02", periods=40))
    result = fit_and_forecast(y, 3, "Weighted Moving Average")
    expected = np.dot(y.iloc[-4:].to_numpy(), [0.05, 0.20, 0.35, 0.40])
    assert np.allclose(result["forecast"].to_numpy(), expected)
    assert len(result["forecast"]) == 3


def test_model_comparison_common_scale_and_export_alignment():
    y = stationary_return_prices(220)
    models = ["Naive (random walk)", "Drift", "Weighted Moving Average", "AR(p) on returns (AICc search)"]
    result = run_model_comparison(y, test_fraction=0.10, seasonal_period=5,
                                  included_models=models, auto_max_order=2)
    comp = result["comparison"]
    assert set(comp["Model"]) == set(models)
    assert result["test_predictions"].index.equals(result["test"].index)
    assert result["validation_predictions"].index.equals(result["validation"].index)
    assert len(result["test_predictions"]) == len(result["test"])
    naive = comp.loc[comp["Model"] == "Naive (random walk)"].iloc[0]
    direct = compute_metrics(result["test"], result["test_predictions"]["Naive (random walk)"],
                             float(result["train_validation"].iloc[-1]),
                             mase_scale(result["train_validation"]))
    assert naive["Test MAE"] == pytest.approx(direct["MAE"])
    assert naive["Test RMSE"] == pytest.approx(direct["RMSE"])
    # CSV export preserves the aligned target rows and forecast columns.
    roundtrip = pd.read_csv(pd.io.common.StringIO(result["test_predictions"].to_csv(index_label="Date")), parse_dates=["Date"], index_col="Date")
    assert len(roundtrip) == len(result["test"])
    assert "Actual" in roundtrip.columns


def test_auto_arima_returns_selected_order_when_available():
    pytest.importorskip("pmdarima")
    result = fit_and_forecast(series(180), 5, "Auto ARIMA", auto_max_order=2)
    assert result["arima_order"] is not None
    assert len(result["arima_order"]) == 3
    assert len(result["forecast"]) == 5
    assert result["aicc"] is not None


def test_invalid_ticker_history_like_series_has_clear_error():
    with pytest.raises(ValueError, match="At least 40"):
        clean_price_series(pd.Series([1.0, 2.0, 3.0]))


def test_catalogue_names_are_unique():
    assert len(MODEL_NAMES) == len(set(MODEL_NAMES))


def test_auto_sarima_exposes_nonseasonal_and_seasonal_orders():
    pytest.importorskip("pmdarima")
    y = series(200)
    result = fit_and_forecast(y, 6, "Auto SARIMA", seasonal_period=5,
                              auto_max_order=2, auto_max_seasonal_order=1)
    assert result["arima_order"] is not None and len(result["arima_order"]) == 3
    assert result["seasonal_order"] is not None and len(result["seasonal_order"]) == 4
    assert result["seasonal_order"][3] == 5
    assert len(result["forecast"]) == 6


def test_diagnostic_helpers_return_expected_series():
    from diagnostics import adf_table, acf_pacf_table, classical_decomposition, ljung_box_test
    y = series(220)
    adf = adf_table(y)
    assert set(adf["Series"]) == {"Price level", "First difference", "Simple returns"}
    corr, bound = acf_pacf_table(y.pct_change().dropna(), max_lags=20)
    assert {"Lag", "ACF", "PACF"}.issubset(corr.columns)
    assert 0 < bound < 1
    dec = classical_decomposition(y, period=5, model="additive")
    assert {"Observed", "Trend", "Seasonal", "Remainder"}.issubset(dec.columns)
    lb = ljung_box_test(np.diff(y.to_numpy()), model_df=0)
    assert "p-value" in lb and "Interpretation" in lb


def test_split_fraction_rounds_to_whole_observations_for_shorter_history():
    parts = chronological_split(series(60), 0.15)
    assert len(parts["validation"]) == 9
    assert len(parts["test"]) == 9
    assert len(parts["train"]) == 42


def test_ar_family_rejects_returns_without_adf_stationarity_evidence():
    y = series(220)
    with pytest.raises(ValueError, match="return ADF p-value"):
        fit_and_forecast(y, 5, "AR(p) on returns (AICc search)", auto_max_order=2)


def test_full_model_catalogue_smoke_on_stationary_returns():
    pytest.importorskip("pmdarima")
    result = run_model_comparison(
        stationary_return_prices(220, seed=123), test_fraction=0.10, seasonal_period=5,
        included_models=MODEL_NAMES, auto_max_order=2, auto_max_seasonal_order=1,
    )
    assert len(result["comparison"]) == len(MODEL_NAMES)
    assert not result["skipped"]
    assert result["validation_predictions"].index.equals(result["validation"].index)
    assert result["test_predictions"].index.equals(result["test"].index)
    assert result["test_predictions"].shape[0] == len(result["test"])
    assert result["comparison"]["Model"].is_unique

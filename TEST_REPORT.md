# Verification report

Date: 24 October 2026 (environment date)

## Automated checks performed

- **Python syntax compilation:** `app.py`, `data_source.py`, `forecasting.py`, `evaluation.py`, `diagnostics.py` and `tests/test_forecasting.py` compiled successfully using `py_compile`.
- **Pytest:** `15 passed in 14.23s` on the final automated run before packaging; tests use synthetic data and require no live market access.
- **Full model-catalogue smoke test:** the full 15-model catalogue was run on a positive 220-observation synthetic price path generated from stationary Gaussian simple returns. With 10% validation and test blocks and bounded auto-search (`auto_max_order=2`, `auto_max_seasonal_order=1`), all 15 models completed with no skipped models. Training/validation/test sizes were 176/22/22; validation and test prediction indexes matched their actual targets exactly. A direct Auto ARIMA fit also returned the requested 20-step horizon.
- **Stationarity gate:** AR/MA/ARMA-on-return and legacy ARIMA-on-return fits are screened with ADF; a deterministic trend/seasonality series whose return ADF p-value does not reject the unit-root null is skipped with an explicit explanation, instead of being silently treated as stationary.
- **Metric/CSV checks:** tests cover known MAE/RMSE/MASE/directional-accuracy calculations, rejection of misaligned timestamps, and a CSV round-trip for aligned test forecasts.
- **Automatic order reporting:** tests check that Auto ARIMA and Auto SARIMA return their selected orders and forecasts of the requested length.

## Environment limitation

- `streamlit` and `yfinance` were not installed in the available build environment.
- Installing the requirements was attempted, but package-index access failed because network/DNS resolution was unavailable.
- Therefore, the Streamlit app has been syntax-compiled but **has not been launched end-to-end here**. Yahoo Finance search, live historical downloads, currency metadata, and Streamlit Community Cloud deployment have **not** been verified here.
- No real-stock leaderboard or winner is claimed in this project because a reproducible live-data run could not be made in this environment. Run the app and export its CSVs before using actual metrics in the presentation.

## Test commands to rerun locally

```bash
python -m py_compile app.py data_source.py forecasting.py evaluation.py diagnostics.py tests/test_forecasting.py
python -m pytest -q
streamlit run app.py
```

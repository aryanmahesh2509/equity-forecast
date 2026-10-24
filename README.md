# Equity Forecast Lab — ATSA

**A comparative, reproducible forecasting research app for Advanced Time Series Analysis.** The project compares classical baselines, exponential smoothing and ARIMA-family models on historical equity prices using chronology-preserving validation and final testing. It is an educational research tool, not an investment or trading recommendation.

## Features

- Indian equities, US equities/ETFs, market indices, Yahoo Finance search, and custom ticker entry.
- Daily, weekly and monthly observations, with a selectable historical window and forecast horizon.
- Fifteen candidates: naïve/random walk, drift, simple moving average, weighted moving average, linear trend regression, SES, damped Holt trend, Holt–Winters additive/multiplicative, AICc-selected AR/MA/ARMA families on returns, legacy ARIMA(2,0,2) on returns, Auto ARIMA and Auto SARIMA.
- Chronological training, validation and untouched test blocks. The validation score selects the winner; final-test outcomes do not select the winner.
- Common price-level MAE, RMSE, MAPE, sMAPE, MASE and directional accuracy metrics.
- Actual selected ARIMA/SARIMA orders, fit-specific orders, AICc, skipped-model reasons, ADF, ACF/PACF, Ljung–Box and optional classical decomposition.
- CSV downloads for the historical prices, model comparison, aligned final-test predictions and future forecast.
- Syllabus-to-implementation mapping, code/data audit, and presentation/viva guide.

## Project structure

```text
ATSA_Equity_Forecast_Lab/
├── app.py                  # Streamlit UI, caching, state and charts
├── data_source.py          # Yahoo Finance data acquisition, ticker search and currency metadata
├── forecasting.py          # Model fitting and future-index logic
├── evaluation.py           # Data cleaning, split protocol, canonical metrics and comparison
├── diagnostics.py          # ADF, ACF/PACF, Ljung–Box, decomposition
├── tests/
│   └── test_forecasting.py # Synthetic-data regression tests
├── requirements.txt        # Application dependencies
├── requirements-dev.txt    # App dependencies plus pytest
├── SYLLABUS_MAPPING.md     # Explicit mapping to the four supplied ATSA decks
├── BUG_AUDIT.md            # Confirmed findings and known limitations
├── PRESENTATION_GUIDE.md   # Talk structure and viva preparation
├── MODEL_GUIDE.md          # Model/parameter reference
└── .gitignore
```

## Run locally (Windows PowerShell)

Use Python 3.12 for a reproducible deployment environment. Streamlit Community Cloud currently defaults to Python 3.12; select the same version when deploying. [Streamlit deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process RemoteSigned` for the current terminal, or activate the environment using the relevant command for your shell. Do not commit `.venv`, secrets or cache files.

### macOS/Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
```

## How to use the dashboard

1. Choose an instrument from the market-specific preset list, search Yahoo Finance by company or ticker, or enter a custom Yahoo Finance symbol directly.
2. Choose historical range, frequency and forecast horizon.
3. In Recommended analysis mode, use the defaults. In Advanced research mode, change the selection metric, split size, model subset, SMA window, seasonal period and bounded ARIMA search ranges.
4. Run the comparison. Review the validation-selected model and its performance relative to the naïve baseline.
5. Inspect the Model comparison tab for validation scores, final-test scores, and the distinct model orders used for validation/test fits.
6. Inspect the Diagnostics tab. A significant Ljung–Box result can signal remaining autocorrelation; it does not automatically make a forecast unusable.
7. Download the CSVs for reproducible reporting.

## Evaluation protocol

Let `f` be the validation/test fraction. For data of length `n`, each holdout block is `max(1, round(n*f))` observations; training is the earliest remaining block and must retain at least 40 observations. The split is chronological:

- **Training:** fits candidate models.
- **Validation:** predicts the next block from a fixed forecast origin; chooses the winner using the selected validation metric.
- **Final test:** candidate models are refitted on training + validation, then forecast this untouched block. These scores are reported for comparison and are not used to choose the winner.
- **Future forecast:** the selected model configuration is fit on all available observations. Auto ARIMA/SARIMA can select a different order on the larger final dataset; the app labels the future-fit specification separately.

All compared models are evaluated on the same price-level target dates, even where a candidate is fitted internally to percentage returns. Return forecasts are compounded into a price path before scoring.

Metrics: MAE and RMSE are in price units; MAPE and sMAPE are percentages; MASE divides MAE by the training series' mean absolute one-step naive error; directional accuracy compares signs of actual changes and forecast-path changes from the fixed origin. Direction accuracy is supplementary and can be low for flat forecasts.

AICc is used within comparable model-order searches. It is not used as a direct cross-family performance score. The ADF test uses a unit-root null; failure to reject is not proof of non-stationarity. ACF/PACF guide interpretation but do not deterministically identify an optimal model.

## Model catalogue and scope

See `MODEL_GUIDE.md` for the practical role of each candidate and `SYLLABUS_MAPPING.md` for which subjects are explicitly covered in Unit 1, Unit 2, combined Units 3–6 and Unit 7. Regression with ARIMA errors is deferred until aligned exogenous variables can be supplied. Hierarchical/grouped forecasting and OLS/MinT reconciliation are deferred because a single ticker is not a meaningful hierarchy. They are proposed as a future multi-series extension rather than artificially inserted into this application.

The original ARIMA notebook's fixed `(2,0,2)` return model is retained only as a legacy comparison. The main evaluation is a fixed-origin multi-step holdout protocol; it is not the same as walk-forward one-step forecasting.

## Deploy to Streamlit Community Cloud

Streamlit Community Cloud deploys from a GitHub repository and installs dependencies from `requirements.txt`. [Official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [dependency guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

1. Create a GitHub repository, e.g. `atsa-equity-forecast-lab`.
2. Put the files in this folder at the repository root. Do not upload `.venv/`, cache files or personal credentials.
3. Commit and push the repository.
4. Open [Streamlit Community Cloud](https://share.streamlit.io/), select **Create app**, then choose your repository, branch and `app.py` entrypoint.
5. In advanced deployment settings, choose Python 3.12 to match local testing.
6. Deploy and inspect build logs if dependency installation fails.

The app requires network access to Yahoo Finance for live retrieval of historical observations. “Latest available” means the most recent observation returned by that provider, not guaranteed real-time quotes. Yahoo symbol coverage and metadata availability vary.

## Testing

```bash
python -m pytest -q
```

Tests use generated synthetic series; no live market download is required. They cover cleaning without forward fills, split chronology, metric definitions/alignment, naïve and weighted baselines, forecast length, comparison output alignment, the stationarity gate for AR/MA/ARMA-on-return models, automatic ARIMA order reporting, and a full 15-model smoke test on synthetic data. See the test run output from your local environment before claiming a passing result in a report.

## Limitations

- Stock-price forecasts are uncertain and past backtest performance is not a promise of future market behaviour.
- Future dates approximate business days, Fridays or month ends; actual exchange holidays are not modelled.
- Auto SARIMA's seasonal period is a hypothesis, not evidence that seasonal structure exists.
- Model-based prediction intervals are shown for Auto ARIMA/SARIMA only and depend on residual/distributional assumptions.
- Fixed-origin multi-step evaluation differs from one-step rolling evaluation; comparing results from different protocols can be misleading.
- Provider failures, unsupported tickers and model-fitting errors are surfaced, but users should inspect model-skip reasons before interpreting a result.

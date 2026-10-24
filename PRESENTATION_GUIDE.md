# Equity Forecast Lab — ATSA presentation guide

## Suggested title

**Equity Forecast Lab: Comparative Evaluation of Classical Time-Series Models for Stock Price Forecasting**

## Research question

How do classical statistical and time-series models compare when forecasting equity prices, and which model produces the strongest out-of-sample results under a consistent evaluation methodology?

Secondary questions:
- Does a model selected on validation also generalise to the final test period?
- Do trend, smoothing or seasonal models improve on the naïve baseline?
- Does systematic order selection improve on the legacy ARIMA(2,0,2) specification?
- Do selected models and diagnostics change across tickers and observation frequencies?

## Slide structure (suggested 7–9 minutes; adjust to the actual time limit)

1. **Motivation and evolution (30–40 sec).** The earlier project explored ARIMA before it was taught; the ATSA version replaces guessed parameters with formal identification, evaluation and diagnostics.
2. **Problem/objectives (30 sec).** Compare model families on a common equity-price target; select by validation and test on unseen observations.
3. **Data and exploratory analysis (45 sec).** State ticker, provider, frequency, date range and observation count. Show a historical price chart and ADF/return observations. Mention data provider limitations and that prices are adjusted (yfinance `auto_adjust=True`).
4. **Model families (60 sec).** Baselines; regression/trend; exponential smoothing; AR/MA/ARMA; ARIMA and SARIMA. Explicitly distinguish SMA/WMA smoothing from stochastic MA(q).
5. **ARIMA order selection (60 sec).** Explain p,d,q, ADF, differencing, ACF/PACF as aids, bounded Auto ARIMA order search and AICc. Show the selected order from the run, not a fixed example.
6. **Seasonality (30–45 sec).** Explain P,D,Q,m and why a period (5, 52 or 12) is a testable assumption; show whether seasonal candidates helped this ticker.
7. **Evaluation protocol (60 sec).** Show exact split counts and dates. Training fits models; validation selects winner; final test reports generalisation. All models are scored on the same price-level timestamps.
8. **Results and diagnostics (60–90 sec).** Insert actual exported comparison table from a reproducible run. Identify validation winner; show test score separately, baseline comparison, ADF, residual plot and Ljung–Box interpretation. Do not select the “winner” based on test metrics.
9. **Dashboard/demo, limitations and future work (45–60 sec).** Search a second ticker; show model orders, exports, and why results can change. Limitations: markets are noisy, no exogenous predictors yet, and no hierarchy for Unit 7 reconciliation.

### Results slide template — fill only after running the app

- Ticker/frequency/date range: `[record from the run]`
- Training / validation / final test observations and date ranges: `[export these from the app]`
- Validation selection metric: `[MAE by default, or the actual setting]`
- Validation-selected model and validation score: `[from comparison CSV]`
- Naive validation score: `[from comparison CSV]`
- Winner final-test MAE / RMSE / sMAPE / MASE / directional accuracy: `[from comparison CSV]`
- Best test score among candidates (describe only; do not retrospectively change winner): `[from comparison CSV]`
- ARIMA/SARIMA validation-fit, test-fit and future-fit orders: `[record the distinct labels shown by the app]`
- Ljung–Box p-value and interpretation: `[from diagnostics tab]`

No numbers should be copied from an older run after changing ticker, window, frequency, horizon, metric, candidate set or search bounds. The app does not contain a hard-coded winning model.

## How to explain the methodology

- **Why a chronological split?** Random splitting breaks temporal order and can leak future regimes into training. Sequential blocks imitate forecasting into later time.
- **Why validation and a separate test?** Validation selects the model. The test is a separate estimate of generalisation and is not used to choose the winner.
- **Why is the future order possibly different?** Auto ARIMA searches the supplied sample. A training fit, a train+validation refit and an all-data future fit can select different orders.
- **Why include naive?** If sophisticated methods cannot beat a simple benchmark, their extra complexity may not add forecasting value for that experiment.
- **Why not rank all models by AICc?** AICc is a likelihood-based candidate-selection criterion. Differencing and likelihood specifications differ, and it is not the same as held-out price-level forecast error.
- **Why use MASE?** It scales MAE by the training series' naive one-step error, helping comparisons across price scales, provided the denominator is nonzero and correctly computed.
- **Why not claim a seasonal model is better?** Frequency provides a possible season length, not evidence that the price series has stable seasonal behaviour. Validation and test performance decide whether it helps.

## Viva preparation

**1. What is a time series?** Observations indexed in temporal order, often with autocorrelation and possibly trend, seasonal and cyclic components.

**2. What is stationarity?** A process whose statistical properties relevant to modelling, such as mean and autocovariance, do not change over time. ARMA models generally assume stationarity.

**3. What is the ADF null hypothesis?** The series has a unit root. A small p-value is evidence against that null, but the test is not conclusive proof of stationarity.

**4. Why difference a series?** To remove stochastic trend / unit-root behaviour and make a series more suitable for ARMA terms. Over-differencing can harm the model.

**5. How do ACF and PACF differ?** ACF measures correlation with a lagged series value; PACF measures the lag relationship after accounting for shorter lags. They guide candidate identification, not automatic certainty.

**6. Explain AR, MA and ARMA.** AR uses prior observations; MA uses current/past innovations; ARMA combines both for stationary data.

**7. Explain p,d,q.** p is AR order, d is differencing order, q is MA innovation order.

**8. What does Auto ARIMA do?** Searches candidate ARIMA orders under configured bounds; uses ADF to help select d and AICc to rank fits. It does not guarantee lowest future forecast error.

**9. How is SARIMA different?** It adds seasonal AR, differencing and MA terms, with period m.

**10. Additive vs multiplicative seasonality?** Additive assumes seasonal magnitude is roughly constant; multiplicative assumes it scales with the level and requires positive observations.

**11. SMA vs MA(q)?** SMA is an average of recent observed values; MA(q) is a stochastic model for innovations. They are not interchangeable.

**12. What does AICc measure?** Likelihood fit penalised for complexity with a small-sample correction; lower is preferable among comparable candidate fits, not necessarily across unrelated models.

**13. What is the Ljung–Box test?** A portmanteau test for residual autocorrelation up to a lag. A low p-value suggests remaining serial dependence.

**14. What does MAE measure?** Average absolute price error in price units. It is interpretable and not as dominated by large errors as RMSE.

**15. Why report RMSE too?** It squares errors before averaging and penalises large misses more.

**16. Why report MASE?** It scales MAE against a simple training-set naive error and aids comparisons across scales.

**17. What is leakage?** Using data from the future of a forecast origin during model fitting, feature creation, scaling or selection.

**18. Why isn't the test set used to choose the winner?** It would turn the test set into another validation set and overstate generalisation.

**19. Why can Auto ARIMA and naive have identical forecasts?** The selected model may be ARIMA(0,1,0) without drift, equivalent to a random-walk price forecast.

**20. Why is direction accuracy poor for a naive model?** Its predicted change is generally zero, so it cannot predict whether each next move is up or down even when its price-level error is competitive.

**21. Why are hierarchical/MinT methods deferred?** Unit 7 reconciliation needs multiple related series and a meaningful aggregation matrix. One stock price alone does not provide a hierarchy.

**22. What would you improve next?** Rolling-origin evaluation, additional exogenous market/sector predictors, robust uncertainty calibration, and a multi-series setup for coherent reconciliation—each with separate methodology and leakage controls.

## Demo checklist

- Run one Indian equity (e.g. TCS.NS) and one other symbol; note actual available data ranges.
- Capture the actual run's validation winner and test metrics to CSV.
- Show selected model orders and explain why validation/test/future orders might differ.
- Show the naive baseline comparison and residual diagnostics.
- Show how unavailable/failed models are reported.
- Do not describe Yahoo Finance history as guaranteed real-time data or present forecasts as investment advice.

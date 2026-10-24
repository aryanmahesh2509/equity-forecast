# ATSA model and parameter guide

## Sidebar settings

- **Stock / index:** security selected from the relevant market list, Yahoo Finance search results, or direct ticker entry.
- **Historical data:** requested history (`2y`, `5y`, `10y`, `max`). More observations can help fit seasonal models but do not guarantee better forecasts.
- **Observation frequency:** daily, weekly or monthly sample. The evaluation compares observations at the selected frequency; future calendar labels are approximate.
- **Forecast horizon:** number of future observations to forecast, not calendar days.
- **Validation/test size:** advanced setting. Each is the chosen fraction, rounded to whole observations. The remaining training block must contain at least 40 observations.
- **Selection metric:** MAE, RMSE, MAPE, sMAPE or MASE on validation predictions only.
- **SMA lookback:** number of most recent observations averaged by the SMA forecast.
- **Seasonal period m:** default is 5 daily trading sessions, 52 weekly observations or 12 monthly observations; advanced mode allows alternatives to test a hypothesis.
- **Auto ARIMA max p/q:** maximum non-seasonal AR/MA order searched within the bounded automatic/family search.
- **Auto SARIMA max P/Q:** maximum seasonal AR/MA order in the bounded seasonal search. Seasonal differencing is bounded separately.
- **Models to include:** subset of candidates in Advanced research mode; the validation winner is only the best among included models.

## Model catalogue

| Model | What it forecasts | When it may work well | Limitations |
|---|---|---|---|
| Naive / random walk | Last price repeated forward | Strong baseline for hard-to-predict prices | Cannot project trend/seasonality |
| Drift | Last price plus historical mean change per horizon step | Persistent average trend | Reversals make extrapolated slope misleading |
| Simple Moving Average | Constant forecast at latest k-price mean | Smooth, stable levels | Lags turning points; parameter k matters |
| Weighted Moving Average | Constant forecast using latest four prices with lecture-example weights | Recent observations should count more than older ones | Weights are fixed and not learned; may lag changes |
| Linear Trend Regression | OLS straight line against time | Approximately linear trend | Misses nonlinear dynamics and serially correlated errors |
| Simple Exponential Smoothing | Level only | No clear trend or seasonality | Cannot explicitly represent trend or seasonal cycles |
| Holt Linear Trend (damped) | Level + damped trend | Trend exists but should not extend indefinitely | Trend may still be unstable |
| Holt–Winters Additive | Level + damped trend + additive seasonality | Seasonal amplitude roughly constant | Needs several complete cycles; seasonality may be absent |
| Holt–Winters Multiplicative | Level + damped trend + multiplicative seasonality | Seasonal amplitude proportional to the level | Requires positive values and stable proportional seasonality |
| AR(p) on returns (AICc search) | Return series with autoregressive terms | Return autocorrelation exists | Candidate search is bounded; not a guarantee of forecastability |
| MA(q) on returns (AICc search) | Return series with moving-average innovations | Short-lived shock dependence exists | Error terms are unobserved; specification assumptions matter |
| ARMA(p,q) on returns (AICc search) | Stationary returns with AR and MA terms | Both types of short-lag dependence may exist | AICc selects only within the configured candidate set |
| Legacy ARIMA(2,0,2) on returns | Fixed legacy order, forecasts returns then compounds prices | Reproduce and compare the earlier project | Order was manually chosen in the earlier semester; not automatically optimal |
| Auto ARIMA | Price levels with searched p,d,q | Differencing and order need systematic selection | Search is bounded; result depends on data/window/settings |
| Auto SARIMA | Price levels plus seasonal P,D,Q,m | Repeating seasonal pattern is plausible | Extra parameters can overfit; seasonality may not improve test forecasts |

### Do not confuse two meanings of “moving average”

- **SMA/WMA smoothing forecasts** average observed price levels with fixed equal or weighted coefficients.
- **MA(q) stochastic time-series models** model a series as a function of current and previous forecast innovations. These are different model families, even though both use the phrase “moving average.”

## ARIMA parameters and selection

`ARIMA(p,d,q)`:

- `p` — number of autoregressive terms.
- `d` — number of non-seasonal differences.
- `q` — number of moving-average innovation terms.

`SARIMA(p,d,q)(P,D,Q,m)` adds seasonal AR order `P`, seasonal differencing `D`, seasonal MA order `Q`, and seasonal period `m`.

Auto ARIMA uses a bounded stepwise search, uses the ADF test to help select `d`, and ranks candidate fits using AICc. Auto SARIMA also tests seasonal differencing (OCSB in the implemented configuration) and searches bounded seasonal orders. A selected order is not an oracle: it is the best candidate under those search constraints and the fitted sample.

Orders can legitimately differ across fits: training-only fit for validation predictions, training+validation fit for the final test, and all-data fit for the future forecast. The UI labels these separately.

## Diagnostics

- **ADF:** null hypothesis is a unit root. Small p-values provide evidence against that null; failure to reject does not prove non-stationarity.
- **ACF:** autocorrelation at successive lags; can reveal dependence or repeated seasonal lags.
- **PACF:** lag-specific partial autocorrelation after accounting for shorter lags; helps suggest AR structure.
- **AIC/AICc/BIC:** fit-complexity criteria. Lower values are preferred only for compatible candidate fits; do not rank unrelated methods based on AICc alone.
- **Ljung–Box:** tests residual serial correlation up to a chosen lag. Significant results indicate remaining dependence at that lag, not that every use of the model must be discarded.
- **Classical decomposition:** exploratory separation of observed series into trend, seasonality and remainder. It uses training+validation data and does not automatically justify seasonal modelling.

## Metrics

Let actual observations be `y_t` and predictions be `yhat_t`.

- **MAE:** mean of `|y_t - yhat_t|`; lower is better; measured in price units.
- **RMSE:** square root of mean squared error; lower is better; penalizes large errors more.
- **MAPE:** mean absolute percentage error; can behave poorly around zero or near-zero actual values.
- **sMAPE:** symmetric percentage-style error; denominator also can be zero, so implementation uses a small numerical floor.
- **MASE:** MAE divided by mean absolute one-step change from the training series. Values below 1 outperform that training naive scale; denominator must come only from training data.
- **Directional accuracy:** percentage of matching signs of actual versus forecast-path price changes from the same fixed origin. Supplementary only; a flat forecast can show poor direction accuracy.

## Evaluation protocol

1. Fit each eligible candidate on training observations.
2. Forecast the whole validation block from a fixed origin.
3. Select the winner using the configured validation metric.
4. Refit each candidate on training + validation, then forecast the untouched test block for reporting.
5. Fit the selected candidate on all available prices for future forecasts.

Do not use final-test results to retroactively choose the winner. The protocol is fixed-origin multi-step forecasting, not rolling-origin one-step forecasting.

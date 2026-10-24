# Project audit: confirmed issues, methodology risks, and changes

Audited inputs: original project ZIP, the current revised source, all slide text across the four ATSA decks, and the `ARIMA_MODEL` Jupyter notebook supplied in the conversation.

## Confirmed issues in the supplied notebook/source

1. **Artificial weekday observations in the original notebook.** `df.asfreq('B').ffill()` inserts weekdays that were not trading observations, including exchange holidays, and carries forward prices. That creates artificial zero returns and changes the target series. The new data loader keeps provider observations and does not forward-fill weekends/holidays.
2. **Forecast/actual index alignment in the original notebook is off by one in its first test target.** Returns are created after the first price observation, but the first test forecast is initialized against `Close.iloc[split]` while the actual-price slice starts at that same index. The first held-out return corresponds to the next price observation. This misaligns the first predicted and actual price values.
3. **The original “2025 forecast” mixes a rolling one-step idea with compounded multi-step price predictions.** It appends newly observed returns to history after each period, but the price path is compounded from the previous forecast rather than reset to the latest observed price for each one-step forecast. For an actual rolling one-step test, each prediction should be based on the actual information available at that origin, and its price target must be the next correctly aligned observation.
4. **The original notebook uses a fixed ARIMA(2,0,2) order without a documented selection procedure.** The new app retains this as a legacy benchmark but adds a bounded AICc search for AR/MA/ARMA families and Auto ARIMA / Auto SARIMA.
5. **Earlier dashboard order ambiguity.** Automatic model orders may change when fitted on training data, training + validation, or all observations. One table field cannot represent all three. The revised results label validation-fit order, test-refit order and future-fit specification separately.
6. **Potentially misleading “best model” interpretation.** A model with the best validation score may not have the lowest final-test error. The updated application selects only on validation and reports test performance separately. Test results must not be used retrospectively to change the winner.
7. **Global warning suppression.** The former forecasting module suppressed UserWarning and RuntimeWarning globally, which can hide useful information. The updated module no longer suppresses warnings globally. Auto ARIMA uses bounded candidate search; unsuccessful models are reported in the UI.
8. **Data cleaning should not invent prices.** The new cleaner removes missing/infinite/non-positive price observations and duplicate timestamps (keeping the last occurrence); it does not fill arbitrary calendar gaps. Non-positive observations are treated as invalid price records, not replaced.
9. **One metric implementation is required.** The updated evaluation module computes MAE, RMSE, MAPE, sMAPE, MASE and directional accuracy in one canonical function and requires identical target timestamps for actual and predicted series.
10. **Directional accuracy definition needed to be explicit.** It is now computed by comparing signs of actual price changes and forecast-path price changes, both referenced to the last actual price at the fixed forecast origin. It is a supplementary measure; a flat forecast can score poorly even if price-level MAE is competitive.
11. **Historical AICc is not a universal cross-model score.** AICc is shown where available for fitted ARIMA-family models and is used within comparable order searches. Validation errors, not AICc across unrelated families, determine the displayed winner.
12. **Seasonality must not be presumed.** Seasonal candidates require enough observations for multiple seasonal cycles. Failure/insufficient history is reported in the skipped-model section, and seasonal models can lose to baselines.

## Changes made in the rebuilt project

- Split code into `app.py` (UI, state and caching), `data_source.py` (Yahoo Finance acquisition/search/metadata), `forecasting.py` (model fits), `evaluation.py` (chronological split, common metrics, comparison protocol), and `diagnostics.py` (ADF, ACF/PACF, Ljung–Box and classical decomposition).
- Added Weighted Moving Average using the lecture example's weights, keeping it clearly distinct from the stochastic MA(q) model.
- Replaced fixed AR(1)/MA(1)/ARMA(1,1) family choices with bounded AICc order searches in each family. The original fixed ARIMA(2,0,2) remains a separate legacy benchmark.
- Added validation-fit and test-refit order columns, plus the future-fit model specification.
- Added MASE as an optional advanced metric; its denominator is calculated from training observations only.
- Added train / validation / test counts and date ranges, CSV export of comparison and aligned test predictions, and clear warnings about forecast-date approximations and model-based intervals.
- Added separate Indian equities, US equities/ETFs and indices menus; a Yahoo Finance search box; and a raw custom ticker field. Provider search is best-effort and can only return symbols Yahoo Finance recognises.
- Added model-skip reasons, clearer error messages, and bounded Auto ARIMA/SARIMA search settings.
- Added synthetic tests for cleaning, chronological splitting, metric consistency, alignment, baselines, model horizon and CSV round-trip.

## Known limitations / not claimed as tested

- Streamlit and yfinance were not installed in the available build environment. The UI was syntax-compiled but not launched end-to-end, and live Yahoo Finance retrieval/currency metadata were not verified here.
- Model engine tests run on synthetic data and do not establish which model wins on current TCS or another real ticker.
- Future observation dates approximate weekdays, Friday week-ends or month-end; exchange-specific holidays are not modelled.
- Auto ARIMA prediction intervals are shown only for that fitting path. Intervals are model-conditional and can be unreliable if residual assumptions fail.
- The main experiment is fixed-origin multi-step validation/test, not walk-forward one-step testing. The protocol is documented and consistent across candidates.
- Regression with ARIMA errors is deferred until aligned exogenous predictors are supplied. Hierarchical reconciliation is deferred until a meaningful multi-series structure exists.

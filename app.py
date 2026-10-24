from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from diagnostics import acf_pacf_table, adf_table, classical_decomposition, ljung_box_test
from data_source import download_prices as _download_prices, get_currency as _get_currency, search_yahoo_tickers as _search_yahoo_tickers
from evaluation import clean_price_series, run_model_comparison
from forecasting import MODEL_DESCRIPTIONS, MODEL_NAMES, fit_and_forecast

st.set_page_config(page_title="ATSA | Equity Forecast Lab", page_icon="📈", layout="wide")

INDIAN_TICKERS = {
    "Tata Consultancy Services": "TCS.NS", "Reliance Industries": "RELIANCE.NS",
    "Infosys": "INFY.NS", "HDFC Bank": "HDFCBANK.NS", "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS", "ITC": "ITC.NS", "Larsen & Toubro": "LT.NS",
    "Bharti Airtel": "BHARTIARTL.NS", "Axis Bank": "AXISBANK.NS", "Wipro": "WIPRO.NS",
    "Tata Motors": "TATAMOTORS.NS", "Maruti Suzuki": "MARUTI.NS",
    "Hindustan Unilever": "HINDUNILVR.NS", "Mahindra & Mahindra": "M&M.NS",
    "Bajaj Finance": "BAJFINANCE.NS", "L&T Technology Services": "LTTS.NS",
}
US_TICKERS = {
    "Apple": "AAPL", "Microsoft": "MSFT", "NVIDIA": "NVDA", "Alphabet (Google)": "GOOGL",
    "Amazon": "AMZN", "Meta Platforms": "META", "Tesla": "TSLA", "Berkshire Hathaway B": "BRK-B",
    "JPMorgan Chase": "JPM", "Walmart": "WMT", "Coca-Cola": "KO", "Exxon Mobil": "XOM",
    "SPDR S&P 500 ETF": "SPY", "Invesco QQQ ETF": "QQQ",
}
INDEX_TICKERS = {"Nifty 50": "^NSEI", "BSE Sensex": "^BSESN", "S&P 500": "^GSPC", "NASDAQ Composite": "^IXIC", "Dow Jones Industrial Average": "^DJI"}
MARKET_MAP = {"India equities": INDIAN_TICKERS, "US equities / ETFs": US_TICKERS, "Market indices": INDEX_TICKERS}

st.markdown("""
<style>
.block-container {padding-top: 1.25rem; padding-bottom: 2.2rem; max-width: 1550px;}
[data-testid="stMetric"] {border: 1px solid rgba(120,130,150,.22); padding: 12px 15px; border-radius: 12px;}
.small-note {font-size: .88rem; color: #777;}
</style>
""", unsafe_allow_html=True)
st.title("Equity Forecast Lab")
st.caption("Advanced Time Series Analysis · Comparative evaluation of classical forecasting methods")
st.write("Compare baseline, exponential-smoothing and ARIMA-family models using a chronological validation split and a separate final test period.")


@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(ticker: str, period: str, interval_label: str) -> pd.Series:
    return _download_prices(ticker, period, interval_label)


@st.cache_data(ttl=900, show_spinner=False)
def search_yahoo_tickers(query: str) -> tuple[list[dict[str, str]], str | None]:
    return _search_yahoo_tickers(query)


@st.cache_data(ttl=3600, show_spinner=False)
def get_currency(ticker: str) -> str | None:
    return _get_currency(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def compare_cached(prices: pd.Series, interval_label: str, test_fraction: float, seasonal_period: int,
                   sma_window: int, included_models: tuple[str, ...], auto_max_order: int,
                   auto_max_seasonal_order: int, selection_metric: str, horizon: int):
    # metric and horizon are included in this cache key to invalidate runs whenever experiment settings change.
    return run_model_comparison(prices, interval=interval_label, test_fraction=test_fraction,
                                seasonal_period=seasonal_period, sma_window=sma_window,
                                included_models=included_models, auto_max_order=auto_max_order,
                                auto_max_seasonal_order=auto_max_seasonal_order)


with st.sidebar:
    st.header("Experiment settings")
    analysis_mode = st.radio("Mode", ["Recommended analysis", "Advanced research"], index=0,
                             help="Recommended mode uses sensible defaults; advanced mode exposes experimental controls.")
    query = st.text_input("Search Yahoo Finance", placeholder="Company or ticker: Tata Motors, TMCV, AAPL")
    search_results, search_error = search_yahoo_tickers(query)
    if search_error:
        st.warning(search_error)
    search_labels = [r["label"] for r in search_results]
    searched_symbol = ""
    if search_labels:
        selected_search = st.selectbox("Search results", search_labels)
        searched_symbol = search_results[search_labels.index(selected_search)]["symbol"]

    with st.form("analysis_settings"):
        market = st.selectbox("Market / universe", list(MARKET_MAP.keys()), index=0)
        universe = MARKET_MAP[market]
        preset_name = st.selectbox("Popular stock / index", list(universe.keys()), index=0)
        custom_ticker = st.text_input("Or enter a Yahoo Finance ticker", placeholder="e.g. TMCV.NS, AAPL, ^NSEI")
        history = st.selectbox("Historical data", ["2y", "5y", "10y", "max"], index=1)
        frequency = st.selectbox("Observation frequency", ["Daily", "Weekly", "Monthly"], index=0)
        horizon = st.select_slider(
            "Future forecast horizon (observations)", options=[5, 10, 20, 30, 60], value=20,
            help="Controls the future forecast chart/CSV. Validation and final-test forecast lengths are set by the holdout split.",
        )

        default_period = {"Daily": 5, "Weekly": 52, "Monthly": 12}[frequency]
        test_pct, metric, sma_window = 0.15, "MAE", 20
        seasonal_period, max_order, max_seasonal_order = default_period, 3, 1
        included_models = tuple(MODEL_NAMES)
        if analysis_mode == "Advanced research":
            st.divider()
            st.caption("Advanced evaluation controls")
            test_pct = st.select_slider("Validation and final test size each", options=[0.10, 0.15, 0.20], value=0.15,
                                        format_func=lambda v: f"{int(v*100)}%")
            metric = st.selectbox("Selection metric (validation only)", ["MAE", "RMSE", "MAPE (%)", "sMAPE (%)", "MASE"], index=0)
            sma_window = st.select_slider("SMA lookback", options=[5, 10, 20, 50], value=20)
            seasonal_period = st.selectbox("Seasonal period m", sorted(set([3, 4, 5, 7, 12, 26, 52])), index=[3,4,5,7,12,26,52].index(default_period))
            included_models = tuple(st.multiselect("Models to include", MODEL_NAMES, default=MODEL_NAMES))
            max_order = st.slider("Auto ARIMA max p/q", min_value=1, max_value=5, value=3)
            max_seasonal_order = st.slider("Auto SARIMA max P/Q", min_value=0, max_value=2, value=1)
        submitted = st.form_submit_button("Run model comparison", type="primary", use_container_width=True)

if submitted:
    ticker = custom_ticker.strip().upper() or searched_symbol or universe[preset_name]
    if not ticker:
        st.error("Choose a preset, select a search result, or enter a ticker.")
    elif analysis_mode == "Advanced research" and not included_models:
        st.error("Choose at least one model.")
    else:
        st.session_state["experiment"] = {
            "ticker": ticker, "history": history, "frequency": frequency, "horizon": int(horizon),
            "test_pct": float(test_pct), "metric": metric, "sma_window": int(sma_window),
            "seasonal_period": int(seasonal_period), "included_models": included_models,
            "auto_max_order": int(max_order), "auto_max_seasonal_order": int(max_seasonal_order),
            "analysis_mode": analysis_mode,
        }

config = st.session_state.get("experiment")
if config is None:
    st.info("Choose a stock or index, set the experiment, and click **Run model comparison**.")
    with st.expander("What is included and what is not?"):
        st.markdown("""
**Implemented:** naïve/random walk, drift, simple and weighted moving-average baselines, linear trend regression, SES, damped Holt trend, Holt–Winters additive and multiplicative, bounded AICc searches over AR(p), MA(q), ARMA(p,q) on returns, legacy ARIMA(2,0,2) on returns, Auto ARIMA and Auto SARIMA.

**Syllabus topics used as diagnostics/context:** time plots, stationarity, ACF/PACF, ADF, AIC/AICc/BIC discussion, prediction intervals, residuals, Ljung–Box and chronological test evaluation.

**Not in the core app:** regression with ARIMA errors requires exogenous predictor data; hierarchical/grouped forecasts and MinT reconciliation require a defensible set of related series and aggregation constraints. These are described as future extensions in the syllabus map.
        """)
    st.stop()

ticker = config["ticker"]
frequency = config["frequency"]
try:
    with st.spinner(f"Downloading {ticker} history from Yahoo Finance…"):
        prices = download_prices(ticker, config["history"], frequency)
    with st.spinner("Fitting model candidates and evaluating chronological blocks…"):
        result = compare_cached(prices, frequency, config["test_pct"], config["seasonal_period"],
                               config["sma_window"], tuple(config["included_models"]),
                               config["auto_max_order"], config["auto_max_seasonal_order"],
                               config["metric"], config["horizon"])
except Exception as exc:
    st.error(f"Analysis could not be completed: {type(exc).__name__}: {exc}")
    st.info("Check the ticker and network connection. Seasonal models may need a longer history. Open the Models skipped section if some candidates failed.")
    st.stop()

comparison = result["comparison"].copy()
metric_col = f"Validation {config['metric']}"
if metric_col not in comparison or comparison[metric_col].notna().sum() == 0:
    st.error(f"No models produced a valid {config['metric']} selection score. Try another metric or model set.")
    st.stop()
comparison = comparison.sort_values(metric_col, ascending=True, na_position="last").reset_index(drop=True)
best_model = str(comparison.iloc[0]["Model"])
best_score = float(comparison.iloc[0][metric_col])

latest = float(prices.iloc[-1])
previous = float(prices.iloc[-2])
price_delta = latest - previous
price_pct = price_delta / previous * 100 if previous else np.nan
currency = get_currency(ticker)
currency_prefix = f"{currency} " if currency else ""
train, validation, test = result["train"], result["validation"], result["test"]
naive_row = comparison.loc[comparison["Model"] == "Naive (random walk)"]
naive_val = float(naive_row.iloc[0][metric_col]) if len(naive_row) and pd.notna(naive_row.iloc[0][metric_col]) else np.nan
if np.isfinite(naive_val) and naive_val > 0:
    improvement = (naive_val - best_score) / naive_val * 100
    baseline_text = (f"{best_model} has {improvement:.1f}% lower validation {config['metric']} than the naïve baseline."
                     if improvement >= 0 else f"{best_model} has {abs(improvement):.1f}% higher validation {config['metric']} than the naïve baseline.")
else:
    baseline_text = "Naïve baseline comparison is unavailable for this model selection."

st.markdown(f"### {ticker} · {frequency.lower()} observations")
cols = st.columns(4)
cols[0].metric("Latest adjusted close", f"{currency_prefix}{latest:,.2f}", f"{price_delta:+,.2f} ({price_pct:+.2f}%)")
cols[1].metric("Valid observations", f"{len(prices):,}")
cols[2].metric("Validation winner", best_model)
cols[3].metric(f"Validation {config['metric']}", f"{best_score:,.4f}")
st.caption(f"Source: Yahoo Finance via yfinance · adjusted prices (auto-adjusted Close) · {prices.index.min().date()} to {prices.index.max().date()} · frequency: {frequency} · currency: {currency or 'not reported by provider'}")
st.info(f"**Selection result:** {baseline_text} The final test period remains separate and is reported as evidence of out-of-sample performance, not used to select the winner.")

with st.expander("Chronological split details", expanded=False):
    split_cols = st.columns(3)
    for col, label, series in [(split_cols[0], "Training", train), (split_cols[1], "Validation", validation), (split_cols[2], "Final test", test)]:
        col.metric(label, f"{len(series)} observations")
        col.caption(f"{series.index.min().date()} to {series.index.max().date()}")
    st.caption("Validation and final test each use the configured fraction, rounded to whole observations. The split is rejected if fewer than 40 observations remain for training; short holdout blocks should be interpreted cautiously.")

forecast_tab, comparison_tab, diagnostics_tab, history_tab = st.tabs(["Forecast", "Model comparison", "Diagnostics", "Historical data"])

with forecast_tab:
    st.subheader("Future forecast")
    model_options = comparison["Model"].tolist()
    selector_key = "future_model_selector:" + ":".join([
        ticker, frequency, config["history"], str(config["test_pct"]), config["metric"],
        "|".join(config["included_models"]),
    ])
    selected_future_model = st.selectbox("Forecast with", model_options,
                                         index=model_options.index(best_model), key=selector_key,
                                         help="Defaults to the validation winner. You can inspect other candidates without changing the selected research winner.")
    try:
        future_result = fit_and_forecast(prices, config["horizon"], selected_future_model,
                                         seasonal_period=config["seasonal_period"], sma_window=config["sma_window"],
                                         interval=frequency, auto_max_order=config["auto_max_order"],
                                         auto_max_seasonal_order=config["auto_max_seasonal_order"])
    except Exception as exc:
        st.error(f"Future forecast failed for {selected_future_model}: {type(exc).__name__}: {exc}")
        st.stop()
    future = future_result["forecast"]
    hist_n = 180 if frequency == "Daily" else 104 if frequency == "Weekly" else 60
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.tail(hist_n).index, y=prices.tail(hist_n).values, mode="lines", name="Historical adjusted close", line=dict(width=2)))
    if future_result["interval"] is not None:
        band = future_result["interval"]
        fig.add_trace(go.Scatter(x=future.index, y=band["Upper 95%"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=future.index, y=band["Lower 95%"], mode="lines", fill="tonexty", line=dict(width=0), opacity=.18, name="95% model interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=future.index, y=future.values, mode="lines+markers", name="Forecast", line=dict(width=3, dash="dash")))
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=25, b=10), xaxis_title="Observation date (approximate future calendar)", yaxis_title=f"Adjusted price ({currency or 'provider currency'})", legend=dict(orientation="h", y=1.03))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Future-fit specification:** {future_result['details']}")
    st.caption("Auto ARIMA/SARIMA orders can differ between the validation fit, the test-period refit, and the final full-history fit. The specification above is the one used for this future forecast.")
    forecast_df = pd.DataFrame({"Forecast date (approx.)": future.index, "Ticker": ticker, "Model": selected_future_model,
                                "Frequency": frequency, "Forecast adjusted close": future.values})
    if future_result["interval"] is not None:
        forecast_df["Lower 95%"] = future_result["interval"]["Lower 95%"].to_numpy()
        forecast_df["Upper 95%"] = future_result["interval"]["Upper 95%"].to_numpy()
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    st.download_button("Download future forecast CSV", forecast_df.to_csv(index=False).encode(),
                       file_name=f"{ticker.replace('^','').replace('.','_')}_forecast.csv", mime="text/csv")
    st.caption("Future dates are business-day / Friday / month-end approximations. Exchange-specific holidays are not modelled. Prediction intervals are available only for Auto ARIMA/SARIMA here and depend on model assumptions.")

with comparison_tab:
    st.subheader("Model comparison")
    st.write(f"**Winner by validation {config['metric']}:** {best_model}. The test period was not used to choose it.")
    st.caption(f"Chronological protocol: {len(train)} training observations → {len(validation)} validation observations → {len(test)} untouched test observations. Models are evaluated on the common price-level scale.")
    visible_cols = ["Model", "ARIMA order (p,d,q)", "Seasonal order (P,D,Q,m)", "Validation AICc",
                    f"Validation {config['metric']}", "Test-fit ARIMA order", "Test-fit seasonal order", "Test-fit AICc", "Val/test fit warning count",
                    "Test MAE", "Test RMSE", "Test MAPE (%)", "Test sMAPE (%)", "Test MASE", "Test Direction Accuracy (%)", "Parameters"]
    view = comparison[visible_cols].rename(columns={
        f"Validation {config['metric']}": f"Validation {config['metric']} (selection)",
        "Parameters": "Test-fit specification",
    })
    st.dataframe(view.style.format({c: "{:,.4f}" for c in view.columns if c not in {
        "Model", "Parameters", "Test-fit specification", "ARIMA order (p,d,q)", "Seasonal order (P,D,Q,m)",
        "Test-fit ARIMA order", "Test-fit seasonal order", "Val/test fit warning count"}}), use_container_width=True, hide_index=True)
    st.caption("Validation-fit orders are estimated on training data; Test-fit orders are estimated after refitting on training + validation. These can legitimately differ for automatic searches. Information criteria help compare compatible candidate fits; they are not cross-family test scores.")
    with st.expander("Information criteria (ARIMA-family fits where available)"):
        criteria_cols = ["Model", "Validation AIC", "Validation AICc", "Validation BIC", "Test-fit AIC", "Test-fit AICc", "Test-fit BIC"]
        st.dataframe(comparison[criteria_cols].style.format({c: "{:,.3f}" for c in criteria_cols[1:]}),
                     use_container_width=True, hide_index=True)

    chart_choice = st.radio("Holdout prediction chart", ["Validation block", "Final test block"], horizontal=True)
    pred_frame = result["validation_predictions"] if chart_choice == "Validation block" else result["test_predictions"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred_frame.index, y=pred_frame["Actual"], mode="lines", name=f"Actual {chart_choice.lower()}", line=dict(width=3, color="#111827")))
    display_models = comparison["Model"].head(3).tolist()
    for model in display_models:
        if model in pred_frame:
            fig.add_trace(go.Scatter(x=pred_frame.index, y=pred_frame[model], mode="lines", name=model, line=dict(width=1.8)))
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Date", yaxis_title=f"Adjusted price ({currency or 'provider currency'})")
    st.plotly_chart(fig, use_container_width=True)
    warning_models = {}
    for fit_stage, stage_fits in [("Validation fit", result["validation_fits"]), ("Test fit", result["test_fits"])]:
        for candidate, fitted in stage_fits.items():
            diag = fitted.get("diagnostics", {})
            messages = list(diag.get("fit_warnings", [])) + list(diag.get("failed_candidates", []))
            if messages:
                warning_models[f"{fit_stage} · {candidate}"] = messages
    if warning_models:
        with st.expander(f"Fit/search warnings ({sum(len(v) for v in warning_models.values())})"):
            st.caption("Warnings are shown rather than silently discarded. Some bounded candidate fits may fail while a valid candidate is still selected.")
            for candidate, messages in warning_models.items():
                st.markdown(f"**{candidate}**")
                for message in messages:
                    st.write(f"- {message}")

    st.download_button("Download model-comparison CSV", comparison.to_csv(index=False).encode(),
                       file_name=f"{ticker.replace('^','').replace('.','_')}_model_comparison.csv", mime="text/csv")
    test_export = result["test_predictions"].copy()
    test_export.index.name = "Date"
    st.download_button("Download aligned test predictions CSV", test_export.to_csv().encode(),
                       file_name=f"{ticker.replace('^','').replace('.','_')}_test_predictions.csv", mime="text/csv")
    if result["skipped"]:
        with st.expander(f"Models skipped or failed ({len(result['skipped'])})"):
            for name, reason in result["skipped"].items():
                st.write(f"**{name}:** {reason}")

with diagnostics_tab:
    st.subheader("ATSA diagnostics")
    st.markdown("#### Stationarity: Augmented Dickey–Fuller test")
    diag_prices = result["train_validation"]
    st.caption("Diagnostics use training + validation observations only; final-test outcomes do not inform model selection.")
    try:
        adf = adf_table(diag_prices)
        st.dataframe(adf, use_container_width=True, hide_index=True)
        st.caption("ADF null hypothesis: the tested series has a unit root. p < 0.05 is evidence against the null; failure to reject does not prove non-stationarity.")
    except Exception as exc:
        st.warning(f"ADF diagnostics unavailable: {type(exc).__name__}: {exc}")

    st.markdown("#### Classical decomposition (exploratory)")
    show_decomp = st.checkbox("Show trend / seasonal / remainder decomposition", value=False,
                              help="Exploratory decomposition on training + validation only; it does not select a model.")
    if show_decomp:
        decomp_kind = st.selectbox("Decomposition form", ["additive", "multiplicative"], index=0)
        try:
            decomp = classical_decomposition(diag_prices, config["seasonal_period"], decomp_kind)
            decomp_fig = go.Figure()
            for component in ["Observed", "Trend", "Seasonal", "Remainder"]:
                decomp_fig.add_trace(go.Scatter(x=decomp.index, y=decomp[component], mode="lines", name=component))
            decomp_fig.update_layout(title=f"Classical {decomp_kind} decomposition · period {config['seasonal_period']}",
                                     height=420, margin=dict(l=10, r=10, t=35, b=10), xaxis_title="Date")
            st.plotly_chart(decomp_fig, use_container_width=True)
            st.caption("Decomposition is descriptive, not evidence that seasonality will persist or that a seasonal model will forecast better.")
        except Exception as exc:
            st.warning(f"Decomposition unavailable: {type(exc).__name__}: {exc}")

    st.markdown("#### ACF and PACF of simple returns")
    ret = diag_prices.pct_change().dropna()
    try:
        corr, bound = acf_pacf_table(ret, max_lags=30)
        left, right = st.columns(2)
        acf_fig = go.Figure(go.Bar(x=corr["Lag"], y=corr["ACF"], name="ACF"))
        pacf_fig = go.Figure(go.Bar(x=corr["Lag"], y=corr["PACF"], name="PACF"))
        for f in (acf_fig, pacf_fig):
            f.add_hline(y=bound, line_dash="dash", line_color="gray")
            f.add_hline(y=-bound, line_dash="dash", line_color="gray")
            f.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Lag")
        acf_fig.update_layout(title="ACF")
        pacf_fig.update_layout(title="PACF")
        left.plotly_chart(acf_fig, use_container_width=True)
        right.plotly_chart(pacf_fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"ACF/PACF unavailable: {type(exc).__name__}: {exc}")

    st.markdown("#### Residual diagnostics for the selected model")
    residuals = result["residuals"].get(best_model, np.asarray([]))
    test_fit = result["test_fits"].get(best_model, {})
    order = test_fit.get("arima_order")
    seasonal_order = test_fit.get("seasonal_order")
    model_df = 0
    if order:
        model_df += order[0] + order[2]
    if seasonal_order:
        model_df += seasonal_order[0] + seasonal_order[2]
    lb = ljung_box_test(residuals, model_df=model_df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Ljung–Box lag", "—" if pd.isna(lb["Lag"]) else int(lb["Lag"]))
    c2.metric("Q statistic", "—" if pd.isna(lb["Q statistic"]) else f"{lb['Q statistic']:.3f}")
    c3.metric("p-value", "—" if pd.isna(lb["p-value"]) else f"{lb['p-value']:.4f}")
    st.info(lb["Interpretation"] + ". A non-significant result does not prove residuals are perfectly white noise.")
    res_fig = go.Figure(go.Scatter(y=residuals, mode="lines", name="Residual"))
    res_fig.add_hline(y=0, line_dash="dash", line_color="gray")
    res_fig.update_layout(title=f"Training residuals from test-period fit: {best_model}", height=300, margin=dict(l=10, r=10, t=35, b=10), xaxis_title="Observation", yaxis_title="Residual")
    st.plotly_chart(res_fig, use_container_width=True)

with history_tab:
    st.subheader("Historical adjusted prices")
    st.dataframe(prices.rename("Adjusted Close (auto-adjusted)").to_frame().tail(300), use_container_width=True)
    st.download_button("Download historical-price CSV", prices.rename("Adjusted Close (auto-adjusted)").to_frame().to_csv().encode(),
                       file_name=f"{ticker.replace('^','').replace('.','_')}_history.csv", mime="text/csv")
    st.caption("Data retrieved through Yahoo Finance using yfinance. Missing observations are dropped; weekends and exchange holidays are not forward-filled.")

st.divider()
st.markdown("<div class='small-note'>Educational research application, not financial advice. Equity prices are noisy, affected by events and regime changes, and may not be predictable from historical prices alone. Model rankings are specific to the ticker, frequency, sample period, forecast horizon, evaluation protocol and chosen metric.</div>", unsafe_allow_html=True)

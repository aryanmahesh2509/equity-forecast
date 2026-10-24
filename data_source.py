"""Yahoo Finance data-access helpers; presentation and caching stay in app.py."""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from evaluation import clean_price_series


def download_prices(ticker: str, period: str, interval_label: str) -> pd.Series:
    """Download auto-adjusted Close observations; never fill non-trading dates."""
    interval_map = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
    raw = yf.download(ticker, period=period, interval=interval_map[interval_label], auto_adjust=True,
                       progress=False, threads=False, timeout=20)
    if raw is None or raw.empty:
        raise ValueError(f"Yahoo Finance returned no data for '{ticker}'. Verify the symbol and try a longer history.")
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Yahoo Finance response has no Close field.")
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close[ticker] if ticker in close.columns else close.iloc[:, 0]
    else:
        if "Close" not in raw.columns:
            raise ValueError("Yahoo Finance response has no Close field.")
        close = raw["Close"]
    close = pd.Series(close, name=ticker)
    close.index = pd.to_datetime(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close = clean_price_series(close, min_obs=40)
    close.name = ticker
    return close


def search_yahoo_tickers(query: str) -> tuple[list[dict[str, str]], str | None]:
    """Search Yahoo Finance for quote symbols by company name or ticker."""
    if len(query.strip()) < 2:
        return [], None
    try:
        search = yf.Search(query.strip(), max_results=15, news_count=0)
        results, seen = [], set()
        for quote in (getattr(search, "quotes", []) or []):
            symbol = str(quote.get("symbol", "")).strip().upper()
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            name = str(quote.get("longname") or quote.get("shortname") or quote.get("displayName") or symbol)
            exchange = str(quote.get("exchDisp") or quote.get("exchange") or "")
            results.append({"label": f"{name} ({symbol})" + (f" · {exchange}" if exchange else ""), "symbol": symbol})
        return results, None
    except Exception as exc:
        return [], f"Yahoo Finance symbol search failed: {type(exc).__name__}: {exc}"


def get_currency(ticker: str) -> str | None:
    """Best-effort currency from Yahoo Finance metadata; return None rather than guessing."""
    try:
        info = yf.Ticker(ticker).fast_info
        currency = info.get("currency") if hasattr(info, "get") else None
        if not currency:
            currency = getattr(info, "currency", None)
        return str(currency) if currency else None
    except Exception:
        return None

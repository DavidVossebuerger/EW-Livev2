"""Market data adapters built on top of yfinance."""
from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_SECONDS = 300
_FALLBACK_RISK_FREE = 0.02


@dataclass
class _CacheEntry:
    value: float
    timestamp: float


_CACHE: Dict[Tuple[str, str], _CacheEntry] = {}


def _now() -> float:
    return time.time()


def _cache_key(namespace: str, identifier: str = "") -> Tuple[str, str]:
    return namespace, identifier


def _read_cache(key: Tuple[str, str]) -> Optional[float]:
    entry = _CACHE.get(key)
    if not entry:
        return None
    if _now() - entry.timestamp <= CACHE_SECONDS:
        return entry.value
    _CACHE.pop(key, None)
    return None


def _write_cache(key: Tuple[str, str], value: float) -> None:
    _CACHE[key] = _CacheEntry(value=value, timestamp=_now())


def _normalize_ticker(ticker: str) -> str:
    if not ticker or not ticker.strip():
        raise ValueError("ticker must be a non-empty string")
    return ticker.strip().upper()


def _fetch_history(
    ticker: str,
    *,
    period: str = "30d",
    start: Optional[dt.datetime] = None,
    end: Optional[dt.datetime] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    history_kwargs: Dict[str, object] = {"interval": interval}
    if start or end:
        if start:
            history_kwargs["start"] = start
        if end:
            history_kwargs["end"] = end
    else:
        history_kwargs["period"] = period

    try:
        data = yf.Ticker(ticker).history(**history_kwargs)
    except Exception as exc:  # pragma: no cover - network errors
        raise RuntimeError(f"Failed to download data for {ticker}: {exc}") from exc

    if data.empty:
        raise RuntimeError(f"No market data returned for {ticker}.")
    return data


def get_stock_price(ticker: str, *, as_of: Optional[dt.datetime] = None) -> float:
    """Fetch the latest available closing price for *ticker*."""

    norm_ticker = _normalize_ticker(ticker)
    cache_key = _cache_key("price", norm_ticker)
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    start = end = None
    if as_of is not None:
        end = as_of + dt.timedelta(days=1)
        start = as_of - dt.timedelta(days=5)

    data = _fetch_history(norm_ticker, period="5d", start=start, end=end)
    price = float(data["Close"].dropna().iloc[-1])
    _write_cache(cache_key, price)
    return price


def get_historical_volatility(ticker: str, period: int = 30) -> float:
    """Return annualised volatility based on daily log returns."""

    if period < 2:
        raise ValueError("period must be at least 2 trading days")
    norm_ticker = _normalize_ticker(ticker)
    cache_key = _cache_key("hist_vol", f"{norm_ticker}:{period}")
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    data = _fetch_history(norm_ticker, period=f"{period + 5}d")
    close = data["Close"].dropna().tail(period + 1)
    if close.shape[0] < 2:
        raise RuntimeError("Insufficient data to compute historical volatility.")
    log_returns = np.log(close / close.shift(1)).dropna()
    volatility = float(log_returns.std(ddof=1) * np.sqrt(252))
    _write_cache(cache_key, volatility)
    return volatility


def get_risk_free_rate(default: float = _FALLBACK_RISK_FREE) -> float:
    """Return a proxy for the risk-free rate using the 10Y Treasury yield."""

    cache_key = _cache_key("risk_free", "^TNX")
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    try:
        data = _fetch_history("^TNX", period="5d")
        rate = float(data["Close"].dropna().iloc[-1]) / 100.0
    except Exception:  # pragma: no cover - fallback path
        rate = float(default)

    _write_cache(cache_key, rate)
    return rate


def _clear_cache() -> None:
    """Utility hook for unit tests to reset the in-memory cache."""

    _CACHE.clear()


def get_price_history(ticker: str, window_days: int = 90) -> pd.Series:
    """Return a pandas Series of closing prices for *ticker* over the last *window_days*."""

    if window_days < 2:
        raise ValueError("window_days must be at least 2")

    norm_ticker = _normalize_ticker(ticker)
    data = _fetch_history(norm_ticker, period=f"{window_days + 5}d")
    close = data["Close"].dropna().tail(window_days)
    if close.empty:
        raise RuntimeError("No price history available for requested window.")
    close.index = pd.to_datetime(close.index)
    return close

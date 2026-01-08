"""Tests for the market data fetcher utilities."""

import pandas as pd
import numpy as np
import pytest

from src.data import data_fetcher


@pytest.fixture(autouse=True)
def clear_cache() -> None:
    data_fetcher._clear_cache()
    yield
    data_fetcher._clear_cache()


def test_get_stock_price_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def fake_fetch(ticker: str, **_: object) -> pd.DataFrame:
        calls["count"] += 1
        return pd.DataFrame({"Close": [150.0]})

    monkeypatch.setattr(data_fetcher, "_fetch_history", fake_fetch)

    first = data_fetcher.get_stock_price("aapl")
    second = data_fetcher.get_stock_price("AAPL")

    assert first == pytest.approx(150.0)
    assert second == pytest.approx(150.0)
    assert calls["count"] == 1


def test_historical_volatility_matches_expected(monkeypatch: pytest.MonkeyPatch) -> None:
    closes = pd.Series([100.0, 101.0, 102.5, 101.5, 103.0])
    expected = float(np.log(closes / closes.shift(1)).dropna().std(ddof=1) * np.sqrt(252))

    def fake_fetch(ticker: str, **_: object) -> pd.DataFrame:
        return pd.DataFrame({"Close": closes})

    monkeypatch.setattr(data_fetcher, "_fetch_history", fake_fetch)

    vol = data_fetcher.get_historical_volatility("msft", period=4)
    assert vol == pytest.approx(expected, rel=1e-12)


def test_risk_free_rate_returns_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_fetch(*_: object, **__: object) -> pd.DataFrame:  # type: ignore[return-value]
        raise RuntimeError("boom")

    monkeypatch.setattr(data_fetcher, "_fetch_history", failing_fetch)

    rate = data_fetcher.get_risk_free_rate(default=0.015)
    assert rate == pytest.approx(0.015)


def test_get_price_history_returns_recent_window(monkeypatch: pytest.MonkeyPatch) -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    closes = pd.Series([100, 102, 101, 103, 104, 105], index=dates)

    def fake_fetch(ticker: str, **_: object) -> pd.DataFrame:
        return pd.DataFrame({"Close": closes})

    monkeypatch.setattr(data_fetcher, "_fetch_history", fake_fetch)

    history = data_fetcher.get_price_history("spy", window_days=3)

    assert isinstance(history, pd.Series)
    assert list(history.index) == list(dates[-3:])
    assert list(history.values) == [103, 104, 105]

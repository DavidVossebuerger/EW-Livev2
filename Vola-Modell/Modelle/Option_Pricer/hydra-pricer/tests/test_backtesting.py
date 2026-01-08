"""Tests for delta-hedging backtests."""

import pandas as pd
import pytest

from src.analytics.backtesting import BacktestResult, delta_hedge_backtest


def test_delta_hedge_backtest_constant_prices_has_small_pnl() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.Series(100.0, index=dates)

    result = delta_hedge_backtest(
        prices=prices,
        strike=100.0,
        rate=0.0,
        volatility=0.2,
        maturity_days=5,
        option_type="call",
    )

    assert isinstance(result, BacktestResult)
    assert len(result.history) == len(prices)
    assert result.pnl == pytest.approx(0.0, abs=1.0)


def test_delta_hedge_backtest_raises_for_empty_prices() -> None:
    with pytest.raises(ValueError):
        delta_hedge_backtest(
            prices=pd.Series(dtype=float),
            strike=100.0,
            rate=0.01,
            volatility=0.2,
            maturity_days=10,
            option_type="call",
        )

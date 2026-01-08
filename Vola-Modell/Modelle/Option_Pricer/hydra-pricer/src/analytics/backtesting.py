"""Simple backtesting utilities for options strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.models.black_scholes import BlackScholesModel

OptionType = Literal["call", "put"]


@dataclass
class BacktestResult:
    """Summary of a delta-hedging backtest."""

    pnl: float
    final_portfolio_value: float
    history: pd.DataFrame


def delta_hedge_backtest(
    prices: pd.Series,
    strike: float,
    rate: float,
    volatility: float,
    maturity_days: int,
    option_type: OptionType = "call",
) -> BacktestResult:
    """Simulate daily delta-hedging for a single European option."""

    if prices.empty:
        raise ValueError("prices series must contain data")
    dates = prices.index
    dt = 1 / 252

    cash = 0.0
    delta_prev = 0.0
    shares = 0.0
    option_values = []
    deltas = []
    cash_history = []
    shares_history = []

    for i, (date, spot) in enumerate(prices.items()):
        remaining = max((maturity_days - i) / 252, 1e-6)
        model = BlackScholesModel(spot=float(spot), strike=strike, maturity=remaining, rate=rate, volatility=volatility)
        option_value = model.call_price() if option_type == "call" else model.put_price()
        delta = model.delta(option_type)

        # Adjust hedge
        delta_change = delta - delta_prev
        cash -= delta_change * spot
        shares += delta_change
        cash *= np.exp(rate * dt)

        option_values.append(option_value)
        deltas.append(delta)
        cash_history.append(cash)
        shares_history.append(shares)

        delta_prev = delta

    final_portfolio = cash + shares * prices.iloc[-1] - option_values[-1]
    history = pd.DataFrame(
        {
            "spot": prices.values,
            "option_value": option_values,
            "delta": deltas,
            "shares": shares_history,
            "cash": cash_history,
        },
        index=dates,
    )
    return BacktestResult(pnl=float(final_portfolio), final_portfolio_value=float(final_portfolio), history=history)

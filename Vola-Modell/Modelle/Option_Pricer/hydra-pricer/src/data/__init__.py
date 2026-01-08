"""Data access layer."""

from .data_fetcher import get_historical_volatility, get_price_history, get_risk_free_rate, get_stock_price

__all__ = [
    "get_stock_price",
    "get_historical_volatility",
    "get_price_history",
    "get_risk_free_rate",
]

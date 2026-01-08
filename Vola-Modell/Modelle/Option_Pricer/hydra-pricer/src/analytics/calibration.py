"""Implied volatility calibration utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.models.black_scholes import BlackScholesModel

OptionType = Literal["call", "put"]


@dataclass
class CalibrationResult:
    """Container with calibration summary."""

    implied_vol: float
    iterations: int
    converged: bool


def implied_volatility(
    option_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: OptionType = "call",
    initial_guess: float = 0.2,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> CalibrationResult:
    """Compute Black-Scholes implied volatility using Newton-Raphson."""

    sigma = max(initial_guess, 1e-4)
    option_type = option_type.lower()

    for iteration in range(1, max_iter + 1):
        model = BlackScholesModel(spot=spot, strike=strike, maturity=maturity, rate=rate, volatility=sigma)
        model_price = model.call_price() if option_type == "call" else model.put_price()
        diff = model_price - option_price
        if abs(diff) < tol:
            return CalibrationResult(implied_vol=sigma, iterations=iteration, converged=True)

        vega = model.vega()
        if vega < 1e-8:
            break
        sigma -= diff / vega
        if sigma <= 0:
            sigma = 1e-4

    return CalibrationResult(implied_vol=sigma, iterations=iteration, converged=False)

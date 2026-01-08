"""Tests for implied volatility calibration."""

import pytest

from src.analytics.calibration import implied_volatility
from src.models.black_scholes import BlackScholesModel


def test_implied_volatility_recovers_reference_sigma() -> None:
    reference_sigma = 0.25
    model = BlackScholesModel(spot=100.0, strike=100.0, maturity=1.0, rate=0.02, volatility=reference_sigma)
    market_price = model.call_price()

    result = implied_volatility(
        option_price=market_price,
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.02,
        option_type="call",
        initial_guess=0.2,
    )

    assert result.converged
    assert result.implied_vol == pytest.approx(reference_sigma, rel=1e-3)

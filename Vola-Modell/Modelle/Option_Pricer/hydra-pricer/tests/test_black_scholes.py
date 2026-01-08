"""Regression tests for the Black-Scholes model."""

import pytest

from src.greeks import compute_bsm_greeks
from src.models.black_scholes import BlackScholesModel


def _reference_model() -> BlackScholesModel:
    return BlackScholesModel(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2)


def test_black_scholes_call_price_matches_reference() -> None:
    model = _reference_model()
    assert model.call_price() == pytest.approx(10.4506, rel=1e-4)


def test_black_scholes_put_price_matches_reference() -> None:
    model = _reference_model()
    assert model.put_price() == pytest.approx(5.5735, rel=1e-4)


def test_black_scholes_greeks_match_reference_values() -> None:
    model = _reference_model()
    assert model.delta("call") == pytest.approx(0.63683, rel=1e-4)
    assert model.delta("put") == pytest.approx(-0.36317, rel=1e-4)
    assert model.gamma() == pytest.approx(0.018762, rel=1e-4)
    assert model.vega() == pytest.approx(37.52403, rel=1e-4)
    assert model.theta("call") == pytest.approx(-0.01757268, rel=1e-6)
    assert model.theta("put") == pytest.approx(-0.00454214, rel=1e-6)
    assert model.rho("call") == pytest.approx(53.23248, rel=1e-4)
    assert model.rho("put") == pytest.approx(-41.89046, rel=1e-4)


def test_compute_bsm_greeks_wrapper_matches_model_methods() -> None:
    model = _reference_model()
    greeks = compute_bsm_greeks(model, option_type="call")
    assert greeks.delta == pytest.approx(model.delta("call"), rel=1e-10)
    assert greeks.gamma == pytest.approx(model.gamma(), rel=1e-10)
    assert greeks.vega == pytest.approx(model.vega(), rel=1e-10)
    assert greeks.theta == pytest.approx(model.theta("call"), rel=1e-10)
    assert greeks.rho == pytest.approx(model.rho("call"), rel=1e-10)

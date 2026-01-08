"""Tests for the CRR binomial model."""

import pytest

from src.greeks import compute_binomial_greeks, compute_bsm_greeks
from src.models.binomial import BinomialModel
from src.models.black_scholes import BlackScholesModel


def _bsm_reference() -> BlackScholesModel:
    return BlackScholesModel(spot=100.0, strike=100.0, maturity=1.0, rate=0.05, volatility=0.2)


def test_binomial_matches_bsm_for_european_call() -> None:
    bsm = _bsm_reference()
    binomial = BinomialModel(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=500,
        option_type="call",
        exercise="european",
    )
    assert binomial.price() == pytest.approx(bsm.call_price(), rel=1e-3)


def test_american_put_at_least_european_value() -> None:
    european = BinomialModel(
        spot=90.0,
        strike=100.0,
        maturity=1.0,
        rate=0.03,
        volatility=0.25,
        steps=200,
        option_type="put",
        exercise="european",
    )
    american = BinomialModel(
        spot=90.0,
        strike=100.0,
        maturity=1.0,
        rate=0.03,
        volatility=0.25,
        steps=200,
        option_type="put",
        exercise="american",
    )
    assert american.price() >= european.price()


def test_binomial_greeks_approximate_bsm_values() -> None:
    binomial = BinomialModel(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        steps=1_000,
        option_type="call",
        exercise="european",
    )
    greeks_bin = compute_binomial_greeks(binomial)
    greeks_bsm = compute_bsm_greeks(_bsm_reference(), option_type="call")

    assert greeks_bin.delta == pytest.approx(greeks_bsm.delta, rel=3e-2)
    assert greeks_bin.gamma == pytest.approx(greeks_bsm.gamma, rel=5e-2)
    assert greeks_bin.vega == pytest.approx(greeks_bsm.vega, rel=1e-1)
    assert greeks_bin.theta == pytest.approx(greeks_bsm.theta, rel=2e-1)
    assert greeks_bin.rho == pytest.approx(greeks_bsm.rho, rel=1e-1)

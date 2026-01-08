"""Tests for the Monte Carlo pricing module."""

import numpy as np
import pytest

from src.models.black_scholes import BlackScholesModel
from src.models.monte_carlo import MonteCarloModel


def test_monte_carlo_converges_for_european_call() -> None:
    model = MonteCarloModel(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
        n_simulations=50_000,
        n_steps=252,
        payoff="european_call",
        rng_seed=123,
    )
    mc_price = model.price()
    bsm_price = BlackScholesModel(
        spot=100.0,
        strike=100.0,
        maturity=1.0,
        rate=0.05,
        volatility=0.2,
    ).call_price()
    assert mc_price == pytest.approx(bsm_price, abs=0.35)


def test_monte_carlo_summary_contains_confidence_interval() -> None:
    model = MonteCarloModel(
        spot=120.0,
        strike=110.0,
        maturity=0.5,
        rate=0.02,
        volatility=0.15,
        n_simulations=20_000,
        n_steps=126,
        payoff="european_put",
        rng_seed=7,
    )
    summary = model.summarize()
    assert summary["price"] > 0
    assert summary["stderr"] > 0
    ci_low, ci_high = summary["conf_interval"]
    assert ci_low < summary["price"] < ci_high
    assert summary["n_simulations"] == 20_000


@pytest.mark.parametrize(
    "payoff_type,expected",
    [
        ("european_call", 25.0),
        ("european_put", 0.0),
        ("asian_call", 8.75),
        ("asian_put", 0.0),
        ("lookback_call", 25.0),
        ("lookback_put", 5.0),
    ],
)
def test_payoff_calculations_match_expected(payoff_type: str, expected: float) -> None:
    model = MonteCarloModel(
        spot=100.0,
        strike=105.0,
        maturity=1.0,
        rate=0.0,
        volatility=0.2,
        n_simulations=1,
        n_steps=3,
        payoff=payoff_type,
        rng_seed=1,
    )
    sample_paths = np.array([[100.0, 115.0, 110.0, 130.0]])
    payoffs = model._payoffs(sample_paths)
    assert payoffs.shape == (1,)
    assert payoffs[0] == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    "payoff_type,paths,expected",
    [
        ("down_and_out_call", np.array([[100.0, 92.0, 97.0, 110.0]]), 0.0),
        ("down_and_out_call", np.array([[100.0, 105.0, 108.0, 115.0]]), 10.0),
        ("down_and_in_call", np.array([[100.0, 92.0, 97.0, 110.0]]), 5.0),
        ("down_and_in_call", np.array([[100.0, 105.0, 108.0, 115.0]]), 0.0),
        ("up_and_out_put", np.array([[100.0, 130.0, 125.0, 90.0]]), 0.0),
        ("up_and_in_put", np.array([[100.0, 130.0, 125.0, 90.0]]), 15.0),
    ],
)
def test_barrier_payoffs_respect_barrier_logic(payoff_type: str, paths: np.ndarray, expected: float) -> None:
    model = MonteCarloModel(
        spot=100.0,
        strike=105.0,
        maturity=1.0,
        rate=0.0,
        volatility=0.2,
        n_simulations=1,
        n_steps=paths.shape[1] - 1,
        payoff=payoff_type,
        rng_seed=1,
        barrier_level=95.0 if "down" in payoff_type else 120.0,
    )
    payoffs = model._payoffs(paths)
    assert payoffs.shape == (paths.shape[0],)
    assert payoffs[0] == pytest.approx(expected, rel=1e-9)

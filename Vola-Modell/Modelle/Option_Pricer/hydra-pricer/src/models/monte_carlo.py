"""Monte Carlo pricing model implementation."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Dict, Literal, Optional, Tuple

import numpy as np

PayoffType = Literal[
    "european_call",
    "european_put",
    "asian_call",
    "asian_put",
    "lookback_call",
    "lookback_put",
    "down_and_out_call",
    "down_and_out_put",
    "up_and_out_call",
    "up_and_out_put",
    "down_and_in_call",
    "down_and_in_put",
    "up_and_in_call",
    "up_and_in_put",
]


@dataclass
class MonteCarloModel:
    """Vectorised Monte-Carlo pricer supporting several payoff types."""

    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    n_simulations: int = 10_000
    n_steps: int = 252
    payoff: PayoffType = "european_call"
    rng_seed: Optional[int] = 42
    barrier_level: Optional[float] = None

    def __post_init__(self) -> None:
        self._validate_inputs()
        self._dt = self.maturity / self.n_steps
        self._rng = np.random.default_rng(self.rng_seed)

    def _validate_inputs(self) -> None:
        if self.spot <= 0 or self.strike <= 0:
            raise ValueError("Spot and strike must be positive.")
        if self.maturity <= 0:
            raise ValueError("Maturity must be positive.")
        if self.volatility <= 0:
            raise ValueError("Volatility must be positive.")
        if self.n_simulations <= 0 or self.n_steps <= 0:
            raise ValueError("Simulation and step counts must be positive.")
        barrier_payoffs = {
            "down_and_out_call",
            "down_and_out_put",
            "up_and_out_call",
            "up_and_out_put",
            "down_and_in_call",
            "down_and_in_put",
            "up_and_in_call",
            "up_and_in_put",
        }
        if self.payoff in barrier_payoffs and (self.barrier_level is None or self.barrier_level <= 0):
            raise ValueError("Barrier payoffs require barrier_level > 0.")

    def simulate_paths(self) -> np.ndarray:
        """Simulate GBM asset paths using log-Euler discretization."""

        drift = (self.rate - 0.5 * self.volatility**2) * self._dt
        diffusion = self.volatility * sqrt(self._dt)
        shocks = self._rng.standard_normal((self.n_simulations, self.n_steps))
        log_returns = drift + diffusion * shocks
        cumulative = np.cumsum(log_returns, axis=1)
        paths = np.empty((self.n_simulations, self.n_steps + 1))
        paths[:, 0] = self.spot
        paths[:, 1:] = self.spot * np.exp(cumulative)
        return paths

    def _payoffs(self, paths: np.ndarray) -> np.ndarray:
        payoff_type = self.payoff.lower()
        terminal = paths[:, -1]

        if payoff_type == "european_call":
            return np.maximum(terminal - self.strike, 0.0)
        if payoff_type == "european_put":
            return np.maximum(self.strike - terminal, 0.0)

        average_price = paths.mean(axis=1)
        if payoff_type == "asian_call":
            return np.maximum(average_price - self.strike, 0.0)
        if payoff_type == "asian_put":
            return np.maximum(self.strike - average_price, 0.0)

        path_max = paths.max(axis=1)
        path_min = paths.min(axis=1)
        if payoff_type == "lookback_call":
            return np.maximum(path_max - self.strike, 0.0)
        if payoff_type == "lookback_put":
            return np.maximum(self.strike - path_min, 0.0)

        if payoff_type in {
            "down_and_out_call",
            "down_and_out_put",
            "up_and_out_call",
            "up_and_out_put",
            "down_and_in_call",
            "down_and_in_put",
            "up_and_in_call",
            "up_and_in_put",
        }:
            if self.barrier_level is None:
                raise ValueError("barrier_level must be provided for barrier payoffs")

            is_call = payoff_type.endswith("_call")
            base_payoff = np.maximum(terminal - self.strike, 0.0) if is_call else np.maximum(self.strike - terminal, 0.0)

            touched: np.ndarray
            if payoff_type.startswith("down"):
                touched = path_min <= self.barrier_level
            else:
                touched = path_max >= self.barrier_level

            is_knock_in = "_in_" in payoff_type
            active_mask = touched if is_knock_in else ~touched
            return np.where(active_mask, base_payoff, 0.0)

        raise ValueError("Unsupported payoff type provided.")

    def price(self) -> float:
        """Estimate the discounted expected payoff."""

        paths = self.simulate_paths()
        payoffs = self._payoffs(paths)
        discount = exp(-self.rate * self.maturity)
        return float(discount * payoffs.mean())

    def summarize(self) -> Dict[str, float | Tuple[float, float] | int]:
        """Return price, standard error and 95% CI for diagnostics."""

        paths = self.simulate_paths()
        payoffs = self._payoffs(paths)
        discount = exp(-self.rate * self.maturity)
        discounted = discount * payoffs
        price = float(discounted.mean())
        stderr = float(discounted.std(ddof=1) / np.sqrt(self.n_simulations))
        ci_half_width = 1.96 * stderr
        return {
            "price": price,
            "stderr": stderr,
            "conf_interval": (price - ci_half_width, price + ci_half_width),
            "n_simulations": self.n_simulations,
        }

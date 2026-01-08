"""Cox-Ross-Rubinstein binomial model implementation."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Literal

import numpy as np


OptionType = Literal["call", "put"]
ExerciseStyle = Literal["european", "american"]


@dataclass
class BinomialModel:
    """CRR tree supporting European and American payoff styles.

    Args:
        spot: Current underlying price.
        strike: Option strike price.
        maturity: Time to maturity in years.
        rate: Continuously compounded risk-free rate.
        volatility: Annualised volatility.
        steps: Number of binomial time steps.
        option_type: Either "call" or "put".
        exercise: Either "european" or "american".
    """

    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    steps: int
    option_type: OptionType = "call"
    exercise: ExerciseStyle = "american"

    def __post_init__(self) -> None:
        self.option_type = self.option_type.lower()
        self.exercise = self.exercise.lower()

        if self.steps <= 0:
            raise ValueError("Number of steps must be positive.")
        if self.spot <= 0 or self.strike <= 0:
            raise ValueError("Spot and strike must be positive.")
        if self.maturity <= 0:
            raise ValueError("Maturity must be positive.")
        if self.volatility <= 0:
            raise ValueError("Volatility must be positive.")

        self._dt = self.maturity / self.steps
        self._u = float(np.exp(self.volatility * sqrt(self._dt)))
        self._d = 1.0 / self._u
        growth = exp(self.rate * self._dt)
        self._p = (growth - self._d) / (self._u - self._d)
        if not (0 < self._p < 1):
            raise ValueError("Risk-neutral probability out of bounds. Adjust inputs.")

    def _asset_prices(self, step: int) -> np.ndarray:
        j = np.arange(step + 1)
        return self.spot * (self._u ** j) * (self._d ** (step - j))

    def _intrinsic(self, prices: np.ndarray) -> np.ndarray:
        if self.option_type == "call":
            return np.maximum(prices - self.strike, 0.0)
        if self.option_type == "put":
            return np.maximum(self.strike - prices, 0.0)
        raise ValueError("option_type must be 'call' or 'put'.")

    def price(self) -> float:
        """Return the option price via backward induction."""

        discount = exp(-self.rate * self._dt)
        prices = self._asset_prices(self.steps)
        option_values = self._intrinsic(prices)

        for step in range(self.steps, 0, -1):
            option_values = discount * (
                self._p * option_values[1:] + (1 - self._p) * option_values[:-1]
            )
            if self.exercise == "american":
                prices = self._asset_prices(step - 1)
                option_values = np.maximum(option_values, self._intrinsic(prices))

        return float(option_values[0])

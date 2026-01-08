"""Black-Scholes-Merton pricing engine implementation."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt
from typing import Dict, Tuple

from scipy.stats import norm


@dataclass
class BlackScholesModel:
    """Closed-form Black-Scholes pricer with analytic Greeks.

    Args:
        spot: Current underlying price (S).
        strike: Strike price (K).
        maturity: Time to maturity in years (T).
        rate: Continuously compounded risk-free rate (r).
        volatility: Annualised volatility (sigma).
    """

    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float

    def __post_init__(self) -> None:
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if self.spot <= 0 or self.strike <= 0:
            raise ValueError("Spot and strike must be positive.")
        if self.maturity <= 0:
            raise ValueError("Maturity must be positive.")
        if self.volatility <= 0:
            raise ValueError("Volatility must be positive.")

    def _d1_d2(self) -> Tuple[float, float]:
        tau_sqrt = sqrt(self.maturity)
        denom = self.volatility * tau_sqrt
        d1 = (
            log(self.spot / self.strike)
            + (self.rate + 0.5 * self.volatility**2) * self.maturity
        ) / denom
        d2 = d1 - self.volatility * tau_sqrt
        return d1, d2

    def call_price(self) -> float:
        """Calculate the Black-Scholes price for a European call option."""
        d1, d2 = self._d1_d2()
        discounted_strike = self.strike * exp(-self.rate * self.maturity)
        return self.spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)

    def put_price(self) -> float:
        """Calculate the Black-Scholes price for a European put option."""
        d1, d2 = self._d1_d2()
        discounted_strike = self.strike * exp(-self.rate * self.maturity)
        return discounted_strike * norm.cdf(-d2) - self.spot * norm.cdf(-d1)

    def delta(self, option_type: str = "call") -> float:
        """Return the option delta.

        Args:
            option_type: Either "call" or "put".
        """

        d1, _ = self._d1_d2()
        option_type = option_type.lower()
        if option_type == "call":
            return float(norm.cdf(d1))
        if option_type == "put":
            return float(norm.cdf(d1) - 1.0)
        raise ValueError("option_type must be 'call' or 'put'.")

    def gamma(self) -> float:
        """Return the option gamma."""

        d1, _ = self._d1_d2()
        return float(norm.pdf(d1) / (self.spot * self.volatility * sqrt(self.maturity)))

    def vega(self) -> float:
        """Return the option vega (per 1 volatility unit)."""

        d1, _ = self._d1_d2()
        return float(self.spot * norm.pdf(d1) * sqrt(self.maturity))

    def theta(self, option_type: str = "call") -> float:
        """Return the option theta per calendar day.

        Args:
            option_type: Either "call" or "put".
        """

        d1, d2 = self._d1_d2()
        pdf_term = (self.spot * norm.pdf(d1) * self.volatility) / (2 * sqrt(self.maturity))
        discounted_strike = self.strike * exp(-self.rate * self.maturity)
        option_type = option_type.lower()
        if option_type == "call":
            theta_annual = -pdf_term - self.rate * discounted_strike * norm.cdf(d2)
        elif option_type == "put":
            theta_annual = -pdf_term + self.rate * discounted_strike * norm.cdf(-d2)
        else:
            raise ValueError("option_type must be 'call' or 'put'.")
        return float(theta_annual / 365.0)

    def rho(self, option_type: str = "call") -> float:
        """Return the option rho."""

        _, d2 = self._d1_d2()
        discounted_factor = self.strike * self.maturity * exp(-self.rate * self.maturity)
        option_type = option_type.lower()
        if option_type == "call":
            return float(discounted_factor * norm.cdf(d2))
        if option_type == "put":
            return float(-discounted_factor * norm.cdf(-d2))
        raise ValueError("option_type must be 'call' or 'put'.")

    def get_all_greeks(self) -> Dict[str, float]:
        """Return a dictionary containing all primary Greeks."""

        return {
            "delta_call": self.delta("call"),
            "delta_put": self.delta("put"),
            "gamma": self.gamma(),
            "vega": self.vega(),
            "theta_call": self.theta("call"),
            "theta_put": self.theta("put"),
            "rho_call": self.rho("call"),
            "rho_put": self.rho("put"),
        }

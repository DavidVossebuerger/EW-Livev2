"""Greeks calculation utilities."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only for type checking
    from src.models.black_scholes import BlackScholesModel
    from src.models.binomial import BinomialModel


@dataclass
class GreeksResult:
    """Container for Greek sensitivities."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def compute_bsm_greeks(model: "BlackScholesModel", option_type: str = "call") -> GreeksResult:
    """Return analytic Greeks for a given Black-Scholes instance."""

    option_type = option_type.lower()
    delta = model.delta(option_type)
    gamma = model.gamma()
    vega = model.vega()
    theta = model.theta(option_type)
    rho = model.rho(option_type)
    return GreeksResult(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def compute_binomial_greeks(model: "BinomialModel") -> GreeksResult:
    """Approximate Greeks via finite differences for the binomial tree."""

    def reprice(**kwargs: float) -> float:
        bumped = replace(model, **kwargs)
        return bumped.price()

    price_0 = model.price()

    spot_h = max(model.spot * 5e-2, 1e-2)
    if model.spot - spot_h <= 0:
        spot_h = model.spot * 0.5
    price_spot_up = reprice(spot=model.spot + spot_h)
    price_spot_down = reprice(spot=model.spot - spot_h)
    delta = (price_spot_up - price_spot_down) / (2 * spot_h)
    gamma = (price_spot_up - 2 * price_0 + price_spot_down) / (spot_h**2)

    vol_h = max(model.volatility * 1e-3, 1e-4)
    if model.volatility - vol_h <= 0:
        vol_h = model.volatility * 0.5
    price_vol_up = reprice(volatility=model.volatility + vol_h)
    price_vol_down = reprice(volatility=model.volatility - vol_h)
    vega = (price_vol_up - price_vol_down) / (2 * vol_h)

    rate_h = max(abs(model.rate) * 1e-4, 1e-4)
    price_rate_up = reprice(rate=model.rate + rate_h)
    price_rate_down = reprice(rate=model.rate - rate_h)
    rho = (price_rate_up - price_rate_down) / (2 * rate_h)

    time_h = max(model.maturity * 1e-3, 1e-4)
    if model.maturity - time_h <= 0:
        time_h = model.maturity * 0.5
    price_time_up = reprice(maturity=model.maturity + time_h)
    price_time_down = reprice(maturity=model.maturity - time_h)
    theta_annual = (price_time_down - price_time_up) / (2 * time_h)
    theta = theta_annual / 365.0

    return GreeksResult(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def as_dict(result: GreeksResult) -> Dict[str, float]:
    """Convert a GreeksResult into a serializable dictionary."""
    return {
        "delta": result.delta,
        "gamma": result.gamma,
        "vega": result.vega,
        "theta": result.theta,
        "rho": result.rho,
    }

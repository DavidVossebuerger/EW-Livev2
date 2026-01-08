"""Utility helpers shared across modules."""
from __future__ import annotations

from typing import Iterable


def ensure_positive(value: float, name: str) -> None:
    """Validate that `value` is strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def moving_average(values: Iterable[float]) -> float:
    """Return the arithmetic mean of an iterable."""
    data = list(values)
    if not data:
        raise ValueError("values must not be empty")
    return sum(data) / len(data)

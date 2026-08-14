"""Risk model / measure registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from iqrp.app.risk.market.beta import beta
from iqrp.app.risk.market.volatility import realized_volatility
from iqrp.app.risk.tail.cvar import historical_cvar, monte_carlo_cvar, parametric_cvar
from iqrp.app.risk.tail.drawdown import max_drawdown
from iqrp.app.risk.tail.expected_shortfall import expected_shortfall
from iqrp.app.risk.tail.var import (
    filtered_historical_var,
    historical_var,
    monte_carlo_var,
    parametric_var,
)

_REGISTRY: dict[str, Callable[..., Any]] = {
    "historical_var": historical_var,
    "parametric_var": parametric_var,
    "monte_carlo_var": monte_carlo_var,
    "filtered_historical_var": filtered_historical_var,
    "historical_cvar": historical_cvar,
    "parametric_cvar": parametric_cvar,
    "monte_carlo_cvar": monte_carlo_cvar,
    "expected_shortfall": expected_shortfall,
    "realized_volatility": realized_volatility,
    "beta": beta,
    "max_drawdown": max_drawdown,
}


def register(name: str, fn: Callable[..., Any]) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("empty registry name")
    _REGISTRY[key] = fn


def get(name: str) -> Callable[..., Any]:
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown risk measure '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def available() -> list[str]:
    return sorted(_REGISTRY)


def clear_custom() -> None:
    """Remove non-builtin registrations (test helper)."""
    builtins = {
        "historical_var",
        "parametric_var",
        "monte_carlo_var",
        "filtered_historical_var",
        "historical_cvar",
        "parametric_cvar",
        "monte_carlo_cvar",
        "expected_shortfall",
        "realized_volatility",
        "beta",
        "max_drawdown",
    }
    for k in list(_REGISTRY):
        if k not in builtins:
            del _REGISTRY[k]

"""Portfolio estimator / method registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from iqrp.app.portfolio.covariance.ewma import ewma_covariance
from iqrp.app.portfolio.covariance.factor import factor_covariance
from iqrp.app.portfolio.covariance.robust import robust_covariance
from iqrp.app.portfolio.covariance.sample import sample_covariance
from iqrp.app.portfolio.covariance.shrinkage import ledoit_wolf_covariance, shrinkage_covariance
from iqrp.app.portfolio.expected_returns.black_litterman import black_litterman_posterior
from iqrp.app.portfolio.expected_returns.forecast import forecast_expected_returns
from iqrp.app.portfolio.expected_returns.historical import historical_expected_returns
from iqrp.app.portfolio.expected_returns.shrinkage import shrinkage_expected_returns

_REGISTRY: dict[str, Callable[..., Any]] = {
    "sample_covariance": sample_covariance,
    "ewma_covariance": ewma_covariance,
    "shrinkage_covariance": shrinkage_covariance,
    "ledoit_wolf_covariance": ledoit_wolf_covariance,
    "factor_covariance": factor_covariance,
    "robust_covariance": robust_covariance,
    "forecast_expected_returns": forecast_expected_returns,
    "historical_expected_returns": historical_expected_returns,
    "shrinkage_expected_returns": shrinkage_expected_returns,
    "black_litterman_posterior": black_litterman_posterior,
}

_BUILTINS = frozenset(_REGISTRY)


def register(name: str, fn: Callable[..., Any]) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("empty registry name")
    _REGISTRY[key] = fn


def get(name: str) -> Callable[..., Any]:
    key = name.strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown portfolio method '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def available() -> list[str]:
    return sorted(_REGISTRY)


def clear_custom() -> None:
    """Remove non-builtin registrations (test helper)."""
    for k in list(_REGISTRY):
        if k not in _BUILTINS:
            del _REGISTRY[k]

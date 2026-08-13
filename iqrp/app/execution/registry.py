"""Execution algorithm / component registry."""

from __future__ import annotations

from typing import Any, Callable, Type

from iqrp.app.execution.algorithms.adaptive import AdaptiveAlgorithm
from iqrp.app.execution.algorithms.arrival_price import ArrivalPriceAlgorithm
from iqrp.app.execution.algorithms.base import ExecutionAlgorithm
from iqrp.app.execution.algorithms.implementation_shortfall import ImplementationShortfallAlgorithm
from iqrp.app.execution.algorithms.limit import LimitAlgorithm
from iqrp.app.execution.algorithms.liquidity_seeking import LiquiditySeekingAlgorithm
from iqrp.app.execution.algorithms.market import MarketAlgorithm
from iqrp.app.execution.algorithms.opportunistic import OpportunisticAlgorithm
from iqrp.app.execution.algorithms.pov import POVAlgorithm
from iqrp.app.execution.algorithms.twap import TWAPAlgorithm
from iqrp.app.execution.algorithms.vwap import VWAPAlgorithm

AlgoFactory = Callable[..., ExecutionAlgorithm]

_ALGO_REGISTRY: dict[str, Type[ExecutionAlgorithm] | AlgoFactory] = {
    "twap": TWAPAlgorithm,
    "vwap": VWAPAlgorithm,
    "pov": POVAlgorithm,
    "is": ImplementationShortfallAlgorithm,
    "implementation_shortfall": ImplementationShortfallAlgorithm,
    "arrival": ArrivalPriceAlgorithm,
    "arrival_price": ArrivalPriceAlgorithm,
    "adaptive": AdaptiveAlgorithm,
    "market": MarketAlgorithm,
    "limit": LimitAlgorithm,
    "liquidity_seeking": LiquiditySeekingAlgorithm,
    "opportunistic": OpportunisticAlgorithm,
}

_BUILTINS = frozenset(_ALGO_REGISTRY)


def register_algorithm(name: str, factory: Type[ExecutionAlgorithm] | AlgoFactory) -> None:
    key = str(name).strip().lower()
    if not key:
        raise ValueError("empty algorithm name")
    _ALGO_REGISTRY[key] = factory


def get_algorithm(name: str, **kwargs: Any) -> ExecutionAlgorithm:
    key = str(name).strip().lower()
    if key not in _ALGO_REGISTRY:
        raise KeyError(f"Unknown execution algorithm '{name}'. Available: {sorted(_ALGO_REGISTRY)}")
    factory = _ALGO_REGISTRY[key]
    return factory(**kwargs)  # type: ignore[operator]


def available_algorithms() -> list[str]:
    return sorted(_ALGO_REGISTRY)


def clear_custom_algorithms() -> None:
    for k in list(_ALGO_REGISTRY):
        if k not in _BUILTINS:
            del _ALGO_REGISTRY[k]


__all__ = [
    "available_algorithms",
    "clear_custom_algorithms",
    "get_algorithm",
    "register_algorithm",
]

"""Function registry for signal types / evaluators.

CRITICAL: Registration does not imply alpha approval.
Statistical significance alone ≠ alpha.
Historical Sharpe alone cannot approve.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from iqrp.app.alpha.discovery.alternative import (
    alternative_change_signal,
    alternative_zscore_signal,
    sentiment_pressure_signal,
)
from iqrp.app.alpha.discovery.cross_sectional import (
    cross_sectional_rank_signal,
    cross_sectional_zscore_signal,
)
from iqrp.app.alpha.discovery.event_based import (
    earnings_drift_proxy,
    event_impulse_signal,
    surprise_signal,
)
from iqrp.app.alpha.discovery.time_series import (
    mean_reversion_signal,
    momentum_signal,
    trend_signal,
    volatility_signal,
    volume_signal,
)
from iqrp.app.alpha.research.decay import analyze_decay
from iqrp.app.alpha.research.evaluator import evaluate_signal
from iqrp.app.alpha.research.hit_rate import compute_hit_rate
from iqrp.app.alpha.research.information_coefficient import compute_ic
from iqrp.app.alpha.research.persistence import persistence_summary
from iqrp.app.alpha.research.rank_ic import compute_rank_ic
from iqrp.app.alpha.research.seasonality import analyze_seasonality
from iqrp.app.alpha.research.stability import analyze_stability

_REGISTRY: dict[str, Callable[..., Any]] = {
    # discovery / signal types
    "momentum": momentum_signal,
    "mean_reversion": mean_reversion_signal,
    "trend": trend_signal,
    "volatility": volatility_signal,
    "volume": volume_signal,
    "cross_sectional_rank": cross_sectional_rank_signal,
    "cross_sectional_zscore": cross_sectional_zscore_signal,
    "event_impulse": event_impulse_signal,
    "event_surprise": surprise_signal,
    "event_pead": earnings_drift_proxy,
    "alternative_zscore": alternative_zscore_signal,
    "alternative_change": alternative_change_signal,
    "sentiment": sentiment_pressure_signal,
    # evaluators / metrics
    "evaluate_signal": evaluate_signal,
    "compute_ic": compute_ic,
    "compute_rank_ic": compute_rank_ic,
    "compute_hit_rate": compute_hit_rate,
    "analyze_decay": analyze_decay,
    "analyze_stability": analyze_stability,
    "analyze_seasonality": analyze_seasonality,
    "persistence_summary": persistence_summary,
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
        raise KeyError(f"Unknown alpha registry entry '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def available() -> list[str]:
    return sorted(_REGISTRY)


def clear_custom() -> None:
    """Remove non-builtin registrations (test helper)."""
    for k in list(_REGISTRY):
        if k not in _BUILTINS:
            del _REGISTRY[k]

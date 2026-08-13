"""Dimension weighting schemes for the Risk Intelligence Ensemble.

Weights scale contribution while preserving each risk dimension's identity.
They never erase a dimension or replace multi-dimensional risk with a blind average.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from iqrp.app.risk.ensemble.config import EnsembleSettings
from iqrp.app.risk.ensemble.types import RISK_DIMENSIONS

WeightingScheme = Literal[
    "static",
    "risk_budget",
    "regime",
    "dynamic",
    "calibration",
    "stress",
    "user_defined",
]

DIMENSIONS = RISK_DIMENSIONS


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    out = {d: float(max(weights.get(d, 0.0), 0.0)) for d in DIMENSIONS}
    total = sum(out.values())
    if total <= 1e-12:
        equal = 1.0 / len(DIMENSIONS)
        return {d: equal for d in DIMENSIONS}
    return {d: out[d] / total for d in DIMENSIONS}


def static_weights(settings: EnsembleSettings) -> dict[str, float]:
    return _normalize_weights(dict(settings.static_weights))


def user_defined_weights(settings: EnsembleSettings, overrides: dict[str, float] | None = None) -> dict[str, float]:
    base = dict(settings.static_weights)
    base.update(dict(settings.user_defined_weights or {}))
    if overrides:
        base.update({str(k): float(v) for k, v in overrides.items()})
    return _normalize_weights(base)


def risk_budget_weights(
    settings: EnsembleSettings,
    *,
    dimension_scores: dict[str, float] | None = None,
) -> dict[str, float]:
    """Allocate more weight to dimensions consuming more of the risk budget."""
    base = static_weights(settings)
    scores = dimension_scores or {}
    tilted = {}
    for d in DIMENSIONS:
        s = float(np.clip(scores.get(d, base[d]), 0.0, 1.0))
        tilted[d] = base[d] * (0.5 + s)
    return _normalize_weights(tilted)


def regime_weights(
    settings: EnsembleSettings,
    *,
    regime: str = "normal",
) -> dict[str, float]:
    """Stress/crisis regimes up-weight tail, drawdown, correlation, liquidity."""
    base = static_weights(settings)
    scale = float(settings.regime_scales.get(str(regime).lower(), 1.0))
    stress_dims = {"tail", "drawdown", "correlation", "liquidity", "market"}
    tilted = {}
    for d in DIMENSIONS:
        factor = scale if d in stress_dims else 1.0 / max(scale, 1e-6)
        tilted[d] = base[d] * factor
    return _normalize_weights(tilted)


def dynamic_weights(
    settings: EnsembleSettings,
    *,
    dimension_scores: dict[str, float] | None = None,
    disagreement: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Up-weight elevated dimensions; down-weight high-disagreement model/tail soft signals."""
    base = risk_budget_weights(settings, dimension_scores=dimension_scores)
    disc = float((disagreement or {}).get("overall_disagreement", 0.0) or 0.0)
    if disc > settings.disagreement.high_disagreement:
        # Preserve hard dimensions; soften model weight under disagreement
        base["model"] = base["model"] * 0.5
        base["drawdown"] = base["drawdown"] * 1.15
        base["tail"] = base["tail"] * 1.10
    return _normalize_weights(base)


def calibration_weights(
    settings: EnsembleSettings,
    *,
    calibration_stats: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Down-weight poorly calibrated soft model dimensions; keep drawdown/liquidity identity."""
    base = static_weights(settings)
    stats = calibration_stats or {}
    var_bias = abs(float(stats.get("var_exceedance_bias", 0.0) or 0.0))
    vol_bias = abs(float(stats.get("vol_calibration_error", 0.0) or 0.0))
    if var_bias > settings.calibration.tolerance_band:
        base["tail"] = base["tail"] * (1.0 + min(var_bias, 1.0))
        base["model"] = base["model"] * 0.7
    if vol_bias > settings.calibration.tolerance_band:
        base["market"] = base["market"] * (1.0 + min(vol_bias, 1.0))
    return _normalize_weights(base)


def stress_weights(settings: EnsembleSettings) -> dict[str, float]:
    """Stress-test weighting: emphasize tail, drawdown, correlation, liquidity."""
    stressed = {
        "market": 0.12,
        "tail": 0.28,
        "liquidity": 0.15,
        "concentration": 0.08,
        "correlation": 0.14,
        "drawdown": 0.18,
        "model": 0.03,
        "operational": 0.02,
    }
    # Merge lightly with static so config still influences
    base = static_weights(settings)
    blended = {d: 0.35 * base[d] + 0.65 * stressed[d] for d in DIMENSIONS}
    return _normalize_weights(blended)


def resolve_weights(
    settings: EnsembleSettings,
    *,
    scheme: WeightingScheme | None = None,
    dimension_scores: dict[str, float] | None = None,
    disagreement: dict[str, Any] | None = None,
    calibration_stats: dict[str, Any] | None = None,
    regime: str = "normal",
    user_overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    scheme_name: str = scheme or settings.weighting_scheme
    if scheme_name == "user_defined":
        return user_defined_weights(settings, user_overrides)
    if scheme_name == "risk_budget":
        return risk_budget_weights(settings, dimension_scores=dimension_scores)
    if scheme_name == "regime":
        return regime_weights(settings, regime=regime)
    if scheme_name == "dynamic":
        return dynamic_weights(
            settings, dimension_scores=dimension_scores, disagreement=disagreement
        )
    if scheme_name == "calibration":
        return calibration_weights(settings, calibration_stats=calibration_stats)
    if scheme_name == "stress":
        return stress_weights(settings)
    return static_weights(settings)


class WeightResolver:
    def __init__(self, settings: EnsembleSettings) -> None:
        self.settings = settings

    def resolve(self, **kwargs: Any) -> dict[str, float]:
        return resolve_weights(self.settings, **kwargs)

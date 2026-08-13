"""Regime confidence and uncertainty estimates."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math.statistics.entropy import entropy
from iqrp.app.regimes.ensemble.disagreement import consensus_score, hard_agreement


def posterior_confidence(proba: np.ndarray) -> np.ndarray:
    """Per-step confidence = max posterior probability."""
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim == 1:
        return np.array([float(np.max(p))], dtype=np.float64)
    return p.max(axis=1)


def credible_mass_interval(
    proba: np.ndarray,
    *,
    level: float = 0.95,
) -> tuple[float, float]:
    """Highest-probability mass interval over regimes for a single distribution."""
    p = np.asarray(proba, dtype=np.float64).reshape(-1)
    p = p / max(float(p.sum()), 1e-300)
    order = np.argsort(p)[::-1]
    cum = 0.0
    included: list[int] = []
    for i in order:
        included.append(int(i))
        cum += float(p[i])
        if cum >= level:
            break
    masses = [float(p[i]) for i in included]
    return (min(masses), max(masses)) if masses else (0.0, 0.0)


def expected_persistence(transition: np.ndarray, regime: int) -> float:
    """Expected duration in regime ``i``: 1 / (1 - P_ii)."""
    tm = np.asarray(transition, dtype=np.float64)
    if tm.ndim != 2 or not (0 <= regime < tm.shape[0]):
        return 1.0
    pii = float(np.clip(tm[regime, regime], 0.0, 1.0 - 1e-12))
    return float(1.0 / max(1.0 - pii, 1e-12))


def forecast_uncertainty(step_distributions: np.ndarray) -> np.ndarray:
    """Entropy of each forecast step — ``(H,)``."""
    p = np.asarray(step_distributions, dtype=np.float64)
    if p.ndim == 1:
        return np.array([float(entropy(p))], dtype=np.float64)
    return np.asarray([float(entropy(row)) for row in p], dtype=np.float64)


def confidence_report(
    ensemble_proba: np.ndarray,
    member_proba: list[np.ndarray],
    *,
    transition: np.ndarray | None = None,
    level: float = 0.95,
) -> dict[str, Any]:
    p = np.asarray(ensemble_proba, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    conf = posterior_confidence(p)
    cons = consensus_score(member_proba) if member_proba else np.ones(p.shape[0])
    agree = hard_agreement(member_proba) if member_proba else np.ones(p.shape[0])
    current = p[-1]
    lo, hi = credible_mass_interval(current, level=level)
    regime = int(np.argmax(current))
    persist = expected_persistence(transition, regime) if transition is not None else 1.0
    return {
        "posterior": current,
        "confidence": float(conf[-1]),
        "confidence_timeline": conf,
        "consensus_confidence": float(cons[-1]),
        "model_agreement": float(agree[-1]),
        "credible_interval": (lo, hi),
        "expected_persistence": persist,
        "entropy": float(entropy(current)),
        "regime": regime,
    }

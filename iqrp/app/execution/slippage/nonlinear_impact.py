"""Nonlinear market-impact models (power-law / concave)."""

from __future__ import annotations

from typing import Any

import numpy as np


def nonlinear_impact(
    *,
    quantity: float,
    mid: float,
    adv: float,
    volatility: float = 0.02,
    impact_coeff: float = 0.1,
    exponent: float = 0.6,
    permanent_ratio: float = 0.4,
    floor_participation: float = 1e-12,
) -> dict[str, Any]:
    """Concave power-law impact: η σ mid |q/ADV|^α.

    ``exponent`` in (0, 1] recovers square-root at 0.5 and linear at 1.0.
    """
    qty = abs(float(quantity))
    mid_f = max(float(mid), 1e-12)
    adv_f = max(float(adv), 1e-12)
    vol = max(float(volatility), 0.0)
    alpha = float(exponent)
    if alpha <= 0.0:
        alpha = 0.5
    participation = max(qty / adv_f, float(floor_participation))
    temp = float(impact_coeff) * vol * mid_f * float(participation**alpha)
    perm = float(permanent_ratio) * temp
    return {
        "name": "nonlinear_impact",
        "temporary_impact": float(temp),
        "permanent_impact": float(perm),
        "slippage": float(temp),
        "slippage_bps": float(temp / mid_f * 1e4),
        "participation": float(participation),
        "exponent": alpha,
        "impact_coeff": float(impact_coeff),
    }


def impact_curve(
    participations: Any,
    *,
    mid: float = 100.0,
    volatility: float = 0.02,
    impact_coeff: float = 0.1,
    exponent: float = 0.6,
) -> np.ndarray:
    """Evaluate nonlinear impact (price units) over a participation grid."""
    parts = np.asarray(participations, dtype=np.float64).reshape(-1)
    parts = np.maximum(parts, 0.0)
    return (
        float(impact_coeff)
        * float(volatility)
        * float(mid)
        * np.power(parts, float(exponent))
    )


__all__ = ["impact_curve", "nonlinear_impact"]

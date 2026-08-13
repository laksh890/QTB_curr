"""Square-root market impact for alpha capacity analysis."""

from __future__ import annotations

from typing import Any

import numpy as np


def market_impact_bps(
    participation: Any,
    *,
    impact_coeff: float = 0.1,
    vol: float | np.ndarray = 0.01,
) -> np.ndarray:
    """Impact in bps = impact_coeff * vol * sqrt(participation) * 1e4."""
    p = np.maximum(np.asarray(participation, dtype=np.float64), 0.0)
    v = np.asarray(vol, dtype=np.float64)
    if v.ndim == 0:
        v = np.full(p.shape, float(v))
    else:
        v = np.broadcast_to(v, p.shape).astype(np.float64)
    return float(impact_coeff) * v * np.sqrt(p) * 1e4


def market_impact_cost(
    notional: Any,
    participation: Any,
    *,
    impact_coeff: float = 0.1,
    vol: float | np.ndarray = 0.01,
) -> dict[str, Any]:
    """Dollar market impact."""
    n = np.abs(np.asarray(notional, dtype=np.float64))
    bps = market_impact_bps(participation, impact_coeff=impact_coeff, vol=vol)
    # bps may be ndarray
    bps_arr = np.asarray(bps, dtype=np.float64)
    costs = n * (bps_arr / 1e4)
    return {
        "total": float(np.nansum(costs)),
        "per_unit": costs,
        "impact_bps": bps_arr,
    }

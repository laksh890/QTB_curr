"""Slippage models for alpha economics."""

from __future__ import annotations

from typing import Any

import numpy as np


def slippage_bps(
    participation: Any,
    *,
    base_bps: float = 1.0,
    participation_coeff: float = 10.0,
    vol: float | np.ndarray = 0.01,
) -> np.ndarray:
    """Temporary slippage in bps ≈ base + coeff * vol * sqrt(participation) * 1e4.

    ``vol`` is period volatility (fraction). Participation is traded notional / ADV.
    """
    p = np.maximum(np.asarray(participation, dtype=np.float64), 0.0)
    v = np.asarray(vol, dtype=np.float64)
    if v.ndim == 0:
        v = np.full(p.shape, float(v))
    else:
        v = np.broadcast_to(v, p.shape).astype(np.float64)
    frac = (float(base_bps) / 1e4) + float(participation_coeff) * v * np.sqrt(p)
    return frac * 1e4


def slippage_cost(
    notional: Any,
    participation: Any,
    *,
    base_bps: float = 1.0,
    participation_coeff: float = 10.0,
    vol: float | np.ndarray = 0.01,
) -> dict[str, Any]:
    """Dollar slippage from notional and participation."""
    n = np.abs(np.asarray(notional, dtype=np.float64))
    bps = slippage_bps(
        participation,
        base_bps=base_bps,
        participation_coeff=participation_coeff,
        vol=vol,
    )
    costs = n * (bps / 1e4)
    return {
        "total": float(np.nansum(costs)),
        "per_unit": costs,
        "slippage_bps": bps,
    }

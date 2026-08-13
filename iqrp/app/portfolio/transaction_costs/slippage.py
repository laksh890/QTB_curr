"""Slippage cost models."""

from __future__ import annotations

from typing import Any

import numpy as np


def slippage_cost(
    trades: Any,
    *,
    capital: float = 1.0,
    vols: Any | None = None,
    adv: Any | None = None,
    prices: Any | None = None,
    slippage_bps: float = 0.0,
    participation_coeff: float = 0.5,
) -> dict[str, Any]:
    """Temporary slippage ≈ slippage_bps + participation_coeff * vol * sqrt(participation).

    Trades are weight deltas; participation ≈ (|Δw| * capital / price) / ADV.
    """
    t = np.abs(np.asarray(trades, dtype=np.float64).reshape(-1))
    n = t.size
    cap = max(float(capital), 0.0)

    def _vec(x: Any | None, default: float) -> np.ndarray:
        if x is None:
            return np.full(n, default, dtype=np.float64)
        a = np.asarray(x, dtype=np.float64).reshape(-1)
        if a.size == 1:
            return np.full(n, float(a[0]))
        out = np.full(n, default, dtype=np.float64)
        out[: min(n, a.size)] = a[: min(n, a.size)]
        return out

    vol = np.maximum(_vec(vols, 0.0), 0.0)
    adv_v = np.maximum(_vec(adv, 1e18), 1e-12)
    px = np.maximum(_vec(prices, 1.0), 1e-12)
    notionals = t * cap
    shares = notionals / px
    participation = shares / adv_v
    frac = (float(slippage_bps) / 1e4) + float(participation_coeff) * vol * np.sqrt(
        np.maximum(participation, 0.0)
    )
    costs = notionals * np.maximum(frac, 0.0)
    return {
        "name": "slippage_cost",
        "total": float(np.sum(costs)),
        "per_asset": costs.tolist(),
        "participation": participation.tolist(),
        "slippage_fraction": frac.tolist(),
        "parameters": {
            "slippage_bps": float(slippage_bps),
            "participation_coeff": float(participation_coeff),
            "capital": cap,
        },
    }

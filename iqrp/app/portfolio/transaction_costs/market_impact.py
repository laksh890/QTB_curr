"""Square-root market impact model."""

from __future__ import annotations

from typing import Any

import numpy as np


def market_impact_cost(
    trades: Any,
    *,
    capital: float = 1.0,
    adv: Any | None = None,
    prices: Any | None = None,
    vols: Any | None = None,
    impact_coeff: float = 0.1,
) -> dict[str, Any]:
    """Temporary market impact = impact_coeff * vol * sqrt(participation) * notional.

    Participation = traded shares / ADV.
    """
    t = np.abs(np.asarray(trades, dtype=np.float64).reshape(-1))
    n = t.size
    cap = max(float(capital), 0.0)
    k = max(float(impact_coeff), 0.0)

    def _vec(x: Any | None, default: float) -> np.ndarray:
        if x is None:
            return np.full(n, default, dtype=np.float64)
        a = np.asarray(x, dtype=np.float64).reshape(-1)
        if a.size == 1:
            return np.full(n, float(a[0]))
        out = np.full(n, default, dtype=np.float64)
        out[: min(n, a.size)] = a[: min(n, a.size)]
        return out

    adv_v = np.maximum(_vec(adv, 1e18), 1e-12)
    px = np.maximum(_vec(prices, 1.0), 1e-12)
    vol = np.maximum(_vec(vols, 0.02), 0.0)
    notionals = t * cap
    shares = notionals / px
    participation = shares / adv_v
    impact_frac = k * vol * np.sqrt(np.maximum(participation, 0.0))
    costs = notionals * impact_frac
    return {
        "name": "market_impact_cost",
        "total": float(np.sum(costs)),
        "per_asset": costs.tolist(),
        "participation": participation.tolist(),
        "impact_fraction": impact_frac.tolist(),
        "parameters": {"impact_coeff": k, "capital": cap},
    }

"""Bid-ask spread transaction costs."""

from __future__ import annotations

from typing import Any

import numpy as np


def spread_cost(
    trades: Any,
    *,
    capital: float = 1.0,
    spreads: Any | None = None,
    half_spread: bool = True,
) -> dict[str, Any]:
    """Cost from crossing the spread: participation × spread × notional.

    Default uses half-spread (mid to touch) on traded notional.
    """
    t = np.abs(np.asarray(trades, dtype=np.float64).reshape(-1))
    cap = max(float(capital), 0.0)
    n = t.size
    if spreads is None:
        spr = np.zeros(n, dtype=np.float64)
    else:
        s = np.asarray(spreads, dtype=np.float64).reshape(-1)
        if s.size == 1:
            spr = np.full(n, float(s[0]))
        else:
            spr = np.zeros(n, dtype=np.float64)
            spr[: min(n, s.size)] = s[: min(n, s.size)]
    spr = np.maximum(spr, 0.0)
    factor = 0.5 if half_spread else 1.0
    notionals = t * cap
    costs = notionals * spr * factor
    total = float(np.sum(costs))
    return {
        "name": "spread_cost",
        "total": total,
        "per_asset": costs.tolist(),
        "spreads": spr.tolist(),
        "parameters": {"half_spread": half_spread, "capital": cap},
    }

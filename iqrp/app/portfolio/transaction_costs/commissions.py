"""Commission cost models."""

from __future__ import annotations

from typing import Any

import numpy as np


def commission_cost(
    trades: Any,
    *,
    capital: float = 1.0,
    prices: Any | None = None,
    commission_bps: float = 1.0,
    commission_per_share: float = 0.0,
    min_commission: float = 0.0,
) -> dict[str, Any]:
    """Commission on traded notional / shares.

    ``trades`` may be weight deltas (fraction of capital) or share quantities
    when ``prices`` is provided with ``trade_unit='shares'`` via abs(trades)*price.
    Default: treat trades as weight deltas → notional = |Δw| * capital.
    """
    t = np.asarray(trades, dtype=np.float64).reshape(-1)
    cap = max(float(capital), 0.0)
    bps = max(float(commission_bps), 0.0)
    per_share = max(float(commission_per_share), 0.0)
    floor = max(float(min_commission), 0.0)

    if prices is not None and per_share > 0.0:
        px = np.asarray(prices, dtype=np.float64).reshape(-1)
        n = max(t.size, px.size)
        tt = np.zeros(n)
        pp = np.ones(n)
        tt[: t.size] = t
        pp[: min(px.size, n)] = px[: min(px.size, n)]
        shares = np.abs(tt) * cap / np.maximum(pp, 1e-12)
        notionals = shares * np.maximum(pp, 0.0)
        raw = shares * per_share + notionals * (bps / 1e4)
    else:
        notionals = np.abs(t) * cap
        raw = notionals * (bps / 1e4)

    if floor > 0.0:
        raw = np.where(notionals > 1e-12, np.maximum(raw, floor), 0.0)

    total = float(np.sum(raw))
    return {
        "name": "commission_cost",
        "total": total,
        "per_asset": raw.tolist(),
        "notionals": notionals.tolist() if isinstance(notionals, np.ndarray) else list(notionals),
        "parameters": {
            "commission_bps": bps,
            "commission_per_share": per_share,
            "min_commission": floor,
            "capital": cap,
        },
    }

"""Transaction cost helpers for alpha backtests.

Prefers ``iqrp.app.portfolio.transaction_costs.total_transaction_cost`` when
available; falls back to a local bps × turnover model otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from iqrp.app.portfolio.transaction_costs import total_transaction_cost as _portfolio_tc
except Exception:
    _portfolio_tc = None


def _local_cost(
    weights_old: Any,
    weights_new: Any,
    *,
    capital: float = 1.0,
    cost_bps: float = 5.0,
) -> dict[str, Any]:
    a = np.asarray(weights_old, dtype=np.float64).reshape(-1)
    b = np.asarray(weights_new, dtype=np.float64).reshape(-1)
    n = max(a.size, b.size)
    aa = np.zeros(n)
    bb = np.zeros(n)
    aa[: a.size] = a
    bb[: b.size] = b
    turnover = 0.5 * float(np.sum(np.abs(bb - aa)))
    cap = max(float(capital), 0.0)
    total = cap * turnover * (float(cost_bps) / 1e4)
    return {
        "name": "local_transaction_cost",
        "total": float(total),
        "turnover": turnover,
        "total_bps": float(total / cap * 1e4) if cap > 0 else 0.0,
        "source": "local",
        "parameters": {"cost_bps": float(cost_bps), "capital": cap},
    }


def estimate_transaction_cost(
    weights_old: Any,
    weights_new: Any,
    *,
    capital: float = 1.0,
    cost_bps: float = 5.0,
    prefer_portfolio: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Estimate rebalance cost; portfolio engine if importable, else local."""
    if prefer_portfolio and _portfolio_tc is not None:
        try:
            out = dict(
                _portfolio_tc(
                    weights_old,
                    weights_new,
                    capital=capital,
                    commission_bps=float(kwargs.pop("commission_bps", cost_bps)),
                    **kwargs,
                )
            )
            out["source"] = "portfolio"
            return out
        except Exception:
            pass
    return _local_cost(weights_old, weights_new, capital=capital, cost_bps=cost_bps)

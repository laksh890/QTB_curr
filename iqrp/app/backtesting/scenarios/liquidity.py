"""Liquidity stress scenarios."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.backtesting.performance.returns import as_returns, total_return

__all__ = ["apply_liquidity_shock", "run_liquidity_scenario"]


def apply_liquidity_shock(
    returns: Any,
    *,
    liquidity_scores: Any | None = None,
    shock: float = 0.5,
    participation: float = 0.1,
    cost_coefficient: float = 0.01,
) -> dict[str, Any]:
    """Increase transaction-cost drag under degraded liquidity.

    ``shock`` in [0, 1] scales liquidity down; effective per-period drag is
    ``cost_coefficient * participation / liquidity``.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim == 1:
        n_assets = 1
        base = r.reshape(-1, 1)
    else:
        n_assets = r.shape[1]
        base = r

    if liquidity_scores is None:
        liq = np.ones(n_assets, dtype=np.float64)
    else:
        liq = np.asarray(liquidity_scores, dtype=np.float64).reshape(-1)
        if liq.size == 1:
            liq = np.full(n_assets, float(liq[0]))
        if liq.size != n_assets:
            raise ValueError("liquidity_scores length must match assets")

    shocked_liq = np.clip(liq * (1.0 - float(shock)), 1e-8, None)
    drag = float(cost_coefficient) * float(participation) / shocked_liq
    stressed = base - drag.reshape(1, -1)
    port = stressed.reshape(-1) if n_assets == 1 else stressed

    return {
        "name": "liquidity",
        "kind": "liquidity",
        "shock": float(shock),
        "liquidity": shocked_liq,
        "drag": drag,
        "returns": port,
        "total_drag": float(np.sum(drag) * base.shape[0]),
    }


def run_liquidity_scenario(
    returns: Any,
    *,
    shocks: list[float] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate a grid of liquidity shocks."""
    grid = [0.25, 0.5, 0.75] if shocks is None else list(shocks)
    results = []
    baseline = as_returns(returns if np.asarray(returns).ndim == 1 else np.mean(returns, axis=1))
    base_total = total_return(baseline)
    for s in grid:
        out = apply_liquidity_shock(returns, shock=float(s), **kwargs)
        rr = out["returns"]
        port = as_returns(rr if np.asarray(rr).ndim == 1 else np.mean(rr, axis=1))
        results.append(
            {
                "shock": float(s),
                "total_return": total_return(port),
                "return_vs_base": float(total_return(port) - base_total),
                "total_drag": out["total_drag"],
            }
        )
    return {"name": "liquidity_grid", "kind": "liquidity", "results": results}

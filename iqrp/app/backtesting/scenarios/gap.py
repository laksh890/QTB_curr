"""Market gap scenarios."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.backtesting.performance.drawdown import max_drawdown
from iqrp.app.backtesting.performance.returns import as_returns, total_return

__all__ = ["apply_gap_shock", "run_gap_scenario"]


def apply_gap_shock(
    returns: Any,
    *,
    gap: float = -0.05,
    index: int = 0,
    n_gaps: int = 1,
    spacing: int = 21,
) -> dict[str, Any]:
    """Inject one or more overnight/open gaps into a return path."""
    r = np.asarray(returns, dtype=np.float64).copy()
    if r.size == 0:
        return {"name": "gap", "kind": "gap", "returns": r, "gap_indices": []}

    if r.ndim == 1:
        multi = False
    else:
        multi = True

    indices: list[int] = []
    for k in range(max(int(n_gaps), 1)):
        idx = int(index) + k * max(int(spacing), 1)
        if multi:
            idx = int(np.clip(idx, 0, r.shape[0] - 1))
            r[idx, :] = r[idx, :] + float(gap)
        else:
            idx = int(np.clip(idx, 0, r.size - 1))
            r[idx] = r[idx] + float(gap)
        indices.append(idx)

    return {
        "name": "gap",
        "kind": "gap",
        "gap": float(gap),
        "gap_indices": indices,
        "returns": r,
        "total_return": total_return(as_returns(r if not multi else np.mean(r, axis=1))),
        "max_drawdown": max_drawdown(as_returns(r if not multi else np.mean(r, axis=1))),
    }


def run_gap_scenario(
    returns: Any,
    *,
    gaps: list[float] | None = None,
    index: int = 0,
) -> dict[str, Any]:
    """Evaluate a grid of gap magnitudes."""
    grid = [-0.02, -0.05, -0.10] if gaps is None else list(gaps)
    results = []
    for g in grid:
        out = apply_gap_shock(returns, gap=float(g), index=index)
        results.append(
            {
                "gap": float(g),
                "total_return": out["total_return"],
                "max_drawdown": out["max_drawdown"],
                "gap_indices": out["gap_indices"],
            }
        )
    return {"name": "gap_grid", "kind": "gap", "results": results}

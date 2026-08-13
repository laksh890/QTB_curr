"""Capital budget mapping from weights and total capital."""

from __future__ import annotations

from typing import Any

import numpy as np


def allocate_capital_budgets(
    names: list[str],
    weights: np.ndarray | list[float] | dict[str, float],
    *,
    capital: float = 1.0,
) -> dict[str, Any]:
    """Map normalized weights to absolute capital amounts."""
    n = len(names)
    cap = max(float(capital), 0.0)
    if isinstance(weights, dict):
        w = np.asarray([float(weights.get(nm, 0.0)) for nm in names], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size != n:
            w = np.full(n, 1.0 / n if n else 0.0)
    w = np.maximum(w, 0.0)
    s = float(np.sum(w))
    if s > 0:
        w = w / s
    amounts = {names[i]: float(cap * w[i]) for i in range(n)}
    return {
        "name": "capital_budgets",
        "capital": cap,
        "weights": {names[i]: float(w[i]) for i in range(n)},
        "amounts": amounts,
        "total_allocated": float(sum(amounts.values())),
    }


def clip_capital_to_limits(
    amounts: dict[str, float],
    *,
    max_position_capital: float | None = None,
    max_gross: float | None = None,
) -> dict[str, float]:
    """Clip absolute capital amounts to hard position / gross caps."""
    out = {k: max(float(v), 0.0) for k, v in amounts.items()}
    if max_position_capital is not None:
        cap = max(float(max_position_capital), 0.0)
        out = {k: min(v, cap) for k, v in out.items()}
    if max_gross is not None:
        gross_cap = max(float(max_gross), 0.0)
        total = float(sum(out.values()))
        if total > gross_cap and total > 0:
            scale = gross_cap / total
            out = {k: v * scale for k, v in out.items()}
    return out

"""Build per-strategy StrategyAllocation records from weights and settings."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.capital.config import CapitalSettings
from iqrp.app.risk.capital.types import StrategyAllocation


def build_strategy_allocations(
    names: list[str],
    weights: np.ndarray | list[float] | dict[str, float],
    *,
    capital: float = 1.0,
    risk_budgets: dict[str, float] | None = None,
    settings: CapitalSettings | None = None,
    capacity_scales: dict[str, float] | None = None,
    correlation_scales: dict[str, float] | None = None,
    drawdown_scales: dict[str, float] | None = None,
) -> dict[str, StrategyAllocation]:
    """Construct StrategyAllocation for each name with hard caps from settings."""
    cfg = settings or CapitalSettings.default()
    n = len(names)
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

    cap = max(float(capital), 0.0)
    rb = risk_budgets or {}
    if not rb and n:
        share = 1.0 / n
        rb = {nm: share for nm in names}

    out: dict[str, StrategyAllocation] = {}
    for i, nm in enumerate(names):
        reasons: list[str] = []
        c_scale = float((capacity_scales or {}).get(nm, 1.0))
        corr_scale = float((correlation_scales or {}).get(nm, 1.0))
        dd_scale = float((drawdown_scales or {}).get(nm, 1.0))
        if c_scale < 1.0:
            reasons.append(f"capacity_scale={c_scale:.4f}")
        if corr_scale < 1.0:
            reasons.append(f"correlation_scale={corr_scale:.4f}")
        if dd_scale < 1.0:
            reasons.append(f"drawdown_scale={dd_scale:.4f}")

        out[nm] = StrategyAllocation(
            name=nm,
            capital_budget=float(cap * w[i]),
            risk_budget=float(rb.get(nm, 0.0)),
            weight=float(w[i]),
            max_gross=float(cfg.max_gross_exposure),
            max_net=float(cfg.max_net_exposure),
            max_position=float(min(cfg.max_weight, cfg.max_concentration)),
            max_leverage=float(cfg.max_leverage),
            max_turnover=float(cfg.max_turnover),
            max_participation=float(cfg.max_participation),
            capacity_scale=c_scale,
            correlation_scale=corr_scale,
            drawdown_scale=dd_scale,
            reasons=reasons,
        )
    return out


def allocate_strategy(
    name: str,
    *,
    weight: float,
    capital: float = 1.0,
    risk_budget: float = 0.0,
    settings: CapitalSettings | None = None,
) -> StrategyAllocation:
    """Single-strategy allocation helper."""
    cfg = settings or CapitalSettings.default()
    w = float(np.clip(weight, cfg.min_weight, cfg.max_weight))
    return StrategyAllocation(
        name=name,
        capital_budget=float(max(capital, 0.0) * w),
        risk_budget=float(risk_budget),
        weight=w,
        max_gross=float(cfg.max_gross_exposure),
        max_net=float(cfg.max_net_exposure),
        max_position=float(min(cfg.max_weight, cfg.max_concentration)),
        max_leverage=float(cfg.max_leverage),
        max_turnover=float(cfg.max_turnover),
        max_participation=float(cfg.max_participation),
        reasons=["allocate_strategy"],
    )

"""Portfolio constraint handoff using existing TargetWeights / settings."""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from iqrp.app.backtesting.unified_pipeline.types import PortfolioHandoffResult, StageOutcome
from iqrp.app.portfolio import TargetWeights
from iqrp.app.portfolio.config import PortfolioSettings


def _pid() -> str:
    return f"port_{uuid.uuid4().hex[:16]}"


def apply_portfolio_constraints(
    *,
    instrument: str,
    proposed_weight: float,
    current_weights: dict[str, float],
    settings: PortfolioSettings | None = None,
    max_gross: float | None = None,
    max_position: float | None = None,
    long_only: bool | None = None,
) -> PortfolioHandoffResult:
    """Enforce existing portfolio-style constraints without optimization."""
    cfg = settings or PortfolioSettings.default()
    max_pos = float(max_position if max_position is not None else cfg.max_weight)
    max_g = float(max_gross if max_gross is not None else cfg.max_gross)
    lo = bool(long_only if long_only is not None else cfg.long_only)

    current = float(current_weights.get(instrument, 0.0))
    target = float(proposed_weight)
    reasons: list[str] = []
    outcome = StageOutcome.PORTFOLIO_APPROVED

    if lo and target < -1e-12:
        target = 0.0
        reasons.append("LONG_ONLY_FLATTENED_SHORT")
        outcome = StageOutcome.PORTFOLIO_REDUCED

    if abs(target) > max_pos + 1e-12:
        target = float(np.sign(target)) * max_pos
        reasons.append(f"MAX_POSITION_CAP:{max_pos}")
        outcome = StageOutcome.PORTFOLIO_REDUCED

    # Aggregate gross with other names
    others = {k: float(v) for k, v in current_weights.items() if k != instrument}
    trial = dict(others)
    trial[instrument] = target
    gross = sum(abs(v) for v in trial.values())
    if gross > max_g + 1e-12 and gross > 0:
        # Scale entire portfolio including proposed name to max_g
        scale = max_g / gross
        target = float(target) * scale
        reasons.append(f"MAX_GROSS_SCALE:{max_g}")
        outcome = StageOutcome.PORTFOLIO_REDUCED

    if abs(target) < 1e-15 and abs(proposed_weight) > 1e-12 and lo and proposed_weight < 0:
        outcome = StageOutcome.PORTFOLIO_REJECTED
        reasons.append("SHORT_REJECTED_LONG_ONLY")

    delta = target - current
    return PortfolioHandoffResult(
        portfolio_decision_id=_pid(),
        outcome=outcome,
        target_position_weight=target,
        current_position_weight=current,
        delta_weight=float(delta),
        reason="; ".join(reasons) if reasons else "PORTFOLIO_CONSTRAINTS_OK",
        constraint_reasons=reasons,
    )


def weights_to_target_object(weights: dict[str, float], *, long_only: bool = False) -> TargetWeights:
    names = sorted(weights)
    return TargetWeights(
        names=names,
        weights=[float(weights[n]) for n in names],
        method="unified_pipeline_handoff",
        source="alpha_candidate",
        long_only=long_only,
        meta={"disclaimer": "Not an optimized portfolio — constraint handoff only."},
    )


def weight_to_quantity(weight: float, *, equity: float, price: float) -> float:
    if price <= 0 or not np.isfinite(price):
        return 0.0
    return float(weight) * float(equity) / float(price)


__all__ = [
    "apply_portfolio_constraints",
    "weight_to_quantity",
    "weights_to_target_object",
]

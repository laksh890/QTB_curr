"""Implementation Shortfall (IS) optimal scheduling."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.execution.algorithms.base import (
    ChildSlice,
    ExecutionAlgorithm,
    MarketContext,
    approved_quantity,
    context_float,
    context_side,
    limit_hint,
    n_slices_for_urgency,
    redistribute_to_parent,
    schedule_offsets,
    urgency_from_context,
)
from iqrp.app.execution.types import Urgency


_URGENCY_RISK_AVERSION: dict[Urgency, float] = {
    Urgency.LOW: 0.35,
    Urgency.NORMAL: 1.0,
    Urgency.HIGH: 2.5,
    Urgency.CRITICAL: 6.0,
}


class ImplementationShortfallAlgorithm(ExecutionAlgorithm):
    """Almgren–Chriss-style IS schedule balancing impact vs timing risk.

    Optimizes slice sizes considering expected market impact, spread, volatility,
    urgency (risk aversion), opportunity cost, remaining quantity, and horizon.
    Never increases total quantity beyond the approved residual.
    """

    name = "implementation_shortfall"

    def __init__(
        self,
        *,
        n_slices: int = 8,
        horizon_seconds: float = 300.0,
        impact_coeff: float = 0.1,
        temporary_impact: float = 0.05,
        risk_aversion: float | None = None,
        default_urgency: Urgency | str = Urgency.NORMAL,
    ) -> None:
        super().__init__(default_urgency=default_urgency)
        self.n_slices = max(int(n_slices), 1)
        self.horizon_seconds = max(float(horizon_seconds), 0.0)
        self.impact_coeff = max(float(impact_coeff), 0.0)
        self.temporary_impact = max(float(temporary_impact), 0.0)
        self.risk_aversion = float(risk_aversion) if risk_aversion is not None else None

    def plan(
        self,
        parent_qty: float,
        market_context: MarketContext | None = None,
    ) -> list[ChildSlice]:
        ctx: dict[str, Any] = dict(market_context or {})
        approved = approved_quantity(parent_qty, ctx)
        if approved <= 0.0:
            return []

        urg = urgency_from_context(ctx, self.default_urgency)
        n = n_slices_for_urgency(int(ctx.get("n_slices", self.n_slices)), urg)
        horizon = context_float(ctx, "horizon_seconds", self.horizon_seconds)
        mid = context_float(ctx, "mid", context_float(ctx, "price", 100.0))
        spread = context_float(ctx, "spread", 0.0)
        vol = context_float(ctx, "volatility", 0.02)
        adv = context_float(ctx, "adv", 1e6)
        side = context_side(ctx)

        kappa = self.risk_aversion
        if kappa is None:
            kappa = float(ctx.get("risk_aversion", _URGENCY_RISK_AVERSION[urg]))
        kappa = max(kappa, 1e-8)

        # Permanent impact η, temporary impact ε (price units per unit participation)
        eta = float(ctx.get("impact_coeff", self.impact_coeff))
        eps = float(ctx.get("temporary_impact", self.temporary_impact))
        # Opportunity cost / timing risk ~ sigma^2
        sigma = max(vol, 1e-8)
        dt = horizon / max(n, 1) if horizon > 0 else 1.0

        # Almgren–Chriss continuous trajectory discretized:
        # x_j ≈ X * sinh(κ (T - t_j)) / sinh(κ T)
        # with κ ~ sqrt(lambda * sigma^2 / eta)
        # Higher urgency (kappa) → front-load.
        kappa_ac = np.sqrt(kappa * sigma * sigma / max(eta, 1e-8))
        T = max(horizon, 1e-8)
        remaining = np.zeros(n + 1, dtype=np.float64)
        remaining[0] = approved
        for j in range(n):
            t_j = horizon * (j + 1) / n
            tau = max(T - t_j, 0.0)
            # Inventory trajectory
            if kappa_ac * T < 1e-8:
                remaining[j + 1] = approved * (1.0 - (j + 1) / n)
            else:
                remaining[j + 1] = approved * float(np.sinh(kappa_ac * tau) / np.sinh(kappa_ac * T))
        trades = remaining[:-1] - remaining[1:]
        trades = np.maximum(trades, 0.0)

        # Spread cost bias: widen temporary cost → slightly slower when spreads wide
        spread_bps = (spread / max(mid, 1e-12)) * 1e4
        if spread_bps > 10.0 and urg in {Urgency.LOW, Urgency.NORMAL}:
            # Flatten toward equal (less aggressive) when spreads are wide
            equal = np.full(n, approved / n)
            blend = min((spread_bps - 10.0) / 40.0, 0.6)
            trades = (1.0 - blend) * trades + blend * equal

        # Opportunity cost remaining: if price has moved against us, accelerate
        arrival = context_float(ctx, "arrival_price", mid)
        decision = context_float(ctx, "decision_price", arrival)
        px_now = mid
        adverse = 0.0
        if side == "buy":
            adverse = max(px_now - arrival, 0.0) / max(arrival, 1e-12)
        else:
            adverse = max(arrival - px_now, 0.0) / max(arrival, 1e-12)
        if adverse > 0.0 and urg != Urgency.LOW:
            # Front-load more
            boost = np.linspace(1.0 + 2.0 * adverse, 1.0 - adverse, n)
            boost = np.maximum(boost, 0.1)
            trades = trades * boost

        qtys = redistribute_to_parent(trades.tolist(), approved)
        offsets = schedule_offsets(n, horizon)

        # Limit hints: more aggressive early when IS front-loads
        hints: list[float | None] = []
        for i in range(n):
            # Local urgency boost for early slices under high risk aversion
            local = urg
            if i == 0 and urg in {Urgency.HIGH, Urgency.CRITICAL}:
                local = Urgency.CRITICAL
            hints.append(limit_hint(mid, spread, side, local) if mid > 0 else None)

        # Expected impact metadata (informational)
        participation = approved / max(adv, 1e-12)
        exp_impact = eta * mid * np.sqrt(participation) + eps * mid * participation

        return self._finalize_slices(
            qtys,
            offsets,
            parent_qty=approved,
            market_context=ctx,
            urgency=urg,
            limit_prices=hints,
            metadata=[
                {
                    "algo": self.name,
                    "slice_index": i,
                    "kappa": float(kappa_ac),
                    "dt": float(dt),
                    "expected_impact_px": float(exp_impact),
                    "decision_price": float(decision),
                    "arrival_price": float(arrival),
                }
                for i in range(n)
            ],
        )


__all__ = ["ImplementationShortfallAlgorithm"]

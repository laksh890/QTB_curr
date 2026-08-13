"""Opportunistic execution — wait for favorable liquidity / price conditions."""

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


class OpportunisticAlgorithm(ExecutionAlgorithm):
    """Concentrate size into favorable windows; fall back to schedule as deadline nears.

    Uses spread compression, favorable price vs arrival, and liquidity spikes.
    Urgency shortens wait windows and raises limit aggression but never increases
    total quantity beyond the approved residual.
    """

    name = "opportunistic"

    def __init__(
        self,
        *,
        n_slices: int = 8,
        horizon_seconds: float = 360.0,
        patience: float = 0.6,
        default_urgency: Urgency | str = Urgency.NORMAL,
    ) -> None:
        super().__init__(default_urgency=default_urgency)
        self.n_slices = max(int(n_slices), 1)
        self.horizon_seconds = max(float(horizon_seconds), 0.0)
        # patience in [0,1]: higher → more back-loaded / selective
        self.patience = min(max(float(patience), 0.0), 1.0)

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
        side = context_side(ctx)
        arrival = context_float(ctx, "arrival_price", mid)

        patience = float(ctx.get("patience", self.patience))
        # Urgency reduces patience
        urg_patience = {
            Urgency.LOW: 1.15,
            Urgency.NORMAL: 1.0,
            Urgency.HIGH: 0.55,
            Urgency.CRITICAL: 0.15,
        }[urg]
        patience = min(max(patience * urg_patience, 0.0), 1.0)

        # Opportunity score series if provided; else synthesize from context
        opp = ctx.get("opportunity_scores")
        if opp is None:
            opp = ctx.get("liquidity_spikes")
        if opp is not None:
            scores = np.asarray(opp, dtype=np.float64).reshape(-1)
            scores = np.maximum(scores, 0.0)
            if scores.size != n:
                if scores.size == 0:
                    scores = np.ones(n)
                else:
                    x_old = np.linspace(0.0, 1.0, scores.size)
                    x_new = np.linspace(0.0, 1.0, n)
                    scores = np.interp(x_new, x_old, scores)
        else:
            # Favor mid-horizon windows when patient; front when urgent
            t = np.linspace(0.0, 1.0, n)
            # Favorable price signal
            if side == "buy":
                px_edge = max(arrival - mid, 0.0) / max(arrival, 1e-12)
            else:
                px_edge = max(mid - arrival, 0.0) / max(arrival, 1e-12)
            spread_ref = float(ctx.get("spread_ref", max(spread, 1e-8)))
            spread_edge = max(spread_ref - spread, 0.0) / max(spread_ref, 1e-8)
            liq = float(ctx.get("liquidity", 1.0))
            base = 0.4 + 0.4 * spread_edge + 0.4 * px_edge + 0.2 * min(liq, 2.0)
            # Time shape: patient → hump later; urgent → early
            time_shape = (1.0 - patience) * (1.0 - t) + patience * (0.3 + 0.7 * t)
            # Near deadline always force residual (last slices get floor)
            deadline_boost = np.exp(3.0 * (t - 1.0))  # rises sharply near end
            scores = base * time_shape + 0.35 * deadline_boost

        scores = np.maximum(scores, 1e-8)
        # Blend opportunity weights with uniform residual schedule so we always finish
        opp_w = scores / float(np.sum(scores))
        uni = np.full(n, 1.0 / n)
        # Higher patience → more opportunistic concentration; urgency → more uniform completion
        blend = 0.25 + 0.6 * patience
        if urg == Urgency.CRITICAL:
            blend = 0.1
        weights = blend * opp_w + (1.0 - blend) * uni
        weights = weights / float(np.sum(weights))

        qtys = redistribute_to_parent((weights * approved).tolist(), approved)
        offsets = schedule_offsets(n, horizon)

        # Passive limits when patient / favorable; aggressive near deadline slices
        hints: list[float | None] = []
        for i in range(n):
            local = urg
            frac = (i + 1) / n
            if frac > 0.85 and urg != Urgency.LOW:
                local = Urgency.HIGH if urg == Urgency.NORMAL else urg
            hints.append(limit_hint(mid, spread, side, local) if mid > 0 else None)

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
                    "opportunity_score": float(scores[i]),
                    "patience": patience,
                }
                for i in range(n)
            ],
        )


__all__ = ["OpportunisticAlgorithm"]

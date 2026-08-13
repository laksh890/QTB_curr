"""Aggressive market-order style parent execution (single or few slices)."""

from __future__ import annotations

from typing import Any

from iqrp.app.execution.algorithms.base import (
    ChildSlice,
    ExecutionAlgorithm,
    MarketContext,
    approved_quantity,
    context_float,
    context_side,
    n_slices_for_urgency,
    schedule_offsets,
    urgency_from_context,
)
from iqrp.app.execution.types import Urgency


class MarketAlgorithm(ExecutionAlgorithm):
    """Plan immediate market-style slices; higher urgency → fewer, larger slices."""

    name = "market"

    def __init__(
        self,
        *,
        n_slices: int = 1,
        horizon_seconds: float = 0.0,
        default_urgency: Urgency | str = Urgency.NORMAL,
    ) -> None:
        super().__init__(default_urgency=default_urgency)
        self.n_slices = max(int(n_slices), 1)
        self.horizon_seconds = max(float(horizon_seconds), 0.0)

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
        n = n_slices_for_urgency(self.n_slices, urg)
        # CRITICAL collapses to a single aggressive slice
        if urg == Urgency.CRITICAL:
            n = 1
        base = approved / n
        quantities = [base] * n
        offsets = schedule_offsets(n, self.horizon_seconds)
        mid = context_float(ctx, "mid", context_float(ctx, "price", 0.0))
        spread = context_float(ctx, "spread", 0.0)
        side = context_side(ctx)
        # Market: cross the spread fully (aggression = 1.0 equivalent)
        hints = []
        for _ in range(n):
            if mid > 0.0:
                # Always aggressive: pay/offer through half-spread at least
                if side == "buy":
                    hints.append(float(mid + 0.5 * spread))
                else:
                    hints.append(float(mid - 0.5 * spread))
            else:
                hints.append(None)
        return self._finalize_slices(
            quantities,
            offsets,
            parent_qty=approved,
            market_context=ctx,
            urgency=urg,
            limit_prices=hints,
            metadata=[{"algo": self.name, "style": "market"}] * n,
        )


__all__ = ["MarketAlgorithm"]

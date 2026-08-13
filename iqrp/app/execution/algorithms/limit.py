"""Passive / limit-order style parent execution."""

from __future__ import annotations

from typing import Any

from iqrp.app.execution.algorithms.base import (
    ChildSlice,
    ExecutionAlgorithm,
    MarketContext,
    approved_quantity,
    context_float,
    context_side,
    limit_hint,
    n_slices_for_urgency,
    schedule_offsets,
    urgency_from_context,
)
from iqrp.app.execution.types import Urgency


class LimitAlgorithm(ExecutionAlgorithm):
    """Plan limit slices with urgency-scaled price aggression."""

    name = "limit"

    def __init__(
        self,
        *,
        n_slices: int = 1,
        horizon_seconds: float = 60.0,
        limit_price: float | None = None,
        default_urgency: Urgency | str = Urgency.NORMAL,
    ) -> None:
        super().__init__(default_urgency=default_urgency)
        self.n_slices = max(int(n_slices), 1)
        self.horizon_seconds = max(float(horizon_seconds), 0.0)
        self.limit_price = float(limit_price) if limit_price is not None else None

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
        base = approved / n
        quantities = [base] * n
        offsets = schedule_offsets(n, self.horizon_seconds)
        mid = context_float(ctx, "mid", context_float(ctx, "price", 0.0))
        spread = context_float(ctx, "spread", 0.0)
        side = context_side(ctx)
        fixed = self.limit_price if self.limit_price is not None else ctx.get("limit_price")
        hints: list[float | None] = []
        for _ in range(n):
            if fixed is not None:
                hints.append(float(fixed))
            elif mid > 0.0:
                hints.append(limit_hint(mid, spread, side, urg))
            else:
                hints.append(None)
        return self._finalize_slices(
            quantities,
            offsets,
            parent_qty=approved,
            market_context=ctx,
            urgency=urg,
            limit_prices=hints,
            metadata=[{"algo": self.name, "style": "limit"}] * n,
        )


__all__ = ["LimitAlgorithm"]

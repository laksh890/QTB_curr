"""Liquidity-seeking execution — size into displayed / estimated liquidity."""

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


class LiquiditySeekingAlgorithm(ExecutionAlgorithm):
    """Allocate child sizes proportional to available liquidity / depth."""

    name = "liquidity_seeking"

    def __init__(
        self,
        *,
        n_slices: int = 8,
        horizon_seconds: float = 240.0,
        min_slice_pct: float = 0.02,
        max_slice_pct: float = 0.35,
        default_urgency: Urgency | str = Urgency.NORMAL,
    ) -> None:
        super().__init__(default_urgency=default_urgency)
        self.n_slices = max(int(n_slices), 1)
        self.horizon_seconds = max(float(horizon_seconds), 0.0)
        self.min_slice_pct = min(max(float(min_slice_pct), 0.0), 1.0)
        self.max_slice_pct = min(max(float(max_slice_pct), self.min_slice_pct), 1.0)

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
        adv = context_float(ctx, "adv", 1e6)

        depth = ctx.get("depth") or ctx.get("liquidity_profile") or ctx.get("order_book_sizes")
        if depth is not None:
            d = np.asarray(depth, dtype=np.float64).reshape(-1)
            d = np.maximum(d, 0.0)
            if d.size != n:
                # Repeat / truncate to n
                if d.size == 0:
                    d = np.ones(n, dtype=np.float64)
                else:
                    reps = int(np.ceil(n / d.size))
                    d = np.tile(d, reps)[:n]
            weights = d
        else:
            # Estimate liquidity from ADV and optional liquidity score
            score = float(ctx.get("liquidity", ctx.get("liquidity_score", 1.0)))
            score = max(score, 0.05)
            # Slight front preference under high urgency
            if urg in {Urgency.HIGH, Urgency.CRITICAL}:
                weights = np.linspace(1.5, 0.7, n) * score
            else:
                weights = np.ones(n, dtype=np.float64) * score

        weights = np.maximum(weights, 0.0)
        s = float(np.sum(weights))
        weights = weights / s if s > 0 else np.full(n, 1.0 / n)

        # Clip individual slice percentages
        max_q = approved * self.max_slice_pct
        min_q = approved * self.min_slice_pct
        raw = weights * approved
        raw = np.clip(raw, min_q if n * min_q <= approved else 0.0, max_q)

        # Soft participation vs ADV
        day_seconds = context_float(ctx, "trading_day_seconds", 23400.0)
        horizon_frac = min(horizon / max(day_seconds, 1.0), 1.0)
        expected_vol = max(adv, 1e-12) * max(horizon_frac, 1e-12)
        part_cap = float(ctx.get("participation_cap", 0.25))
        if part_cap > 0:
            per_cap = (expected_vol * part_cap) / n
            raw = np.minimum(raw, per_cap)

        qtys = redistribute_to_parent(raw.tolist(), approved)
        # Higher urgency → more aggressive limits
        hints = [limit_hint(mid, spread, side, urg) if mid > 0 else None for _ in range(n)]
        offsets = schedule_offsets(n, horizon)

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
                    "liquidity_weight": float(weights[i]),
                }
                for i in range(n)
            ],
        )


__all__ = ["LiquiditySeekingAlgorithm"]

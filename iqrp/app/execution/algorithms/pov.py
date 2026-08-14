"""Percentage of Volume (POV) / participation-of-volume execution."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.execution.algorithms.base import (
    ChildSlice,
    ExecutionAlgorithm,
    MarketContext,
    approved_quantity,
    context_float,
    n_slices_for_urgency,
    redistribute_to_parent,
    schedule_offsets,
    urgency_from_context,
)
from iqrp.app.execution.types import Urgency

# Urgency scales target participation (still hard-capped by max_participation).
_URGENCY_POV_MULT: dict[Urgency, float] = {
    Urgency.LOW: 0.7,
    Urgency.NORMAL: 1.0,
    Urgency.HIGH: 1.25,
    Urgency.CRITICAL: 1.5,
}


class POVAlgorithm(ExecutionAlgorithm):
    """Participate at a target rate of market volume with liquidity-aware throttling."""

    name = "pov"

    def __init__(
        self,
        *,
        target_participation: float = 0.10,
        max_participation: float = 0.20,
        n_slices: int = 10,
        horizon_seconds: float = 300.0,
        dynamic: bool = True,
        default_urgency: Urgency | str = Urgency.NORMAL,
    ) -> None:
        super().__init__(default_urgency=default_urgency)
        self.target_participation = max(float(target_participation), 0.0)
        self.max_participation = max(float(max_participation), 0.0)
        self.n_slices = max(int(n_slices), 1)
        self.horizon_seconds = max(float(horizon_seconds), 0.0)
        self.dynamic = bool(dynamic)

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
        adv = context_float(ctx, "adv", context_float(ctx, "average_daily_volume", 1e6))
        day_seconds = context_float(ctx, "trading_day_seconds", 23400.0)
        horizon_frac = min(horizon / max(day_seconds, 1.0), 1.0)
        expected_market_vol = max(adv, 1e-12) * max(horizon_frac, 1e-12)

        target = float(ctx.get("target_participation", self.target_participation))
        max_part = float(ctx.get("max_participation", self.max_participation))
        target *= _URGENCY_POV_MULT[urg]
        target = min(target, max_part) if max_part > 0 else target
        target = max(target, 0.0)

        # Expected child quantities from POV rate
        expected_trade = min(approved, expected_market_vol * target)

        # Volume curve / live pace for dynamic participation
        curve = ctx.get("volume_curve") or ctx.get("expected_volume_path")
        if curve is not None:
            arr = np.asarray(curve, dtype=np.float64).reshape(-1)
            arr = np.maximum(arr, 0.0)
            if arr.size != n:
                x_old = np.linspace(0.0, 1.0, max(arr.size, 1))
                x_new = np.linspace(0.0, 1.0, n)
                cdf = np.cumsum(arr)
                total = float(cdf[-1]) if arr.size else 0.0
                if total > 0:
                    cdf = cdf / total
                    cdf_new = np.interp(x_new, x_old, cdf)
                    weights = np.diff(np.concatenate([[0.0], cdf_new]))
                else:
                    weights = np.ones(n, dtype=np.float64)
            else:
                weights = arr
            weights = np.maximum(weights, 0.0)
            s = float(np.sum(weights))
            weights = weights / s if s > 0 else np.full(n, 1.0 / n)
        else:
            weights = np.full(n, 1.0 / n, dtype=np.float64)

        raw = (weights * expected_trade).tolist()

        # Dynamic throttling based on liquidity / spread / fill rate
        if self.dynamic:
            liquidity = float(ctx.get("liquidity", ctx.get("depth_score", 1.0)))
            liquidity = min(max(liquidity, 0.05), 2.0)
            spread = context_float(ctx, "spread", 0.0)
            mid = context_float(ctx, "mid", context_float(ctx, "price", 1.0))
            spread_bps = (spread / max(mid, 1e-12)) * 1e4 if mid > 0 else 0.0
            # Wide spreads → throttle
            spread_scale = 1.0 / (1.0 + max(spread_bps - 5.0, 0.0) / 20.0)
            fill_rate = float(ctx.get("fill_rate", 1.0))
            fill_rate = min(max(fill_rate, 0.05), 1.5)
            throttle = liquidity * spread_scale * fill_rate
            raw = [q * throttle for q in raw]
            # Still never exceed parent
            if float(sum(raw)) > approved:
                raw = redistribute_to_parent(raw, approved)

        # Per-slice hard participation vs expected market volume in that bucket
        per_bucket_vol = expected_market_vol * weights
        hard_cap = max_part if max_part > 0 else 1.0
        capped = [min(raw[i], float(per_bucket_vol[i]) * hard_cap) for i in range(n)]
        # Residual: if we under-participated due to throttle but urgency is high,
        # push residual into later buckets without exceeding parent or max POV
        shortfall = min(approved, expected_trade) - float(sum(capped))
        if shortfall > 1e-12 and urg in {Urgency.HIGH, Urgency.CRITICAL}:
            for i in range(n - 1, -1, -1):
                if shortfall <= 1e-12:
                    break
                room = max(float(per_bucket_vol[i]) * hard_cap - capped[i], 0.0)
                # Also respect remaining approved residual
                room = min(room, approved - float(sum(capped)))
                take = min(room, shortfall)
                capped[i] += take
                shortfall -= take

        capped = redistribute_to_parent(capped, min(float(sum(capped)), approved))
        # Prefer completing the parent when expected POV capacity allows
        if float(sum(capped)) < approved - 1e-9:
            capacity = float(np.sum(per_bucket_vol)) * hard_cap
            target_fill = min(approved, capacity)
            if target_fill > float(sum(capped)):
                capped = redistribute_to_parent(
                    [c + 1e-12 for c in capped],  # keep structure
                    target_fill,
                )
                # Re-apply per-bucket caps
                capped = [min(capped[i], float(per_bucket_vol[i]) * hard_cap) for i in range(n)]
                capped = redistribute_to_parent(capped, min(float(sum(capped)), approved))

        offsets = schedule_offsets(n, horizon)
        return self._finalize_slices(
            capped,
            offsets,
            parent_qty=approved,
            market_context=ctx,
            urgency=urg,
            metadata=[
                {
                    "algo": self.name,
                    "slice_index": i,
                    "target_participation": target,
                    "max_participation": hard_cap,
                }
                for i in range(n)
            ],
        )


__all__ = ["POVAlgorithm"]

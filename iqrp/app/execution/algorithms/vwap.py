"""Volume-Weighted Average Price (VWAP) execution algorithm."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.execution.algorithms.base import (
    ChildSlice,
    ExecutionAlgorithm,
    MarketContext,
    apply_participation_cap,
    approved_quantity,
    context_float,
    n_slices_for_urgency,
    redistribute_to_parent,
    schedule_offsets,
    urgency_from_context,
)
from iqrp.app.execution.types import Urgency


def normalize_volume_curve(curve: Any, n: int | None = None) -> np.ndarray:
    """Normalize a historical / intraday volume curve to probabilities."""
    arr = np.asarray(curve, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        m = max(int(n or 1), 1)
        return np.full(m, 1.0 / m, dtype=np.float64)
    arr = np.maximum(arr, 0.0)
    if n is not None and n > 0 and arr.size != n:
        # Resample curve to n buckets via cumulative interpolation
        x_old = np.linspace(0.0, 1.0, arr.size)
        x_new = np.linspace(0.0, 1.0, int(n))
        cdf = np.cumsum(arr)
        total = float(cdf[-1])
        if total <= 0.0:
            return np.full(int(n), 1.0 / int(n), dtype=np.float64)
        cdf = cdf / total
        # Invert-ish: allocate mass per new bucket from interpolated CDF
        cdf_new = np.interp(x_new, x_old, cdf)
        weights = np.diff(np.concatenate([[0.0], cdf_new]))
        weights = np.maximum(weights, 0.0)
        s = float(np.sum(weights))
        return weights / s if s > 0 else np.full(int(n), 1.0 / int(n), dtype=np.float64)
    s = float(np.sum(arr))
    if s <= 0.0:
        return np.full(arr.size, 1.0 / arr.size, dtype=np.float64)
    return arr / s


class VWAPAlgorithm(ExecutionAlgorithm):
    """Slice parent quantity according to a volume curve with participation limits."""

    name = "vwap"

    def __init__(
        self,
        *,
        n_slices: int | None = None,
        horizon_seconds: float = 300.0,
        participation_cap: float | None = 0.15,
        volume_curve: Any | None = None,
        adaptive: bool = True,
        default_urgency: Urgency | str = Urgency.NORMAL,
    ) -> None:
        super().__init__(default_urgency=default_urgency)
        self.n_slices = int(n_slices) if n_slices is not None else None
        self.horizon_seconds = max(float(horizon_seconds), 0.0)
        self.participation_cap = float(participation_cap) if participation_cap is not None else None
        self.volume_curve = volume_curve
        self.adaptive = bool(adaptive)

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
        curve = ctx.get("volume_curve", self.volume_curve)
        if curve is None:
            curve = ctx.get("intraday_volume_profile")
        if curve is None:
            # Flat profile fallback (= TWAP weights)
            n_fallback = self.n_slices or int(ctx.get("n_slices", 5))
            curve = np.ones(max(n_fallback, 1), dtype=np.float64)

        target_n = self.n_slices or int(ctx.get("n_slices") or 0) or None
        weights = normalize_volume_curve(curve, target_n)
        n = int(weights.size)
        n = n_slices_for_urgency(n, urg)
        if n != weights.size:
            weights = normalize_volume_curve(weights, n)

        # Adaptive volume estimation: blend historical curve with live volume pace
        if self.adaptive:
            live = ctx.get("live_volume_pace") or ctx.get("realized_volume_curve")
            if live is not None:
                live_w = normalize_volume_curve(live, n)
                alpha = float(ctx.get("adaptive_blend", 0.35))
                alpha = min(max(alpha, 0.0), 1.0)
                weights = (1.0 - alpha) * weights + alpha * live_w
                weights = weights / max(float(np.sum(weights)), 1e-12)

        raw = (weights * approved).tolist()
        adv = context_float(ctx, "adv", context_float(ctx, "average_daily_volume", 1e18))
        horizon = context_float(ctx, "horizon_seconds", self.horizon_seconds)
        day_seconds = context_float(ctx, "trading_day_seconds", 23400.0)
        horizon_frac = min(horizon / max(day_seconds, 1.0), 1.0)
        cap = (
            self.participation_cap
            if self.participation_cap is not None
            else ctx.get("participation_cap")
        )
        capped = apply_participation_cap(
            raw,
            adv=adv,
            participation_cap=float(cap) if cap is not None else None,
            horizon_fraction=horizon_frac,
        )
        # Residual: re-allocate to volume-heavy buckets first without exceeding parent
        shortfall = approved - float(sum(capped))
        if shortfall > 1e-12 and (cap is None or float(cap) <= 0.0):
            capped = redistribute_to_parent(capped, approved)
        elif shortfall > 1e-12:
            order = np.argsort(-weights)
            expected_vol = max(adv, 1e-12) * max(horizon_frac, 1e-12)
            per_slice_cap = (expected_vol * float(cap)) / max(n, 1)
            for idx in order:
                if shortfall <= 1e-12:
                    break
                room = max(per_slice_cap - capped[int(idx)], 0.0)
                take = min(room, shortfall)
                capped[int(idx)] += take
                shortfall -= take
            capped = redistribute_to_parent(capped, min(float(sum(capped)), approved))
        else:
            capped = redistribute_to_parent(capped, approved)

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
                    "volume_weight": float(weights[i]),
                }
                for i in range(n)
            ],
        )


__all__ = ["VWAPAlgorithm", "normalize_volume_curve"]

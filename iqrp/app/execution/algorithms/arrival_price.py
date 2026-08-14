"""Arrival-price benchmark tracking helpers and execution algorithm."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


def arrival_slippage_bps(
    *,
    side: str,
    fill_price: float,
    arrival_price: float,
) -> float:
    """Signed slippage vs arrival in bps (positive = adverse for the side)."""
    arr = max(float(arrival_price), 1e-12)
    fill = float(fill_price)
    s = str(side).strip().lower()
    if s in {"sell", "short", "s"}:
        return float((arr - fill) / arr * 1e4)
    return float((fill - arr) / arr * 1e4)


def decision_slippage_bps(
    *,
    side: str,
    fill_price: float,
    decision_price: float,
) -> float:
    """Signed slippage vs decision price in bps."""
    return arrival_slippage_bps(side=side, fill_price=fill_price, arrival_price=decision_price)


def benchmark_slippage_bps(
    *,
    side: str,
    fill_price: float,
    benchmark_price: float,
) -> float:
    """Generic benchmark slippage in bps."""
    return arrival_slippage_bps(side=side, fill_price=fill_price, arrival_price=benchmark_price)


def vw_average_price(fills: Sequence[Mapping[str, Any]]) -> float:
    """Volume-weighted average fill price from fill dicts with price/quantity."""
    notionals = 0.0
    qty = 0.0
    for f in fills:
        q = abs(float(f.get("quantity", f.get("qty", 0.0))))
        p = float(f.get("price", f.get("fill_price", 0.0)))
        notionals += q * p
        qty += q
    if qty <= 0.0:
        return 0.0
    return notionals / qty


def track_arrival_performance(
    fills: Sequence[Mapping[str, Any]],
    *,
    side: str,
    arrival_price: float,
    decision_price: float | None = None,
    benchmark_price: float | None = None,
) -> dict[str, Any]:
    """Aggregate arrival / decision / benchmark tracking statistics."""
    vwap = vw_average_price(fills)
    total_qty = float(sum(abs(float(f.get("quantity", f.get("qty", 0.0)))) for f in fills))
    decision = float(decision_price) if decision_price is not None else float(arrival_price)
    bench = float(benchmark_price) if benchmark_price is not None else float(arrival_price)
    return {
        "vwap": float(vwap),
        "filled_quantity": total_qty,
        "arrival_price": float(arrival_price),
        "decision_price": decision,
        "benchmark_price": bench,
        "arrival_slippage_bps": (
            arrival_slippage_bps(side=side, fill_price=vwap, arrival_price=arrival_price)
            if vwap > 0
            else 0.0
        ),
        "decision_slippage_bps": (
            decision_slippage_bps(side=side, fill_price=vwap, decision_price=decision)
            if vwap > 0
            else 0.0
        ),
        "benchmark_slippage_bps": (
            benchmark_slippage_bps(side=side, fill_price=vwap, benchmark_price=bench)
            if vwap > 0
            else 0.0
        ),
    }


class ArrivalPriceAlgorithm(ExecutionAlgorithm):
    """Execute while tightly tracking arrival price; accelerate on adverse drift."""

    name = "arrival_price"

    def __init__(
        self,
        *,
        n_slices: int = 6,
        horizon_seconds: float = 180.0,
        drift_tolerance_bps: float = 5.0,
        default_urgency: Urgency | str = Urgency.NORMAL,
    ) -> None:
        super().__init__(default_urgency=default_urgency)
        self.n_slices = max(int(n_slices), 1)
        self.horizon_seconds = max(float(horizon_seconds), 0.0)
        self.drift_tolerance_bps = max(float(drift_tolerance_bps), 0.0)

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
        decision = context_float(ctx, "decision_price", arrival)

        # Measure adverse drift vs arrival
        if side == "buy":
            drift_bps = (mid - arrival) / max(arrival, 1e-12) * 1e4
        else:
            drift_bps = (arrival - mid) / max(arrival, 1e-12) * 1e4

        tol = float(ctx.get("drift_tolerance_bps", self.drift_tolerance_bps))
        # Base schedule: mild front-load; accelerate if adverse drift exceeds tolerance
        if drift_bps > tol or urg in {Urgency.HIGH, Urgency.CRITICAL}:
            # Geometric front-load
            decay = 1.4 if urg != Urgency.CRITICAL else 1.8
            weights = np.array([decay ** (n - i) for i in range(n)], dtype=np.float64)
        elif drift_bps < -tol:
            # Favorable drift — slow down (back-load) unless CRITICAL
            if urg == Urgency.CRITICAL:
                weights = np.ones(n, dtype=np.float64)
            else:
                weights = np.array([1.15**i for i in range(n)], dtype=np.float64)
        else:
            weights = np.ones(n, dtype=np.float64)

        weights = weights / max(float(np.sum(weights)), 1e-12)
        qtys = redistribute_to_parent((weights * approved).tolist(), approved)
        offsets = schedule_offsets(n, horizon)
        local_urg = urg
        if drift_bps > tol and urg != Urgency.CRITICAL:
            local_urg = Urgency.HIGH if urg == Urgency.NORMAL else urg
        hints = [limit_hint(mid, spread, side, local_urg) if mid > 0 else None for _ in range(n)]

        return self._finalize_slices(
            qtys,
            offsets,
            parent_qty=approved,
            market_context=ctx,
            urgency=local_urg,
            limit_prices=hints,
            metadata=[
                {
                    "algo": self.name,
                    "slice_index": i,
                    "arrival_price": float(arrival),
                    "decision_price": float(decision),
                    "drift_bps": float(drift_bps),
                }
                for i in range(n)
            ],
        )


__all__ = [
    "ArrivalPriceAlgorithm",
    "arrival_slippage_bps",
    "benchmark_slippage_bps",
    "decision_slippage_bps",
    "track_arrival_performance",
    "vw_average_price",
]

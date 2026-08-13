"""Multi-venue quantity allocation for smart order routing.

Splits order quantity across eligible venues based on scores and
fillable liquidity. Supports single-venue (winner-take-all) and
multi-venue liquidity splitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from iqrp.app.execution.smart_routing.liquidity import LiquiditySnapshot
from iqrp.app.execution.smart_routing.scoring import VenueScore


AllocationMode = Literal["single", "multi"]


@dataclass(slots=True)
class VenueAllocation:
    """Quantity allocated to one venue."""

    venue_id: str
    quantity: float
    score: float
    fillable_qty: float
    weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue_id": self.venue_id,
            "quantity": float(self.quantity),
            "score": float(self.score),
            "fillable_qty": float(self.fillable_qty),
            "weight": float(self.weight),
        }


@dataclass(slots=True)
class AllocationPlan:
    """Full allocation plan across venues."""

    mode: AllocationMode
    allocations: list[VenueAllocation] = field(default_factory=list)
    residual_qty: float = 0.0
    total_allocated: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "allocations": [a.to_dict() for a in self.allocations],
            "residual_qty": float(self.residual_qty),
            "total_allocated": float(self.total_allocated),
        }


def _lot_round(qty: float, lot_size: float) -> float:
    lot = max(float(lot_size), 1e-12)
    if qty < lot:
        return 0.0
    n = int(qty / lot)
    return float(n) * lot


def allocate_quantity(
    quantity: float,
    scores: list[VenueScore],
    liquidity: dict[str, LiquiditySnapshot],
    *,
    mode: AllocationMode = "multi",
    lot_sizes: dict[str, float] | None = None,
    min_qty: dict[str, float] | None = None,
    max_venues: int | None = None,
) -> AllocationPlan:
    """Allocate ``quantity`` across ranked venues.

    - ``single``: send entire order (capped by fillable) to the top venue.
    - ``multi``: split by score-weighted fillable capacity until filled or exhausted.
    """
    qty = max(float(quantity), 0.0)
    if qty <= 0.0 or not scores:
        return AllocationPlan(mode=mode, residual_qty=qty)

    ranked = sorted(scores, key=lambda s: s.score, reverse=True)
    if max_venues is not None and max_venues > 0:
        ranked = ranked[: int(max_venues)]

    lots = lot_sizes or {}
    mins = min_qty or {}

    if mode == "single":
        top = ranked[0]
        snap = liquidity.get(top.venue_id)
        fillable = float(snap.fillable_qty) if snap is not None else qty
        lot = float(lots.get(top.venue_id, 1.0))
        mn = float(mins.get(top.venue_id, 0.0))
        alloc_qty = _lot_round(min(qty, fillable), lot)
        if alloc_qty < mn:
            alloc_qty = 0.0
        allocs = []
        if alloc_qty > 0:
            allocs.append(
                VenueAllocation(
                    venue_id=top.venue_id,
                    quantity=alloc_qty,
                    score=top.score,
                    fillable_qty=fillable,
                    weight=1.0,
                )
            )
        residual = max(qty - alloc_qty, 0.0)
        return AllocationPlan(
            mode="single",
            allocations=allocs,
            residual_qty=residual,
            total_allocated=alloc_qty,
        )

    # Multi-venue: iterate by rank, allocate score-proportional share of remaining
    # capacity, capped by each venue's fillable quantity.
    positive = [s for s in ranked if s.score > 0]
    if not positive:
        positive = ranked

    remaining = qty
    allocs: list[VenueAllocation] = []
    score_sum = sum(max(s.score, 0.0) for s in positive) or 1.0

    # First pass: target weights from scores
    targets: dict[str, float] = {}
    for s in positive:
        targets[s.venue_id] = qty * (max(s.score, 0.0) / score_sum)

    for s in positive:
        if remaining <= 0:
            break
        snap = liquidity.get(s.venue_id)
        fillable = float(snap.fillable_qty) if snap is not None else remaining
        lot = float(lots.get(s.venue_id, 1.0))
        mn = float(mins.get(s.venue_id, 0.0))
        desired = min(targets.get(s.venue_id, remaining), fillable, remaining)
        alloc_qty = _lot_round(desired, lot)
        if alloc_qty < mn:
            continue
        if alloc_qty <= 0:
            continue
        allocs.append(
            VenueAllocation(
                venue_id=s.venue_id,
                quantity=alloc_qty,
                score=s.score,
                fillable_qty=fillable,
                weight=max(s.score, 0.0) / score_sum,
            )
        )
        remaining -= alloc_qty

    # Second pass: top up residual on venues with leftover capacity
    if remaining > 0:
        for s in positive:
            if remaining <= 0:
                break
            snap = liquidity.get(s.venue_id)
            fillable = float(snap.fillable_qty) if snap is not None else remaining
            already = sum(a.quantity for a in allocs if a.venue_id == s.venue_id)
            leftover = max(fillable - already, 0.0)
            if leftover <= 0:
                continue
            lot = float(lots.get(s.venue_id, 1.0))
            mn = float(mins.get(s.venue_id, 0.0))
            add = _lot_round(min(leftover, remaining), lot)
            if add < mn and already <= 0:
                continue
            if add <= 0:
                continue
            found = False
            for a in allocs:
                if a.venue_id == s.venue_id:
                    a.quantity = float(a.quantity) + add
                    found = True
                    break
            if not found:
                allocs.append(
                    VenueAllocation(
                        venue_id=s.venue_id,
                        quantity=add,
                        score=s.score,
                        fillable_qty=fillable,
                        weight=max(s.score, 0.0) / score_sum,
                    )
                )
            remaining -= add

    total = float(sum(a.quantity for a in allocs))
    return AllocationPlan(
        mode="multi",
        allocations=allocs,
        residual_qty=max(qty - total, 0.0),
        total_allocated=total,
    )


__all__ = [
    "AllocationMode",
    "VenueAllocation",
    "AllocationPlan",
    "allocate_quantity",
]

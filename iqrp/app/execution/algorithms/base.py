"""Execution algorithm abstractions, urgency, and child-slice planning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

from iqrp.app.execution.types import Urgency

# Relative slice-size multipliers. Larger urgency → fewer/larger slices when rebucketed,
# but total planned quantity is always clipped to the approved parent residual.
URGENCY_SLICE_FACTOR: dict[Urgency, float] = {
    Urgency.LOW: 0.75,
    Urgency.NORMAL: 1.0,
    Urgency.HIGH: 1.35,
    Urgency.CRITICAL: 1.75,
}

# Limit aggression as a fraction of half-spread toward (buy) / away-from (sell) the mid.
# Positive = more aggressive (cross more of the spread).
URGENCY_LIMIT_AGGRESSION: dict[Urgency, float] = {
    Urgency.LOW: -0.35,
    Urgency.NORMAL: 0.0,
    Urgency.HIGH: 0.55,
    Urgency.CRITICAL: 1.0,
}


def coerce_urgency(value: Urgency | str | None, default: Urgency = Urgency.NORMAL) -> Urgency:
    if value is None:
        return default
    if isinstance(value, Urgency):
        return value
    key = str(value).strip().upper()
    try:
        return Urgency[key]
    except KeyError:
        return Urgency(key) if key in {u.value for u in Urgency} else default


@dataclass(slots=True)
class ChildSlice:
    """A scheduled child order slice of a parent execution."""

    quantity: float
    not_before_offset: float = 0.0
    limit_price_hint: float | None = None
    urgency: Urgency = Urgency.NORMAL
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.quantity = float(self.quantity)
        self.not_before_offset = float(self.not_before_offset)
        if self.limit_price_hint is not None:
            self.limit_price_hint = float(self.limit_price_hint)
        self.urgency = coerce_urgency(self.urgency)

    @property
    def qty(self) -> float:
        """Alias for quantity (parent/child order terminology)."""
        return self.quantity


MarketContext = Mapping[str, Any]


def context_float(ctx: MarketContext, key: str, default: float) -> float:
    val = ctx.get(key, default)
    if val is None:
        return float(default)
    return float(val)


def context_side(ctx: MarketContext, default: str = "buy") -> str:
    side = str(ctx.get("side", default)).strip().lower()
    return "sell" if side in {"sell", "short", "s"} else "buy"


def approved_quantity(parent_qty: float, market_context: MarketContext | None = None) -> float:
    """Approved residual quantity — algorithms must never plan beyond this."""
    qty = abs(float(parent_qty))
    if market_context is not None:
        residual = market_context.get("residual")
        if residual is not None:
            qty = min(qty, abs(float(residual)))
        approved = market_context.get("approved_quantity")
        if approved is not None:
            qty = min(qty, abs(float(approved)))
        max_qty = market_context.get("max_quantity")
        if max_qty is not None:
            qty = min(qty, abs(float(max_qty)))
    return max(qty, 0.0)


def urgency_from_context(
    market_context: MarketContext | None,
    default: Urgency = Urgency.NORMAL,
) -> Urgency:
    if not market_context:
        return default
    return coerce_urgency(market_context.get("urgency", default), default)


def limit_hint(
    mid: float,
    spread: float,
    side: str,
    urgency: Urgency,
) -> float:
    """Urgency-scaled limit price hint around mid ± half-spread."""
    half = 0.5 * max(float(spread), 0.0)
    aggression = URGENCY_LIMIT_AGGRESSION[coerce_urgency(urgency)]
    # buy: higher aggression → higher limit (more willing to pay)
    # sell: higher aggression → lower limit (more willing to sell)
    if side == "buy":
        return float(mid + half * aggression)
    return float(mid - half * aggression)


def redistribute_to_parent(
    quantities: Sequence[float],
    parent_qty: float,
) -> list[float]:
    """Scale / residual-adjust slice quantities so sum == approved parent qty."""
    target = max(float(parent_qty), 0.0)
    raw = [max(float(q), 0.0) for q in quantities]
    if not raw:
        return []
    total = float(sum(raw))
    if total <= 0.0 or target <= 0.0:
        return [0.0] * len(raw)
    if abs(total - target) < 1e-12:
        out = list(raw)
    else:
        scale = target / total
        out = [q * scale for q in raw]
    # Put residual rounding error on the last non-zero-capable slice
    drift = target - float(sum(out))
    out[-1] = max(out[-1] + drift, 0.0)
    # Final hard clip: never exceed parent
    if float(sum(out)) > target + 1e-9:
        scale = target / float(sum(out))
        out = [q * scale for q in out]
        out[-1] = max(target - float(sum(out[:-1])), 0.0)
    return out


def apply_participation_cap(
    quantities: Sequence[float],
    *,
    adv: float,
    participation_cap: float | None,
    horizon_fraction: float = 1.0,
) -> list[float]:
    """Cap each slice by participation of ADV over the horizon fraction."""
    if participation_cap is None or participation_cap <= 0.0:
        return [max(float(q), 0.0) for q in quantities]
    adv_f = max(float(adv), 1e-12)
    hf = max(float(horizon_fraction), 1e-12)
    # Expected volume over horizon ≈ ADV * horizon_fraction (fraction of day)
    expected_vol = adv_f * hf
    n = max(len(quantities), 1)
    per_slice_cap = (expected_vol * float(participation_cap)) / n
    return [min(max(float(q), 0.0), per_slice_cap) for q in quantities]


def n_slices_for_urgency(base_slices: int, urgency: Urgency) -> int:
    """Higher urgency → fewer slices (more aggressive). Never below 1."""
    base = max(int(base_slices), 1)
    factor = URGENCY_SLICE_FACTOR[coerce_urgency(urgency)]
    # inverse: high factor → fewer slices
    adjusted = int(round(base / factor))
    return max(adjusted, 1)


class ExecutionAlgorithm(ABC):
    """Abstract base for parent → child slice planners."""

    name: str = "base"

    def __init__(self, *, default_urgency: Urgency | str = Urgency.NORMAL) -> None:
        self.default_urgency = coerce_urgency(default_urgency)

    @abstractmethod
    def plan(
        self,
        parent_qty: float,
        market_context: MarketContext | None = None,
    ) -> list[ChildSlice]:
        """Plan child slices for ``parent_qty`` given market context.

        Total planned quantity MUST NOT exceed the approved residual.
        """

    def _finalize_slices(
        self,
        quantities: Sequence[float],
        offsets: Sequence[float],
        *,
        parent_qty: float,
        market_context: MarketContext | None,
        urgency: Urgency | None = None,
        limit_prices: Sequence[float | None] | None = None,
        metadata: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[ChildSlice]:
        ctx: MutableMapping[str, Any] = dict(market_context or {})
        approved = approved_quantity(parent_qty, ctx)
        if approved <= 0.0:
            return []
        urg = coerce_urgency(urgency or urgency_from_context(ctx, self.default_urgency))
        qtys = redistribute_to_parent(quantities, approved)
        offs = list(offsets) if offsets else [0.0] * len(qtys)
        if len(offs) < len(qtys):
            offs = offs + [offs[-1] if offs else 0.0] * (len(qtys) - len(offs))
        limits = list(limit_prices) if limit_prices is not None else [None] * len(qtys)
        if len(limits) < len(qtys):
            limits = limits + [None] * (len(qtys) - len(limits))
        meta_list = list(metadata) if metadata is not None else [{}] * len(qtys)
        if len(meta_list) < len(qtys):
            meta_list = meta_list + [{}] * (len(qtys) - len(meta_list))

        mid = context_float(ctx, "mid", context_float(ctx, "price", 0.0))
        spread = context_float(ctx, "spread", 0.0)
        side = context_side(ctx)
        slices: list[ChildSlice] = []
        for i, q in enumerate(qtys):
            if q <= 0.0:
                continue
            hint = limits[i]
            if hint is None and mid > 0.0:
                hint = limit_hint(mid, spread, side, urg)
            slices.append(
                ChildSlice(
                    quantity=float(q),
                    not_before_offset=float(max(offs[i], 0.0)),
                    limit_price_hint=hint,
                    urgency=urg,
                    metadata=dict(meta_list[i]),
                )
            )
        # Safety: hard-cap residual if floating point drift crept in
        total = float(sum(s.quantity for s in slices))
        if total > approved + 1e-9 and slices:
            scale = approved / total
            for s in slices:
                s.quantity = float(s.quantity * scale)
            drift = approved - float(sum(s.quantity for s in slices))
            slices[-1].quantity = max(slices[-1].quantity + drift, 0.0)
        return [s for s in slices if s.quantity > 0.0]


def schedule_offsets(n: int, horizon_seconds: float, *, jitter: float = 0.0, rng: np.random.Generator | None = None) -> list[float]:
    """Evenly spaced offsets over ``horizon_seconds`` with optional relative jitter."""
    n = max(int(n), 1)
    horizon = max(float(horizon_seconds), 0.0)
    if n == 1:
        return [0.0]
    base = [horizon * i / n for i in range(n)]
    if jitter <= 0.0:
        return base
    gen = rng or np.random.default_rng()
    interval = horizon / n
    max_j = float(jitter) * interval
    out: list[float] = []
    for i, t in enumerate(base):
        if i == 0:
            out.append(0.0)
            continue
        delta = float(gen.uniform(-max_j, max_j))
        out.append(max(0.0, min(horizon, t + delta)))
    out.sort()
    return out

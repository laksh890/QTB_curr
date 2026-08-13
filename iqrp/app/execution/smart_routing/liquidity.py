"""Per-venue liquidity assessment for smart order routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from iqrp.app.execution.smart_routing.venue import Venue
from iqrp.app.execution.types import Side


@dataclass(slots=True)
class LiquiditySnapshot:
    """Liquidity metrics for an instrument at a venue."""

    venue_id: str
    instrument: str
    available_qty: float
    adv: float
    liquidity_score: float
    participation: float
    fillable_qty: float
    depth_ok: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue_id": self.venue_id,
            "instrument": self.instrument,
            "available_qty": float(self.available_qty),
            "adv": float(self.adv),
            "liquidity_score": float(self.liquidity_score),
            "participation": float(self.participation),
            "fillable_qty": float(self.fillable_qty),
            "depth_ok": self.depth_ok,
            "reasons": list(self.reasons),
        }


def assess_liquidity(
    venue: Venue,
    *,
    instrument: str,
    quantity: float,
    side: Side | str = Side.BUY,
    max_participation: float | None = None,
) -> LiquiditySnapshot:
    """Estimate how much of ``quantity`` the venue can reasonably absorb.

    Fillable quantity is limited by displayed available quantity, venue
    max participation vs ADV, and the venue liquidity score.
    """
    del side  # side reserved for asymmetric books; quotes already in state
    state = venue.venue_state
    state.ensure_quotes()
    reasons: list[str] = []

    available = max(float(state.available_qty), 0.0)
    adv = max(float(state.adv), 0.0)
    score = min(max(float(state.liquidity_score), 0.0), 1.0)
    qty = max(float(quantity), 0.0)

    part_cap = float(
        max_participation if max_participation is not None else venue.max_participation
    )
    part_cap = min(max(part_cap, 0.0), 1.0)

    if available <= 0.0 and adv <= 0.0:
        reasons.append("no_liquidity_data")
        # Fall back to score-scaled quantity so scoring can still rank venues
        fillable = qty * score
        participation = 1.0 if qty > 0 else 0.0
        return LiquiditySnapshot(
            venue_id=venue.venue_id,
            instrument=str(instrument),
            available_qty=available,
            adv=adv,
            liquidity_score=score,
            participation=participation,
            fillable_qty=fillable,
            depth_ok=score > 0.0 and fillable > 0.0,
            reasons=reasons,
        )

    # Cap by displayed depth when known
    fillable = qty
    if available > 0.0:
        fillable = min(fillable, available)
        if available < qty:
            reasons.append("insufficient_displayed_qty")

    # Cap by ADV participation
    if adv > 0.0:
        adv_cap = adv * part_cap
        if fillable > adv_cap:
            fillable = adv_cap
            reasons.append("participation_cap")
        participation = float(qty) / adv if adv > 0 else 1.0
    else:
        participation = 0.0

    # Liquidity score gates routability; only soft-scale when depth is unknown
    # or score is critically low. Never invent capacity above displayed/ADV caps.
    if score <= 0.0:
        fillable = 0.0
        reasons.append("zero_liquidity_score")
    elif available <= 0.0 and adv <= 0.0:
        fillable = max(0.0, fillable * score)
    elif score < 0.2:
        fillable = max(0.0, fillable * score)
        reasons.append("low_liquidity_score_scaled")

    if fillable <= 0.0:
        reasons.append("zero_fillable")

    depth_ok = fillable > 0.0 and score > 0.0
    return LiquiditySnapshot(
        venue_id=venue.venue_id,
        instrument=str(instrument),
        available_qty=available,
        adv=adv,
        liquidity_score=score,
        participation=float(participation),
        fillable_qty=float(fillable),
        depth_ok=depth_ok,
        reasons=reasons,
    )


def aggregate_fillable(snapshots: list[LiquiditySnapshot]) -> float:
    """Sum fillable quantity across venues."""
    return float(sum(max(s.fillable_qty, 0.0) for s in snapshots))


__all__ = [
    "LiquiditySnapshot",
    "assess_liquidity",
    "aggregate_fillable",
]

"""Weighted venue scoring for smart order routing.

Scores combine expected price, fees, spread, liquidity, impact,
fill probability, latency, and reliability into a single ranking.
Higher score is better.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from iqrp.app.execution.smart_routing.cost_model import VenueCostEstimate
from iqrp.app.execution.smart_routing.liquidity import LiquiditySnapshot
from iqrp.app.execution.smart_routing.venue import Venue

DEFAULT_WEIGHTS: dict[str, float] = {
    "price": 0.25,
    "fees": 0.10,
    "spread": 0.10,
    "liquidity": 0.15,
    "impact": 0.15,
    "fill_prob": 0.10,
    "latency": 0.05,
    "reliability": 0.10,
}


@dataclass(slots=True)
class ScoreWeights:
    """Configurable scoring weights (normalized on use)."""

    price: float = 0.25
    fees: float = 0.10
    spread: float = 0.10
    liquidity: float = 0.15
    impact: float = 0.15
    fill_prob: float = 0.10
    latency: float = 0.05
    reliability: float = 0.10

    @classmethod
    def from_mapping(cls, data: Mapping[str, float] | None) -> ScoreWeights:
        if not data:
            return cls()
        base = DEFAULT_WEIGHTS.copy()
        for key, value in data.items():
            if key in base:
                base[key] = float(value)
        return cls(**base)

    def normalized(self) -> dict[str, float]:
        raw = {
            "price": float(self.price),
            "fees": float(self.fees),
            "spread": float(self.spread),
            "liquidity": float(self.liquidity),
            "impact": float(self.impact),
            "fill_prob": float(self.fill_prob),
            "latency": float(self.latency),
            "reliability": float(self.reliability),
        }
        total = sum(max(v, 0.0) for v in raw.values())
        if total <= 0:
            return dict(DEFAULT_WEIGHTS)
        return {k: max(v, 0.0) / total for k, v in raw.items()}


@dataclass(slots=True)
class VenueScore:
    """Per-venue score with component breakdown."""

    venue_id: str
    score: float
    components: dict[str, float] = field(default_factory=dict)
    raw: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue_id": self.venue_id,
            "score": float(self.score),
            "components": dict(self.components),
            "raw": dict(self.raw),
        }


def _invert_cost(value: float, scale: float) -> float:
    """Map a non-negative cost-like value to [0, 1] where lower cost → higher score."""
    v = max(float(value), 0.0)
    s = max(float(scale), 1e-12)
    return 1.0 / (1.0 + v / s)


def _normalize_side_price(
    expected_price: float,
    *,
    is_buy: bool,
    peer_prices: list[float],
) -> float:
    """Normalize expected price across peers: better price → closer to 1."""
    if not peer_prices or expected_price <= 0:
        return 0.5
    lo = min(peer_prices)
    hi = max(peer_prices)
    if hi <= lo:
        return 1.0
    if is_buy:
        # Lower price is better for buys
        return (hi - expected_price) / (hi - lo)
    # Higher price is better for sells
    return (expected_price - lo) / (hi - lo)


def score_venue(
    venue: Venue,
    *,
    cost: VenueCostEstimate,
    liquidity: LiquiditySnapshot,
    weights: ScoreWeights | Mapping[str, float] | None = None,
    is_buy: bool = True,
    peer_prices: list[float] | None = None,
    latency_ref_ms: float = 50.0,
) -> VenueScore:
    """Compute weighted score for a single venue."""
    w = weights if isinstance(weights, ScoreWeights) else ScoreWeights.from_mapping(weights)
    wn = w.normalized()
    state = venue.venue_state

    price_comp = _normalize_side_price(
        cost.expected_price,
        is_buy=is_buy,
        peer_prices=peer_prices or [cost.expected_price],
    )
    fees_comp = _invert_cost(cost.fee_bps, scale=5.0)
    spread_comp = _invert_cost(cost.spread_bps, scale=10.0)
    liquidity_comp = min(max(float(liquidity.liquidity_score), 0.0), 1.0)
    # Prefer venues that can fill more of the order
    if liquidity.fillable_qty > 0:
        liquidity_comp = 0.5 * liquidity_comp + 0.5 * min(
            liquidity.fillable_qty / max(liquidity.fillable_qty, 1.0), 1.0
        )
    impact_comp = _invert_cost(cost.impact_bps, scale=20.0)
    fill_comp = min(max(float(state.fill_probability), 0.0), 1.0)
    latency_comp = _invert_cost(float(state.latency_ms), scale=float(latency_ref_ms))
    reliability_comp = min(max(float(state.reliability), 0.0), 1.0)

    # Soft preference multiplier
    pref = max(float(venue.preference), 0.0)

    components = {
        "price": price_comp,
        "fees": fees_comp,
        "spread": spread_comp,
        "liquidity": liquidity_comp,
        "impact": impact_comp,
        "fill_prob": fill_comp,
        "latency": latency_comp,
        "reliability": reliability_comp,
    }
    score = pref * sum(wn[k] * components[k] for k in wn)
    raw = {
        "expected_price": float(cost.expected_price),
        "fee_bps": float(cost.fee_bps),
        "spread_bps": float(cost.spread_bps),
        "impact_bps": float(cost.impact_bps),
        "liquidity_score": float(liquidity.liquidity_score),
        "fillable_qty": float(liquidity.fillable_qty),
        "fill_probability": float(state.fill_probability),
        "latency_ms": float(state.latency_ms),
        "reliability": float(state.reliability),
        "preference": float(pref),
    }
    return VenueScore(
        venue_id=venue.venue_id,
        score=float(score),
        components=components,
        raw=raw,
    )


def rank_venues(scores: list[VenueScore]) -> list[VenueScore]:
    """Return venues sorted by descending score."""
    return sorted(scores, key=lambda s: s.score, reverse=True)


__all__ = [
    "DEFAULT_WEIGHTS",
    "ScoreWeights",
    "VenueScore",
    "rank_venues",
    "score_venue",
]

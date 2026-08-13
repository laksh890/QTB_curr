"""Venue runtime state for smart order routing.

Tracks availability, halt status, latency, and liquidity quality.
A venue must be available, not halted, and trading-enabled to receive orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class VenueState:
    """Point-in-time venue operational state used by the router."""

    venue_id: str
    available: bool = True
    halted: bool = False
    trading_enabled: bool = True
    latency_ms: float = 5.0
    liquidity_score: float = 1.0  # [0, 1], higher is better
    reliability: float = 1.0  # [0, 1], historical fill/ack reliability
    fill_probability: float = 0.95  # [0, 1]
    bid: float | None = None
    ask: float | None = None
    mid: float | None = None
    spread_bps: float | None = None
    available_qty: float = 0.0
    adv: float = 0.0
    volatility: float = 0.0
    instruments: set[str] = field(default_factory=set)
    supported_order_types: set[str] = field(default_factory=set)
    fee_bps: float = 1.0
    maker_fee_bps: float = 0.5
    taker_fee_bps: float = 1.0
    tick_size: float = 0.01
    lot_size: float = 1.0
    min_qty: float = 1.0
    max_qty: float = 1e12
    kill_switch: bool = False
    last_update: datetime = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_routable(self) -> bool:
        """Return True when the venue may receive new orders."""
        return (
            bool(self.available)
            and not bool(self.halted)
            and bool(self.trading_enabled)
            and not bool(self.kill_switch)
        )

    def supports_instrument(self, instrument: str) -> bool:
        if not self.instruments:
            return True
        return str(instrument) in self.instruments

    def supports_order_type(self, order_type: object) -> bool:
        if not self.supported_order_types:
            return True
        key = getattr(order_type, "value", order_type)
        return str(key).upper() in {s.upper() for s in self.supported_order_types}

    def ensure_quotes(self) -> None:
        """Derive mid / spread from bid/ask when missing."""
        if self.bid is not None and self.ask is not None and self.bid > 0 and self.ask > 0:
            if self.mid is None:
                self.mid = 0.5 * (float(self.bid) + float(self.ask))
            if self.spread_bps is None and self.mid and self.mid > 0:
                self.spread_bps = 10_000.0 * (float(self.ask) - float(self.bid)) / float(self.mid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "venue_id": self.venue_id,
            "available": self.available,
            "halted": self.halted,
            "trading_enabled": self.trading_enabled,
            "latency_ms": float(self.latency_ms),
            "liquidity_score": float(self.liquidity_score),
            "reliability": float(self.reliability),
            "fill_probability": float(self.fill_probability),
            "bid": self.bid,
            "ask": self.ask,
            "mid": self.mid,
            "spread_bps": self.spread_bps,
            "available_qty": float(self.available_qty),
            "adv": float(self.adv),
            "volatility": float(self.volatility),
            "instruments": sorted(self.instruments),
            "supported_order_types": sorted(self.supported_order_types),
            "fee_bps": float(self.fee_bps),
            "maker_fee_bps": float(self.maker_fee_bps),
            "taker_fee_bps": float(self.taker_fee_bps),
            "tick_size": float(self.tick_size),
            "lot_size": float(self.lot_size),
            "min_qty": float(self.min_qty),
            "max_qty": float(self.max_qty),
            "kill_switch": self.kill_switch,
            "last_update": self.last_update.isoformat(),
            "metadata": dict(self.metadata),
        }


__all__ = ["VenueState"]

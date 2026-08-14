"""Base event types for the institutional event-driven backtester.

CRITICAL — Point-in-time correctness
------------------------------------
No event handler may access market data, features, labels, corporate actions,
universe membership, liquidity, volatility, or model parameters with an
effective timestamp strictly after ``event.timestamp``. Enforce via
:mod:`iqrp.app.backtesting.pit` helpers before any data read.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Canonical backtest event taxonomy (processing order by priority)."""

    MARKET = "MARKET"
    FEATURE = "FEATURE"
    SIGNAL = "SIGNAL"
    FORECAST = "FORECAST"
    RISK = "RISK"
    PORTFOLIO = "PORTFOLIO"
    ORDER = "ORDER"
    EXECUTION = "EXECUTION"
    FILL = "FILL"
    POSITION = "POSITION"
    PNL = "PNL"
    RISK_UPDATE = "RISK_UPDATE"
    SETTLEMENT = "SETTLEMENT"


# Lower number = earlier within the same timestamp.
EVENT_PRIORITY: dict[EventType, int] = {
    EventType.MARKET: 10,
    EventType.FEATURE: 20,
    EventType.SIGNAL: 30,
    EventType.FORECAST: 40,
    EventType.RISK: 50,
    EventType.PORTFOLIO: 60,
    EventType.ORDER: 70,
    EventType.EXECUTION: 80,
    EventType.FILL: 90,
    EventType.POSITION: 100,
    EventType.PNL: 110,
    EventType.RISK_UPDATE: 120,
    EventType.SETTLEMENT: 130,
}


def priority_for(event_type: EventType | str) -> int:
    """Return the canonical priority for an event type."""
    et = event_type if isinstance(event_type, EventType) else EventType(str(event_type))
    return EVENT_PRIORITY[et]


def _new_event_id() -> str:
    return uuid.uuid4().hex


class Event:
    """Simulation event.

    Attributes
    ----------
    timestamp:
        Simulation time at which the event becomes observable. Handlers must
        not read any information with effective time ``> timestamp``.
    event_type:
        Taxonomy member controlling default priority.
    priority:
        Tie-breaker within the same timestamp (lower runs earlier). Defaults
        from :data:`EVENT_PRIORITY`.
    payload:
        Event-specific data (prices, signals, fills, …).
    event_id:
        Stable unique identifier for idempotency / audit.
    """

    __slots__ = ("event_id", "event_type", "payload", "priority", "timestamp")

    def __init__(
        self,
        timestamp: datetime,
        event_type: EventType | str,
        *,
        priority: int | None = None,
        payload: Mapping[str, Any] | None = None,
        event_id: str | None = None,
    ) -> None:
        if timestamp.tzinfo is None:
            raise ValueError(f"Event timestamp must be timezone-aware (got naive {timestamp!r})")
        et = event_type if isinstance(event_type, EventType) else EventType(str(event_type))
        self.timestamp = timestamp
        self.event_type = et
        self.priority = priority_for(et) if priority is None else int(priority)
        self.payload: dict[str, Any] = dict(payload or {})
        self.event_id = event_id or _new_event_id()

    def with_payload(self, **updates: Any) -> Event:
        """Return a copy with merged payload (same type/priority/timestamp)."""
        cloned = object.__new__(type(self))
        Event.__init__(
            cloned,
            timestamp=self.timestamp,
            event_type=self.event_type,
            priority=self.priority,
            payload={**self.payload, **updates},
            event_id=self.event_id,
        )
        return cloned

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(id={self.event_id[:8]}…, "
            f"type={self.event_type.value}, ts={self.timestamp.isoformat()}, "
            f"priority={self.priority})"
        )


__all__ = [
    "EVENT_PRIORITY",
    "Event",
    "EventType",
    "priority_for",
]

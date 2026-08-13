"""Fill / execution-report events."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from iqrp.app.backtesting.event_engine.event import EVENT_PRIORITY, Event, EventType, _new_event_id


class FillEvent(Event):
    """Fill observed at ``timestamp`` (no future liquidity)."""

    def __init__(
        self,
        timestamp: datetime,
        payload: Mapping[str, Any] | None = None,
        *,
        priority: int | None = None,
        event_id: str | None = None,
    ) -> None:
        super().__init__(
            timestamp=timestamp,
            event_type=EventType.FILL,
            priority=EVENT_PRIORITY[EventType.FILL] if priority is None else priority,
            payload=dict(payload or {}),
            event_id=event_id or _new_event_id(),
        )

    @property
    def quantity(self) -> float | None:
        value = self.payload.get("quantity", self.payload.get("qty"))
        return None if value is None else float(value)

    @property
    def price(self) -> float | None:
        value = self.payload.get("price")
        return None if value is None else float(value)


__all__ = ["FillEvent"]

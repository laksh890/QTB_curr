"""Order creation / submission events."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from iqrp.app.backtesting.event_engine.event import EVENT_PRIORITY, Event, EventType, _new_event_id


class OrderEvent(Event):
    """Order intent created at ``timestamp``."""

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
            event_type=EventType.ORDER,
            priority=EVENT_PRIORITY[EventType.ORDER] if priority is None else priority,
            payload=dict(payload or {}),
            event_id=event_id or _new_event_id(),
        )

    @property
    def order_id(self) -> str | None:
        value = self.payload.get("order_id")
        return None if value is None else str(value)

    @property
    def symbol(self) -> str | None:
        value = self.payload.get("symbol")
        return None if value is None else str(value)


__all__ = ["OrderEvent"]

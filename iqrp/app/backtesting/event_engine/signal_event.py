"""Alpha / signal generation events."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from iqrp.app.backtesting.event_engine.event import EVENT_PRIORITY, Event, EventType, _new_event_id


class SignalEvent(Event):
    """Signal ready at ``timestamp`` (must be point-in-time)."""

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
            event_type=EventType.SIGNAL,
            priority=EVENT_PRIORITY[EventType.SIGNAL] if priority is None else priority,
            payload=dict(payload or {}),
            event_id=event_id or _new_event_id(),
        )

    @property
    def signal(self) -> Any:
        return self.payload.get("signal", self.payload.get("value"))


__all__ = ["SignalEvent"]

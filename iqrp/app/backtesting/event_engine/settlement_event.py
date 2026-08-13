"""Settlement / corporate-action application events."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

from iqrp.app.backtesting.event_engine.event import EVENT_PRIORITY, Event, EventType, _new_event_id


class SettlementEvent(Event):
    """Cash / position settlement at ``timestamp``."""

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
            event_type=EventType.SETTLEMENT,
            priority=EVENT_PRIORITY[EventType.SETTLEMENT] if priority is None else priority,
            payload=dict(payload or {}),
            event_id=event_id or _new_event_id(),
        )


__all__ = ["SettlementEvent"]

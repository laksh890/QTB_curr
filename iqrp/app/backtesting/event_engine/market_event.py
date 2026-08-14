"""Market data events (bars, ticks, quotes, books)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from iqrp.app.backtesting.event_engine.event import EVENT_PRIORITY, Event, EventType, _new_event_id


class MarketEvent(Event):
    """Market observation becoming available at ``timestamp``.

    Typical payload keys: ``symbol``, ``open``, ``high``, ``low``, ``close``,
    ``volume``, ``bid``, ``ask``, ``source``.
    """

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
            event_type=EventType.MARKET,
            priority=EVENT_PRIORITY[EventType.MARKET] if priority is None else priority,
            payload=dict(payload or {}),
            event_id=event_id or _new_event_id(),
        )

    @property
    def symbol(self) -> str | None:
        value = self.payload.get("symbol")
        return None if value is None else str(value)

    @property
    def close(self) -> float | None:
        value = self.payload.get("close")
        return None if value is None else float(value)


__all__ = ["MarketEvent"]

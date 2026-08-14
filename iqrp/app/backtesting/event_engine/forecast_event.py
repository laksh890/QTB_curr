"""Forecast generation events."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from iqrp.app.backtesting.event_engine.event import EVENT_PRIORITY, Event, EventType, _new_event_id


class ForecastEvent(Event):
    """Model forecast emitted at ``timestamp`` (training must be PIT-safe)."""

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
            event_type=EventType.FORECAST,
            priority=EVENT_PRIORITY[EventType.FORECAST] if priority is None else priority,
            payload=dict(payload or {}),
            event_id=event_id or _new_event_id(),
        )

    @property
    def model_version(self) -> str | None:
        value = self.payload.get("model_version")
        return None if value is None else str(value)


__all__ = ["ForecastEvent"]

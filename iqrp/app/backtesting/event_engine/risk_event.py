"""Risk evaluation and risk-update events."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from iqrp.app.backtesting.event_engine.event import EVENT_PRIORITY, Event, EventType, _new_event_id


class RiskEvent(Event):
    """Pre-trade / interim risk evaluation at ``timestamp``."""

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
            event_type=EventType.RISK,
            priority=EVENT_PRIORITY[EventType.RISK] if priority is None else priority,
            payload=dict(payload or {}),
            event_id=event_id or _new_event_id(),
        )


class RiskUpdateEvent(Event):
    """Post-fill / mark-to-market risk update at ``timestamp``."""

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
            event_type=EventType.RISK_UPDATE,
            priority=(EVENT_PRIORITY[EventType.RISK_UPDATE] if priority is None else priority),
            payload=dict(payload or {}),
            event_id=event_id or _new_event_id(),
        )


__all__ = ["RiskEvent", "RiskUpdateEvent"]

"""Portfolio construction / target-weight events."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from iqrp.app.backtesting.event_engine.event import EVENT_PRIORITY, Event, EventType, _new_event_id


class PortfolioEvent(Event):
    """Portfolio decision at ``timestamp``."""

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
            event_type=EventType.PORTFOLIO,
            priority=EVENT_PRIORITY[EventType.PORTFOLIO] if priority is None else priority,
            payload=dict(payload or {}),
            event_id=event_id or _new_event_id(),
        )

    @property
    def targets(self) -> Mapping[str, float]:
        raw = self.payload.get("targets", self.payload.get("weights", {}))
        return {str(k): float(v) for k, v in dict(raw or {}).items()}


__all__ = ["PortfolioEvent"]

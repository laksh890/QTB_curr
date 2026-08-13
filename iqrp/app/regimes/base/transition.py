"""First-class regime transition objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class RegimeTransition:
    """A transition between two regime states."""

    previous_state: int
    current_state: int
    probability: float
    confidence: float
    timestamp: datetime | None = None
    previous_name: str | None = None
    current_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if isinstance(self.timestamp, datetime):
            d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegimeTransition:
        ts = data.get("timestamp")
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return cls(
            previous_state=int(data["previous_state"]),
            current_state=int(data["current_state"]),
            probability=float(data["probability"]),
            confidence=float(data["confidence"]),
            timestamp=ts,
            previous_name=data.get("previous_name"),
            current_name=data.get("current_name"),
            metadata=dict(data.get("metadata") or {}),
        )

"""Strongly typed regime state object."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class RegimeState:
    """A single regime assignment at a point in time (or segment)."""

    state_id: int
    state_name: str
    probability: float
    confidence: float
    persistence: float
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration: float | None = None
    features_used: tuple[str, ...] = ()
    model_version: str = "0.0.0"
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("start_time", "end_time", "timestamp"):
            val = d.get(key)
            if isinstance(val, datetime):
                d[key] = val.isoformat()
        d["features_used"] = list(self.features_used)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegimeState:
        def _parse_ts(value: Any) -> datetime | None:
            if value is None:
                return None
            if isinstance(value, datetime):
                return value
            return datetime.fromisoformat(str(value))

        return cls(
            state_id=int(data["state_id"]),
            state_name=str(data["state_name"]),
            probability=float(data["probability"]),
            confidence=float(data["confidence"]),
            persistence=float(data["persistence"]),
            start_time=_parse_ts(data.get("start_time")),
            end_time=_parse_ts(data.get("end_time")),
            duration=None if data.get("duration") is None else float(data["duration"]),
            features_used=tuple(data.get("features_used") or ()),
            model_version=str(data.get("model_version", "0.0.0")),
            timestamp=_parse_ts(data.get("timestamp")),
            metadata=dict(data.get("metadata") or {}),
        )

    @staticmethod
    def now_utc() -> datetime:
        return datetime.now(UTC)

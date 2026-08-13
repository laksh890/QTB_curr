"""Strongly typed latent-state value object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class LatentState:
    """Discrete or indexed latent state at a single time index."""

    state_id: int
    state_name: str
    probability: float
    confidence: float
    timestamp: Any = None
    duration: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": int(self.state_id),
            "state_name": self.state_name,
            "probability": float(self.probability),
            "confidence": float(self.confidence),
            "timestamp": self.timestamp,
            "duration": None if self.duration is None else float(self.duration),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LatentState:
        return cls(
            state_id=int(data["state_id"]),
            state_name=str(data["state_name"]),
            probability=float(data["probability"]),
            confidence=float(data.get("confidence", data["probability"])),
            timestamp=data.get("timestamp"),
            duration=None if data.get("duration") is None else float(data["duration"]),
            metadata=dict(data.get("metadata") or {}),
        )

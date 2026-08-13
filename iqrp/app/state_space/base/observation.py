"""Observation value object for state-space models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Observation:
    """Single-time observation vector with optional metadata."""

    values: np.ndarray
    timestamp: Any = None
    mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", np.asarray(self.values, dtype=np.float64).reshape(-1))
        if self.mask is not None:
            object.__setattr__(self, "mask", np.asarray(self.mask, dtype=bool).reshape(-1))

    @property
    def dim(self) -> int:
        return int(self.values.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values.tolist(),
            "timestamp": self.timestamp,
            "mask": None if self.mask is None else self.mask.tolist(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        return cls(
            values=np.asarray(data["values"], dtype=np.float64),
            timestamp=data.get("timestamp"),
            mask=None if data.get("mask") is None else np.asarray(data["mask"], dtype=bool),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def from_frame_row(cls, values: Any, *, timestamp: Any = None) -> Observation:
        return cls(values=np.asarray(values, dtype=np.float64), timestamp=timestamp)

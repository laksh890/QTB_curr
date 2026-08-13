"""Alpha signal container.

CRITICAL:
- Statistical significance alone ≠ alpha.
- Historical Sharpe alone cannot approve a signal.
- Signal values must be computed point-in-time (no future leakage).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class AlphaSignal:
    """Numeric alpha-candidate series with optional timestamps and metadata.

    This object holds *candidate* signal values for research. Presence of a
    series does **not** imply economic alpha or tradable profitability.
    """

    values: np.ndarray
    timestamps: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    name: str = ""
    definition_id: str | None = None

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.values.ndim != 1:
            raise ValueError(f"AlphaSignal.values must be 1-D, got shape {self.values.shape}")
        if self.timestamps is not None:
            self.timestamps = np.asarray(self.timestamps)
            if len(self.timestamps) != len(self.values):
                raise ValueError(
                    f"timestamps length {len(self.timestamps)} != values length {len(self.values)}"
                )

    @property
    def length(self) -> int:
        return int(len(self.values))

    @property
    def n_finite(self) -> int:
        return int(np.isfinite(self.values).sum())

    def copy(self) -> AlphaSignal:
        return AlphaSignal(
            values=self.values.copy(),
            timestamps=None if self.timestamps is None else self.timestamps.copy(),
            metadata=dict(self.metadata),
            name=self.name,
            definition_id=self.definition_id,
        )

    def with_metadata(self, **kwargs: Any) -> AlphaSignal:
        meta = dict(self.metadata)
        meta.update(kwargs)
        return AlphaSignal(
            values=self.values,
            timestamps=self.timestamps,
            metadata=meta,
            name=self.name,
            definition_id=self.definition_id,
        )

    def slice(self, start: int, end: int | None = None) -> AlphaSignal:
        end = len(self.values) if end is None else end
        ts = None if self.timestamps is None else self.timestamps[start:end]
        return AlphaSignal(
            values=self.values[start:end],
            timestamps=ts,
            metadata=dict(self.metadata),
            name=self.name,
            definition_id=self.definition_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "definition_id": self.definition_id,
            "values": self.values.tolist(),
            "timestamps": None if self.timestamps is None else self.timestamps.tolist(),
            "metadata": dict(self.metadata),
            "length": self.length,
            "n_finite": self.n_finite,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlphaSignal:
        ts = data.get("timestamps")
        return cls(
            values=np.asarray(data["values"], dtype=np.float64),
            timestamps=None if ts is None else np.asarray(ts),
            metadata=dict(data.get("metadata") or {}),
            name=str(data.get("name") or ""),
            definition_id=data.get("definition_id"),
        )

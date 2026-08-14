"""Feature matrix snapshots aligned to retraining events."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


@dataclass
class FeatureSnapshot:
    """Point-in-time feature payload used for a retrain."""

    version: int
    features: Any
    """Array-like feature matrix or frame."""
    start: int
    end: int
    """Half-open ``[start, end)`` row span covered by this snapshot."""
    columns: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        try:
            return int(np.asarray(self.features).shape[0])
        except Exception:
            return max(0, int(self.end) - int(self.start))

    @property
    def n_cols(self) -> int:
        if self.columns:
            return len(self.columns)
        try:
            arr = np.asarray(self.features)
            return int(arr.shape[1]) if arr.ndim > 1 else 1
        except Exception:
            return 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "start": int(self.start),
            "end": int(self.end),
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": list(self.columns),
            "created_at": self.created_at.isoformat(),
            "metadata": dict(self.metadata),
        }


class FeatureSnapshotStore:
    """In-memory store of feature snapshots."""

    def __init__(self) -> None:
        self._items: list[FeatureSnapshot] = []

    @property
    def size(self) -> int:
        return len(self._items)

    def save(
        self,
        features: Any,
        *,
        start: int,
        end: int,
        columns: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        deepcopy_features: bool = True,
    ) -> FeatureSnapshot:
        version = 1 if not self._items else int(self._items[-1].version) + 1
        payload = deepcopy(features) if deepcopy_features else features
        snap = FeatureSnapshot(
            version=version,
            features=payload,
            start=int(start),
            end=int(end),
            columns=list(columns or []),
            metadata=dict(metadata or {}),
        )
        self._items.append(snap)
        return snap

    def get(self, version: int) -> FeatureSnapshot | None:
        for s in self._items:
            if s.version == int(version):
                return s
        return None

    def latest(self) -> FeatureSnapshot | None:
        return self._items[-1] if self._items else None

    def history(self) -> list[FeatureSnapshot]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()

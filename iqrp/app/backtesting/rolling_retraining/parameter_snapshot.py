"""Hyperparameter / configuration snapshots for retraining audits."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass
class ParameterSnapshot:
    """Versioned parameter / hyperparameter bundle."""

    version: int
    params: dict[str, Any]
    created_at: datetime = field(default_factory=_utc_now)
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "params": dict(self.params),
            "created_at": self.created_at.isoformat(),
            "source": self.source,
            "metadata": dict(self.metadata),
        }


class ParameterSnapshotStore:
    """In-memory store of parameter snapshots."""

    def __init__(self) -> None:
        self._items: list[ParameterSnapshot] = []

    @property
    def size(self) -> int:
        return len(self._items)

    def save(
        self,
        params: dict[str, Any],
        *,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ParameterSnapshot:
        version = 1 if not self._items else int(self._items[-1].version) + 1
        snap = ParameterSnapshot(
            version=version,
            params=deepcopy(dict(params)),
            source=source,
            metadata=dict(metadata or {}),
        )
        self._items.append(snap)
        return snap

    def get(self, version: int) -> ParameterSnapshot | None:
        for s in self._items:
            if s.version == int(version):
                return s
        return None

    def latest(self) -> ParameterSnapshot | None:
        return self._items[-1] if self._items else None

    def history(self) -> list[ParameterSnapshot]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()

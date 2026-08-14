"""Versioned in-memory model snapshots for rolling retraining."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


@dataclass
class ModelSnapshot:
    """Immutable-ish record of a fitted model at a point in time."""

    version: int
    model: Any
    trained_through: int
    """Last inclusive observation index used for training."""
    created_at: datetime = field(default_factory=_utc_now)
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    feature_version: int | None = None
    parameter_version: int | None = None
    trigger: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "trained_through": int(self.trained_through),
            "created_at": self.created_at.isoformat(),
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
            "feature_version": self.feature_version,
            "parameter_version": self.parameter_version,
            "trigger": self.trigger,
        }


class ModelRegistry:
    """In-memory versioned registry of model snapshots."""

    def __init__(self) -> None:
        self._snapshots: list[ModelSnapshot] = []
        self._by_version: dict[int, ModelSnapshot] = {}
        self._active_version: int | None = None

    @property
    def size(self) -> int:
        return len(self._snapshots)

    @property
    def active_version(self) -> int | None:
        return self._active_version

    def register(
        self,
        model: Any,
        *,
        trained_through: int,
        metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        feature_version: int | None = None,
        parameter_version: int | None = None,
        trigger: str | None = None,
        activate: bool = True,
        deepcopy_model: bool = True,
    ) -> ModelSnapshot:
        version = 1 if not self._snapshots else int(self._snapshots[-1].version) + 1
        snap = ModelSnapshot(
            version=version,
            model=deepcopy(model) if deepcopy_model else model,
            trained_through=int(trained_through),
            metrics=dict(metrics or {}),
            metadata=dict(metadata or {}),
            feature_version=feature_version,
            parameter_version=parameter_version,
            trigger=trigger,
        )
        self._snapshots.append(snap)
        self._by_version[version] = snap
        if activate:
            self._active_version = version
        return snap

    def get(self, version: int | None = None) -> ModelSnapshot | None:
        if version is None:
            if self._active_version is None:
                return None
            return self._by_version.get(self._active_version)
        return self._by_version.get(int(version))

    def active(self) -> ModelSnapshot | None:
        return self.get(None)

    def activate(self, version: int) -> ModelSnapshot:
        snap = self._by_version.get(int(version))
        if snap is None:
            raise KeyError(f"Unknown model version {version}")
        self._active_version = int(version)
        return snap

    def history(self) -> list[ModelSnapshot]:
        return list(self._snapshots)

    def versions(self) -> list[int]:
        return [s.version for s in self._snapshots]

    def latest(self) -> ModelSnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    def clear(self) -> None:
        self._snapshots.clear()
        self._by_version.clear()
        self._active_version = None

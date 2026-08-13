"""Deployment helpers for selected forecast models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from iqrp.app.forecasting.intelligence.serializer import IntelligenceSerializer


@dataclass
class DeploymentRecord:
    model_name: str
    version: str
    path: str
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "path": self.path,
            "status": self.status,
            "metadata": dict(self.metadata),
        }


class DeploymentManager:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root else Path("artifacts/forecast_intelligence")
        self.root.mkdir(parents=True, exist_ok=True)
        self._active: DeploymentRecord | None = None
        self._history: list[DeploymentRecord] = []
        self._serializer = IntelligenceSerializer()

    def deploy(self, engine: Any, *, name: str = "intelligence") -> DeploymentRecord:
        path = self.root / f"{name}.json"
        self._serializer.save(engine, path)
        version = str(getattr(getattr(engine, "settings", None), "seed", "1.0.0"))
        rec = DeploymentRecord(
            model_name=getattr(engine, "best_model_name", name),
            version=version,
            path=str(path),
            status="active",
            metadata={"leaderboard_size": len(getattr(engine, "_leaderboard", []) or [])},
        )
        if self._active is not None:
            self._active.status = "retired"
            self._history.append(self._active)
        self._active = rec
        return rec

    def rollback(self) -> DeploymentRecord | None:
        if not self._history:
            return self._active
        prev = self._history.pop()
        prev.status = "active"
        if self._active is not None:
            self._active.status = "rolled_back"
        self._active = prev
        return prev

    @property
    def active(self) -> DeploymentRecord | None:
        return self._active

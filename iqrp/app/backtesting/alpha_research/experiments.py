"""Reproducible alpha experiment registry (disk-backed)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class ExperimentSpec:
    experiment_id: str
    timestamp: str
    dataset_id: str
    dataset_checksum: str
    feature_versions: dict[str, str]
    signal_id: str
    signal_version: str
    parameters: dict[str, Any]
    timeframe: str
    holding_period: int
    cost_model: dict[str, Any]
    risk_configuration: dict[str, Any] = field(default_factory=dict)
    random_seed: int | None = None
    software_version: str = "iqrp-alpha-research-0.1.0"
    result_checksum: str = ""
    classification: str = ""
    matrix_row: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExperimentRegistry:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else Path("alpha_experiment_registry.json")
        self._items: dict[str, ExperimentSpec] = {}
        if self.path.exists():
            self.load()

    def register(self, spec: ExperimentSpec, *, persist: bool = True) -> ExperimentSpec:
        self._items[spec.experiment_id] = spec
        if persist:
            self.save()
        return spec

    def list(self) -> list[ExperimentSpec]:
        return [self._items[k] for k in sorted(self._items)]

    def save(self) -> Path:
        payload = {
            "updated_at": datetime.now(UTC).isoformat(),
            "experiments": [e.to_dict() for e in self.list()],
            "disclaimer": "Research experiments only — not profitability claims.",
        }
        self.path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return self.path

    def load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._items = {}
        for row in data.get("experiments", []):
            known = {f.name for f in ExperimentSpec.__dataclass_fields__.values()}  # type: ignore[attr-defined]
            self._items[row["experiment_id"]] = ExperimentSpec(
                **{k: v for k, v in row.items() if k in known}
            )

    @staticmethod
    def result_checksum(payload: MappingLike) -> str:
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    @staticmethod
    def new_id() -> str:
        return str(uuid4())


# typing helper
MappingLike = dict[str, Any]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["ExperimentRegistry", "ExperimentSpec", "now_iso"]

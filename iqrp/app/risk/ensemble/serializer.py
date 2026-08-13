"""JSON serialization for ensemble assessments, decisions, and machine state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.risk.ensemble.types import EnsembleDecision, RiskAssessment


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return _to_jsonable(obj.to_dict())
    if hasattr(obj, "model_dump"):
        return _to_jsonable(obj.model_dump())
    if hasattr(obj, "export_state"):
        return _to_jsonable(obj.export_state())
    try:
        from enum import Enum

        if isinstance(obj, Enum):
            return obj.value
    except Exception:  # noqa: BLE001
        pass
    return str(obj)


class EnsembleSerializer:
    def to_json(self, obj: Any, *, indent: int | None = 2) -> str:
        return json.dumps(_to_jsonable(obj), indent=indent)

    def assessment_to_dict(self, assessment: RiskAssessment) -> dict[str, Any]:
        return _to_jsonable(assessment.to_dict())

    def decision_to_dict(self, decision: EnsembleDecision) -> dict[str, Any]:
        return _to_jsonable(decision.to_dict())

    def ensemble_state_to_dict(self, ensemble: Any) -> dict[str, Any]:
        if hasattr(ensemble, "export_state"):
            return _to_jsonable(ensemble.export_state())
        return {"ensemble": str(ensemble)}

    def save(self, obj: Any, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(obj), encoding="utf-8")
        return p

    def load(self, path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def dump_bytes(self, obj: Any) -> bytes:
        return self.to_json(obj, indent=None).encode("utf-8")

    def load_bytes(self, data: bytes) -> dict[str, Any]:
        return json.loads(data.decode("utf-8"))

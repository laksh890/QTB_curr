"""Serialize / deserialize fitted state-space models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from iqrp.app.state_space.base.state_space_model import StateSpaceModel

M = TypeVar("M", bound=StateSpaceModel)


class StateSpaceSerializer:
    """JSON artifact writer/reader with optional NumPy sidecar."""

    def save(self, model: StateSpaceModel, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = model.export_state()
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        arrays = _extract_arrays(payload.get("algorithm_state") or {})
        if arrays:
            np.savez_compressed(path.with_suffix(".npz"), **arrays)  # type: ignore[arg-type]
        return path

    def load(self, path: Path, *, model_cls: type[M]) -> M:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        sidecar = path.with_suffix(".npz")
        if sidecar.exists():
            data = np.load(sidecar)
            algo = dict(payload.get("algorithm_state") or {})
            for key in data.files:
                algo[key] = data[key].tolist()
            payload["algorithm_state"] = algo
        model = model_cls()
        model.import_state(payload)
        return model


def _extract_arrays(algorithm_state: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key in ("transition_matrix", "means", "variances", "initial"):
        if key in algorithm_state and algorithm_state[key] is not None:
            out[key] = np.asarray(algorithm_state[key], dtype=np.float64)
    return out


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")

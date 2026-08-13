"""Serialize / deserialize ensemble regime models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

M = TypeVar("M")


class EnsembleSerializer:
    def save(self, model: Any, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = model.export_state()
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        algo = payload.get("algorithm_state") or {}
        arrays: dict[str, np.ndarray] = {}
        for key in ("weights", "ensemble_proba", "transition", "last_hard"):
            if key in algo and algo[key] is not None:
                arrays[key] = np.asarray(algo[key], dtype=np.float64)
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
        model.import_state(payload)  # type: ignore[attr-defined]
        return model


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")

"""Serialize / deserialize Kalman filter models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

M = TypeVar("M")


class KalmanSerializer:
    def save(self, model: Any, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = model.export_state()
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        algo = payload.get("algorithm_state") or {}
        arrays: dict[str, np.ndarray] = {}
        for key in (
            "means",
            "covs",
            "pred_means",
            "pred_covs",
            "innovations",
            "innovation_covs",
            "gains",
            "smooth_means",
            "smooth_covs",
        ):
            if key in algo and algo[key] is not None:
                arrays[key] = np.asarray(algo[key], dtype=np.float64)
        system = algo.get("system") or {}
        for key in ("f", "h", "q", "r", "x0", "p0"):
            if key in system and system[key] is not None:
                arrays[f"system_{key}"] = np.asarray(system[key], dtype=np.float64)
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
            system = dict(algo.get("system") or {})
            for key in data.files:
                if key.startswith("system_"):
                    system[key.removeprefix("system_")] = data[key].tolist()
                else:
                    algo[key] = data[key].tolist()
            if system:
                algo["system"] = system
            payload["algorithm_state"] = algo
        model = model_cls()
        model.import_state(payload)  # type: ignore[attr-defined]
        return model


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")

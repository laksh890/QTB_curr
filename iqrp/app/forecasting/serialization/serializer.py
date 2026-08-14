"""JSON (+ optional NPZ) serialization for forecasting models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

M = TypeVar("M")


class ForecastSerializer:
    def save(self, model: Any, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = model.export_state()
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        include_npz = True
        settings = getattr(model, "settings", None) or getattr(model, "_settings", None)
        if settings is not None:
            include_npz = bool(
                getattr(getattr(settings, "serialization", None), "include_npz", True)
            )
        if include_npz:
            arrays = _extract_arrays(payload.get("algorithm_state") or {})
            if arrays:
                np.savez_compressed(path.with_suffix(".npz"), **arrays)
        return path

    def load(self, path: Path, *, model_cls: type[M]) -> M:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        npz_path = path.with_suffix(".npz")
        if npz_path.is_file():
            with np.load(npz_path, allow_pickle=False) as data:
                algo = payload.setdefault("algorithm_state", {})
                for key in data.files:
                    algo[key] = data[key].tolist()
        model = model_cls()
        model.import_state(payload)  # type: ignore[attr-defined]
        return model


def _extract_arrays(state: dict[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, v in state.items():
        if isinstance(v, list) and v and isinstance(v[0], (list, float, int)):
            try:
                arr = np.asarray(v, dtype=np.float64)
                if arr.size >= 16:
                    out[k] = arr
            except Exception:
                continue
        elif isinstance(v, np.ndarray):
            out[k] = v
    return out


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")

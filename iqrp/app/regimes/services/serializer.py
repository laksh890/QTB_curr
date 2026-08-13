"""Serialize / deserialize fitted regime models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from iqrp.app.regimes.base.regime_model import RegimeModel

M = TypeVar("M", bound=RegimeModel)


class RegimeSerializer:
    """JSON artifact writer/reader for regime models."""

    def save(self, model: RegimeModel, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = model.export_state()
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        # Optional numpy sidecar for large arrays
        sidecar = path.with_suffix(".npz")
        tm = payload.get("transition_matrix")
        if tm is not None:
            np.savez_compressed(
                sidecar,
                transition_matrix=np.asarray(tm, dtype=np.float64),
            )
        return path

    def load(self, path: Path, *, model_cls: type[M]) -> M:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        sidecar = path.with_suffix(".npz")
        if sidecar.exists():
            data = np.load(sidecar)
            if "transition_matrix" in data:
                payload["transition_matrix"] = data["transition_matrix"].tolist()
        model = model_cls()
        model.import_state(payload)
        return model


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")

"""Serialization for Forecast Intelligence Engine state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


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
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


class IntelligenceSerializer:
    def save(self, engine: Any, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            engine.export_state() if hasattr(engine, "export_state") else {"engine": str(engine)}
        )
        p.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
        return p

    def load(self, path: str | Path) -> dict[str, Any]:
        p = Path(path)
        return json.loads(p.read_text(encoding="utf-8"))

    def dump_bytes(self, engine: Any) -> bytes:
        payload = (
            engine.export_state() if hasattr(engine, "export_state") else {"engine": str(engine)}
        )
        return json.dumps(_to_jsonable(payload)).encode("utf-8")

    def load_bytes(self, data: bytes) -> dict[str, Any]:
        return json.loads(data.decode("utf-8"))

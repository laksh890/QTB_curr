"""JSON serialization for CapitalAllocation and CapitalAllocator state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.risk.capital.types import CapitalAllocation


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
    return str(obj)


class CapitalSerializer:
    """Save / load CapitalAllocation results and allocator snapshots."""

    def save_allocation(self, allocation: CapitalAllocation | dict[str, Any], path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = allocation.to_dict() if hasattr(allocation, "to_dict") else dict(allocation)
        p.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
        return p

    def load_allocation(self, path: str | Path) -> CapitalAllocation:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return CapitalAllocation.from_dict(data)

    def save_state(self, allocator: Any, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(allocator, "export_state"):
            payload = allocator.export_state()
        elif hasattr(allocator, "settings"):
            settings = allocator.settings
            payload = {
                "settings": settings.model_dump() if hasattr(settings, "model_dump") else dict(settings),
                "last_allocation": (
                    allocator.last_allocation.to_dict()
                    if getattr(allocator, "last_allocation", None) is not None
                    else None
                ),
            }
        else:
            payload = {"allocator": str(allocator)}
        p.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
        return p

    def load_state(self, path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def dump_bytes(self, obj: Any) -> bytes:
        if hasattr(obj, "to_dict"):
            payload = obj.to_dict()
        elif hasattr(obj, "export_state"):
            payload = obj.export_state()
        else:
            payload = {"value": str(obj)}
        return json.dumps(_to_jsonable(payload)).encode("utf-8")

    def load_bytes(self, data: bytes) -> dict[str, Any]:
        return json.loads(data.decode("utf-8"))

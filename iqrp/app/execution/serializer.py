"""JSON serialization for execution objects and reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.execution.order_manager.order import Order
from iqrp.app.execution.order_manager.parent_order import ParentOrder


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
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return _to_jsonable(obj.to_dict())
    if hasattr(obj, "model_dump"):
        return _to_jsonable(obj.model_dump())
    if hasattr(obj, "value") and hasattr(type(obj), "__mro__"):
        try:
            from enum import Enum

            if isinstance(obj, Enum):
                return obj.value
        except Exception:
            pass
    return str(obj)


class ExecutionSerializer:
    """Save / load orders, parents, and execution reports."""

    def save(self, obj: Any, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(obj, "to_dict"):
            payload = obj.to_dict()
        elif hasattr(obj, "model_dump"):
            payload = obj.model_dump()
        elif isinstance(obj, dict):
            payload = obj
        else:
            payload = {"value": str(obj)}
        p.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
        return p

    def load(self, path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def save_order(self, order: Order | dict[str, Any], path: str | Path) -> Path:
        payload = order.to_dict() if hasattr(order, "to_dict") else dict(order)
        return self.save(payload, path)

    def load_order(self, path: str | Path) -> Order:
        return Order.from_dict(self.load(path))

    def save_parent(self, parent: ParentOrder | dict[str, Any], path: str | Path) -> Path:
        payload = parent.to_dict() if hasattr(parent, "to_dict") else dict(parent)
        return self.save(payload, path)

    def load_parent(self, path: str | Path) -> ParentOrder:
        data = self.load(path)
        return ParentOrder(
            parent_id=str(data.get("parent_id") or ""),
            instrument=str(data["instrument"]),
            side=data["side"],
            quantity=float(data["quantity"]),
            strategy_id=data.get("strategy_id"),
            portfolio_id=data.get("portfolio_id"),
            urgency=data.get("urgency", "NORMAL"),
            algo=data.get("algo"),
            filled_qty=float(data.get("filled_qty", 0.0)),
            child_ids=list(data.get("child_ids") or []),
            metadata=dict(data.get("metadata") or {}),
        )

    def dump_bytes(self, obj: Any) -> bytes:
        if hasattr(obj, "to_dict"):
            payload = obj.to_dict()
        elif hasattr(obj, "model_dump"):
            payload = obj.model_dump()
        elif isinstance(obj, dict):
            payload = obj
        else:
            payload = {"value": str(obj)}
        return json.dumps(_to_jsonable(payload)).encode("utf-8")

    def load_bytes(self, data: bytes) -> dict[str, Any]:
        return json.loads(data.decode("utf-8"))


__all__ = ["ExecutionSerializer", "_to_jsonable"]

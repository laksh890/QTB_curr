"""JSON serialization for portfolio objects and optimization results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.portfolio.base.optimizer import OptimizationResult
from iqrp.app.portfolio.base.portfolio import Portfolio
from iqrp.app.portfolio.base.position import Position


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
    if hasattr(obj, "value") and hasattr(type(obj), "__mro__"):
        try:
            from enum import Enum

            if isinstance(obj, Enum):
                return obj.value
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


class PortfolioSerializer:
    """Save / load Portfolio, Position, and OptimizationResult payloads."""

    def save_portfolio(self, portfolio: Portfolio | dict[str, Any], path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = portfolio.to_dict() if hasattr(portfolio, "to_dict") else dict(portfolio)
        p.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
        return p

    def load_portfolio(self, path: str | Path) -> Portfolio:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return Portfolio.from_dict(data)

    def save_result(self, result: OptimizationResult | dict[str, Any], path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        p.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
        return p

    def load_result(self, path: str | Path) -> OptimizationResult:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return OptimizationResult.from_dict(data)

    def save_position(self, position: Position | dict[str, Any], path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = position.to_dict() if hasattr(position, "to_dict") else dict(position)
        p.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
        return p

    def load_position(self, path: str | Path) -> Position:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return Position.from_dict(data)

    def dump_bytes(self, obj: Any) -> bytes:
        if hasattr(obj, "to_dict"):
            payload = obj.to_dict()
        elif hasattr(obj, "model_dump"):
            payload = obj.model_dump()
        else:
            payload = {"value": str(obj)}
        return json.dumps(_to_jsonable(payload)).encode("utf-8")

    def load_bytes(self, data: bytes) -> dict[str, Any]:
        return json.loads(data.decode("utf-8"))

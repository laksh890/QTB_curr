"""Serialize / deserialize backtest results and experiment records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = [
    "to_jsonable",
    "serialize_result",
    "deserialize_result",
    "save_json",
    "load_json",
]


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / dataclasses / enums to JSON-safe types."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return to_jsonable(obj.to_dict())
    if hasattr(obj, "value") and not callable(obj.value):
        # Enum-like
        try:
            return to_jsonable(obj.value)
        except Exception:  # noqa: BLE001
            pass
    if hasattr(obj, "__dict__"):
        return to_jsonable({k: v for k, v in vars(obj).items() if not k.startswith("_")})
    return str(obj)


def serialize_result(result: Any) -> dict[str, Any]:
    """Serialize a :class:`BacktestResult` (or mapping) to a plain dict."""
    if hasattr(result, "to_dict"):
        return to_jsonable(result.to_dict())
    if isinstance(result, Mapping):
        return to_jsonable(dict(result))
    raise TypeError(f"cannot serialize result of type {type(result)!r}")


def deserialize_result(data: Mapping[str, Any]) -> Any:
    """Rehydrate a :class:`BacktestResult` from a mapping."""
    from iqrp.app.backtesting.engine import BacktestResult

    return BacktestResult.from_dict(data)


def save_json(path: str | Path, payload: Any) -> Path:
    """Write JSON-serializable payload to ``path``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    return out


def load_json(path: str | Path) -> Any:
    """Load JSON from ``path``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))

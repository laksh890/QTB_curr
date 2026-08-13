"""JSON serialization for alpha research objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.alpha.base.alpha_signal import AlphaSignal
from iqrp.app.alpha.base.signal_definition import SignalDefinition
from iqrp.app.alpha.base.signal_metadata import SignalMetadata
from iqrp.app.alpha.base.signal_result import SignalResearchReport, SignalStatus


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
    if isinstance(obj, SignalStatus):
        return obj.value
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


class AlphaSerializer:
    """Save / load alpha signals, definitions, and research reports."""

    def save_signal(self, signal: AlphaSignal | dict[str, Any], path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = signal.to_dict() if hasattr(signal, "to_dict") else dict(signal)
        p.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
        return p

    def load_signal(self, path: str | Path) -> AlphaSignal:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return AlphaSignal.from_dict(data)

    def save_definition(
        self, definition: SignalDefinition | dict[str, Any], path: str | Path
    ) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = definition.to_dict() if hasattr(definition, "to_dict") else dict(definition)
        p.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
        return p

    def load_definition(self, path: str | Path) -> SignalDefinition:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return SignalDefinition.from_dict(data)

    def save_report(
        self, report: SignalResearchReport | dict[str, Any], path: str | Path
    ) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        p.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
        return p

    def load_report(self, path: str | Path) -> SignalResearchReport:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return SignalResearchReport.from_dict(data)

    def save_metadata(
        self, metadata: SignalMetadata | dict[str, Any], path: str | Path
    ) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = metadata.to_dict() if hasattr(metadata, "to_dict") else dict(metadata)
        p.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
        return p

    def load_metadata(self, path: str | Path) -> SignalMetadata:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return SignalMetadata.from_dict(data)

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

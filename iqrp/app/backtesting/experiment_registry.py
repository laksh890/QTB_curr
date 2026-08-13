"""Experiment registry with full lineage for institutional backtests.

Every completed / failed / invalidated run is preserved for audit. Lineage
covers data, feature, label, model, risk, portfolio, execution, and code
versions plus seed.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping

from iqrp.app.backtesting.serializer import load_json, save_json, to_jsonable
from iqrp.app.backtesting.types import BacktestState, JSONDict

__all__ = [
    "ExperimentRecord",
    "ExperimentLineage",
    "ExperimentRegistry",
]


@dataclass
class ExperimentLineage:
    """Version fingerprint for reproducibility."""

    data_version: str = "1.0.0"
    feature_version: str = "1.0.0"
    label_version: str = "1.0.0"
    model_version: str = "1.0.0"
    risk_version: str = "1.0.0"
    portfolio_version: str = "1.0.0"
    execution_version: str = "1.0.0"
    code_version: str = "1.0.0"
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ExperimentLineage":
        if not data:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known and k != "extra"}
        extra = dict(data.get("extra") or {})
        for k, v in data.items():
            if k not in known:
                extra[k] = v
        kwargs["extra"] = extra
        return cls(**kwargs)

    @classmethod
    def from_settings(cls, settings: Any, *, seed: int | None = None) -> "ExperimentLineage":
        repro = getattr(settings, "reproducibility", None)
        if repro is None:
            return cls(seed=42 if seed is None else int(seed))
        return cls(
            data_version=str(repro.data_version),
            feature_version=str(repro.feature_version),
            label_version=str(repro.label_version),
            model_version=str(repro.model_version),
            risk_version=str(repro.risk_version),
            portfolio_version=str(repro.portfolio_version),
            execution_version=str(repro.execution_version),
            code_version=str(repro.code_version),
            seed=int(seed if seed is not None else repro.seed),
        )


@dataclass
class ExperimentRecord:
    """Auditable experiment entry (rejected runs are retained)."""

    experiment_id: str
    name: str = "backtest"
    state: str = BacktestState.CREATED.value
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    lineage: ExperimentLineage = field(default_factory=ExperimentLineage)
    config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    invalidated: bool = False
    invalidation_reason: str = ""
    tags: dict[str, Any] = field(default_factory=dict)
    result_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        d = asdict(self)
        d["lineage"] = self.lineage.to_dict()
        return to_jsonable(d)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentRecord":
        payload = dict(data)
        lineage = ExperimentLineage.from_dict(payload.pop("lineage", None))
        return cls(lineage=lineage, **{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})


class ExperimentRegistry:
    """In-memory experiment registry with optional JSON persistence."""

    def __init__(self) -> None:
        self._records: dict[str, ExperimentRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    def create(
        self,
        *,
        name: str = "backtest",
        lineage: ExperimentLineage | Mapping[str, Any] | None = None,
        config: Mapping[str, Any] | None = None,
        experiment_id: str | None = None,
        tags: Mapping[str, Any] | None = None,
    ) -> ExperimentRecord:
        eid = experiment_id or str(uuid.uuid4())
        lin = (
            lineage
            if isinstance(lineage, ExperimentLineage)
            else ExperimentLineage.from_dict(lineage)
        )
        rec = ExperimentRecord(
            experiment_id=eid,
            name=name,
            state=BacktestState.CREATED.value,
            lineage=lin,
            config=dict(config or {}),
            tags=dict(tags or {}),
        )
        self._records[eid] = rec
        return rec

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        return self._records.get(experiment_id)

    def require(self, experiment_id: str) -> ExperimentRecord:
        rec = self.get(experiment_id)
        if rec is None:
            raise KeyError(f"unknown experiment_id: {experiment_id}")
        return rec

    def update_state(self, experiment_id: str, state: BacktestState | str) -> ExperimentRecord:
        rec = self.require(experiment_id)
        rec.state = state.value if isinstance(state, BacktestState) else str(state)
        rec.updated_at = datetime.now(UTC).isoformat()
        return rec

    def register_result(
        self,
        experiment_id: str,
        *,
        state: BacktestState | str,
        metrics: Mapping[str, Any] | None = None,
        warnings: list[str] | None = None,
        result_summary: Mapping[str, Any] | None = None,
        invalidated: bool = False,
        invalidation_reason: str = "",
    ) -> ExperimentRecord:
        rec = self.require(experiment_id)
        rec.state = state.value if isinstance(state, BacktestState) else str(state)
        if metrics is not None:
            rec.metrics = dict(metrics)
        if warnings is not None:
            rec.warnings = list(warnings)
        if result_summary is not None:
            rec.result_summary = dict(result_summary)
        rec.invalidated = bool(invalidated)
        if invalidation_reason:
            rec.invalidation_reason = str(invalidation_reason)
        rec.updated_at = datetime.now(UTC).isoformat()
        return rec

    def invalidate(self, experiment_id: str, reason: str) -> ExperimentRecord:
        rec = self.require(experiment_id)
        rec.state = BacktestState.INVALIDATED.value
        rec.invalidated = True
        rec.invalidation_reason = str(reason)
        rec.warnings = list(rec.warnings) + [f"INVALIDATED: {reason}"]
        rec.updated_at = datetime.now(UTC).isoformat()
        return rec

    def list(
        self,
        *,
        state: str | None = None,
        include_invalidated: bool = True,
    ) -> list[ExperimentRecord]:
        out = list(self._records.values())
        if state is not None:
            out = [r for r in out if r.state == state]
        if not include_invalidated:
            out = [r for r in out if not r.invalidated]
        return out

    def save(self, path: str) -> str:
        payload = {"experiments": [r.to_dict() for r in self._records.values()]}
        save_json(path, payload)
        return str(path)

    def load(self, path: str) -> int:
        data = load_json(path)
        experiments = data.get("experiments", data if isinstance(data, list) else [])
        n = 0
        for item in experiments:
            rec = ExperimentRecord.from_dict(item)
            self._records[rec.experiment_id] = rec
            n += 1
        return n

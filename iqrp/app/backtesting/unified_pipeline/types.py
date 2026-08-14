"""Unified Alpha → Risk → Portfolio → Execution orchestration types.

Wiring/integration only. Not a profitability claim. Research evidence is not a guarantee.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class StageOutcome(str, Enum):
    SIGNAL_REJECTED = "SIGNAL_REJECTED"
    CANDIDATE_REJECTED = "CANDIDATE_REJECTED"
    RISK_APPROVED = "RISK_APPROVED"
    RISK_REDUCED = "RISK_REDUCED"
    RISK_REJECTED = "RISK_REJECTED"
    PORTFOLIO_APPROVED = "PORTFOLIO_APPROVED"
    PORTFOLIO_REDUCED = "PORTFOLIO_REDUCED"
    PORTFOLIO_REJECTED = "PORTFOLIO_REJECTED"
    ORDER_CREATED = "ORDER_CREATED"
    ORDER_REJECTED = "ORDER_REJECTED"
    FILL_PARTIAL = "FILL_PARTIAL"
    FILL_COMPLETE = "FILL_COMPLETE"
    RECONCILIATION_FAILED = "RECONCILIATION_FAILED"
    RECONCILIATION_OK = "RECONCILIATION_OK"
    SKIPPED_FLAT = "SKIPPED_FLAT"


class CandidateRejectionCode(str, Enum):
    NON_FINITE_SIGNAL = "NON_FINITE_SIGNAL"
    INVALID_TIMESTAMP = "INVALID_TIMESTAMP"
    MISSING_INSTRUMENT = "MISSING_INSTRUMENT"
    INVALID_DIRECTION = "INVALID_DIRECTION"
    UNKNOWN_MODEL_VERSION = "UNKNOWN_MODEL_VERSION"
    UNKNOWN_DATASET = "UNKNOWN_DATASET"
    OOS_UNACCEPTABLE = "OOS_UNACCEPTABLE"
    FUTURE_INFORMATION = "FUTURE_INFORMATION"
    STALE_CANDIDATE = "STALE_CANDIDATE"
    DUPLICATE_CANDIDATE = "DUPLICATE_CANDIDATE"
    MISSING_CANDIDATE_ID = "MISSING_CANDIDATE_ID"


@dataclass(frozen=True, slots=True)
class AlphaCandidate:
    """Immutable handoff object from Alpha Research to the trading cascade."""

    candidate_id: str
    signal_id: str
    instrument: str
    timestamp: str
    direction: float  # +1 LONG, -1 SHORT, 0 FLAT
    signal_value: float
    confidence: float | None = None
    expected_horizon: int | None = None
    signal_timeframe: str = ""
    execution_timeframe: str = ""
    source_model: str = ""
    source_model_version: str = ""
    research_configuration: dict[str, Any] = field(default_factory=dict)
    data_version: str = ""
    dataset_checksum: str = ""
    oos_status: str = "UNKNOWN"
    cost_model_id: str = "default_bps"
    experiment_id: str = ""
    requested_weight: float | None = None  # abs exposure in weight space if set
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "signal_id": self.signal_id,
            "instrument": self.instrument,
            "timestamp": self.timestamp,
            "direction": float(self.direction),
            "signal_value": float(self.signal_value),
            "confidence": self.confidence,
            "expected_horizon": self.expected_horizon,
            "signal_timeframe": self.signal_timeframe,
            "execution_timeframe": self.execution_timeframe,
            "source_model": self.source_model,
            "source_model_version": self.source_model_version,
            "research_configuration": dict(self.research_configuration),
            "data_version": self.data_version,
            "dataset_checksum": self.dataset_checksum,
            "oos_status": self.oos_status,
            "cost_model_id": self.cost_model_id,
            "experiment_id": self.experiment_id,
            "requested_weight": self.requested_weight,
            "meta": dict(self.meta),
            "disclaimer": "ALPHA CANDIDATE — research wiring only, not a profitability claim.",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlphaCandidate:
        return cls(
            candidate_id=str(data["candidate_id"]),
            signal_id=str(data["signal_id"]),
            instrument=str(data["instrument"]),
            timestamp=str(data["timestamp"]),
            direction=float(data["direction"]),
            signal_value=float(data["signal_value"]),
            confidence=(None if data.get("confidence") is None else float(data["confidence"])),
            expected_horizon=(
                None if data.get("expected_horizon") is None else int(data["expected_horizon"])
            ),
            signal_timeframe=str(data.get("signal_timeframe", "")),
            execution_timeframe=str(data.get("execution_timeframe", "")),
            source_model=str(data.get("source_model", "")),
            source_model_version=str(data.get("source_model_version", "")),
            research_configuration=dict(data.get("research_configuration") or {}),
            data_version=str(data.get("data_version", "")),
            dataset_checksum=str(data.get("dataset_checksum", "")),
            oos_status=str(data.get("oos_status", "UNKNOWN")),
            cost_model_id=str(data.get("cost_model_id", "default_bps")),
            experiment_id=str(data.get("experiment_id", "")),
            requested_weight=(
                None if data.get("requested_weight") is None else float(data["requested_weight"])
            ),
            meta=dict(data.get("meta") or {}),
        )


@dataclass(slots=True)
class RiskHandoffResult:
    risk_decision_id: str
    outcome: StageOutcome
    requested_exposure: float
    approved_exposure: float
    reason: str
    reason_codes: list[str] = field(default_factory=list)
    limits_triggered: list[dict[str, Any]] = field(default_factory=list)
    recommended_size: float | None = None
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_decision_id": self.risk_decision_id,
            "outcome": self.outcome.value,
            "requested_exposure": self.requested_exposure,
            "approved_exposure": self.approved_exposure,
            "reason": self.reason,
            "reason_codes": list(self.reason_codes),
            "limits_triggered": list(self.limits_triggered),
            "recommended_size": self.recommended_size,
            "audit": dict(self.audit),
        }


@dataclass(slots=True)
class SizingResult:
    requested_size: float
    risk_adjusted_size: float
    final_size: float
    sizing_method: str
    sizing_configuration: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PortfolioHandoffResult:
    portfolio_decision_id: str
    outcome: StageOutcome
    target_position_weight: float
    current_position_weight: float
    delta_weight: float
    reason: str = ""
    constraint_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio_decision_id": self.portfolio_decision_id,
            "outcome": self.outcome.value,
            "target_position_weight": self.target_position_weight,
            "current_position_weight": self.current_position_weight,
            "delta_weight": self.delta_weight,
            "reason": self.reason,
            "constraint_reasons": list(self.constraint_reasons),
        }


@dataclass(slots=True)
class LineageRecord:
    """End-to-end audit lineage for a trade/order."""

    candidate_id: str
    signal_id: str
    model_id: str
    model_version: str
    dataset_id: str
    dataset_checksum: str
    risk_decision_id: str
    portfolio_decision_id: str
    order_id: str
    fill_ids: list[str] = field(default_factory=list)
    trade_ids: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "signal_id": self.signal_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "dataset_id": self.dataset_id,
            "dataset_checksum": self.dataset_checksum,
            "risk_decision_id": self.risk_decision_id,
            "portfolio_decision_id": self.portfolio_decision_id,
            "order_id": self.order_id,
            "fill_ids": list(self.fill_ids),
            "trade_ids": list(self.trade_ids),
            "extra": dict(self.extra),
        }


__all__ = [
    "AlphaCandidate",
    "CandidateRejectionCode",
    "LineageRecord",
    "PortfolioHandoffResult",
    "RiskHandoffResult",
    "SizingResult",
    "StageOutcome",
]

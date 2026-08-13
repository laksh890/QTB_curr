"""Types for the Risk Intelligence Ensemble.

Preserves RiskState from the institutional risk base. DecisionAction is ensemble-specific.
Hard limits cannot be overridden by forecast confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, ClassVar

from iqrp.app.risk.base import RiskState


class DecisionAction(str, Enum):
    APPROVE = "APPROVE"
    APPROVE_REDUCED = "APPROVE_REDUCED"
    REJECT = "REJECT"
    HALT = "HALT"


@dataclass(slots=True)
class NormalizedMetric:
    """Metric mapped to [0, 1] risk score while preserving the original observation."""

    name: str
    original_value: float
    normalized_value: float
    method: str
    reference: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    model_version: str = "1.0.0"
    unit: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "original_value": self.original_value,
            "normalized_value": self.normalized_value,
            "method": self.method,
            "reference": dict(self.reference),
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "unit": self.unit,
            "metadata": dict(self.metadata),
        }


RISK_DIMENSIONS: tuple[str, ...] = (
    "market",
    "tail",
    "liquidity",
    "concentration",
    "correlation",
    "drawdown",
    "model",
    "operational",
)


@dataclass(slots=True)
class RiskScore:
    """Per-dimension risk scores in [0, 1] where 1 = maximum risk. Identity preserved."""

    market: float = 0.0
    tail: float = 0.0
    liquidity: float = 0.0
    concentration: float = 0.0
    correlation: float = 0.0
    drawdown: float = 0.0
    model: float = 0.0
    operational: float = 0.0
    overall: float = 0.0
    weights_applied: dict[str, float] = field(default_factory=dict)
    contributors: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    DIMENSIONS: ClassVar[tuple[str, ...]] = RISK_DIMENSIONS

    def dimension_map(self) -> dict[str, float]:
        return {
            "market": float(self.market),
            "tail": float(self.tail),
            "liquidity": float(self.liquidity),
            "concentration": float(self.concentration),
            "correlation": float(self.correlation),
            "drawdown": float(self.drawdown),
            "model": float(self.model),
            "operational": float(self.operational),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.dimension_map(),
            "overall": float(self.overall),
            "weights_applied": dict(self.weights_applied),
            "contributors": dict(self.contributors),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RiskScore:
        return cls(
            market=float(data.get("market", 0.0)),
            tail=float(data.get("tail", 0.0)),
            liquidity=float(data.get("liquidity", 0.0)),
            concentration=float(data.get("concentration", 0.0)),
            correlation=float(data.get("correlation", 0.0)),
            drawdown=float(data.get("drawdown", 0.0)),
            model=float(data.get("model", 0.0)),
            operational=float(data.get("operational", 0.0)),
            overall=float(data.get("overall", 0.0)),
            weights_applied=dict(data.get("weights_applied") or {}),
            contributors=dict(data.get("contributors") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class RiskAssessment:
    """Unified ensemble assessment retaining raw inputs and per-dimension identity."""

    timestamp: str
    data_version: str
    risk_model_versions: dict[str, str]
    input_metrics: dict[str, Any]
    normalized_metrics: dict[str, NormalizedMetric]
    dimension_scores: RiskScore
    overall_score: float
    confidence: float
    disagreement: dict[str, Any]
    risk_state: RiskState
    budget_recommendation: dict[str, Any]
    max_exposure: float
    recommended_leverage: float
    reasons: list[str] = field(default_factory=list)
    missing_critical: list[str] = field(default_factory=list)
    fallback_applied: bool = False
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "data_version": self.data_version,
            "risk_model_versions": dict(self.risk_model_versions),
            "input_metrics": dict(self.input_metrics),
            "normalized_metrics": {k: v.to_dict() for k, v in self.normalized_metrics.items()},
            "dimension_scores": self.dimension_scores.to_dict(),
            "overall_score": float(self.overall_score),
            "confidence": float(self.confidence),
            "disagreement": dict(self.disagreement),
            "risk_state": self.risk_state.value,
            "budget_recommendation": dict(self.budget_recommendation),
            "max_exposure": float(self.max_exposure),
            "recommended_leverage": float(self.recommended_leverage),
            "reasons": list(self.reasons),
            "missing_critical": list(self.missing_critical),
            "fallback_applied": bool(self.fallback_applied),
            "audit": dict(self.audit),
        }


@dataclass(slots=True)
class EnsembleDecision:
    """Pre-trade / exposure decision produced by the ensemble gate."""

    decision: DecisionAction
    risk_state: RiskState
    risk_score: RiskScore
    risk_confidence: float
    triggered_limits: list[str]
    reasons: list[str]
    required_position_reduction: float
    maximum_permitted_exposure: float
    recommended_leverage: float
    timestamp: str
    data_version: str
    model_versions: dict[str, str]
    audit: dict[str, Any] = field(default_factory=dict)
    proposed_exposure: float = 0.0
    forecast_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "risk_state": self.risk_state.value,
            "risk_score": self.risk_score.to_dict(),
            "risk_confidence": float(self.risk_confidence),
            "triggered_limits": list(self.triggered_limits),
            "reasons": list(self.reasons),
            "required_position_reduction": float(self.required_position_reduction),
            "maximum_permitted_exposure": float(self.maximum_permitted_exposure),
            "recommended_leverage": float(self.recommended_leverage),
            "timestamp": self.timestamp,
            "data_version": self.data_version,
            "model_versions": dict(self.model_versions),
            "audit": dict(self.audit),
            "proposed_exposure": float(self.proposed_exposure),
            "forecast_confidence": float(self.forecast_confidence),
        }


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()

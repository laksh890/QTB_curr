"""Capital allocation result types with full audit trail."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class RiskBudget:
    """Risk budget at a hierarchical scope for a risk type."""

    name: str
    scope: str  # portfolio|strategy|asset|sector|factor|market|account
    risk_type: str  # volatility|var|cvar|liquidity|concentration|drawdown|factor|tail
    budget: float
    used: float = 0.0
    timestamp: str = field(default_factory=_utc_now)
    data_version: str = "1.0.0"
    model_version: str = "1.0.0"
    inputs: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    reasons: list[str] = field(default_factory=list)

    def remaining(self) -> float:
        return max(float(self.budget) - float(self.used), 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scope": self.scope,
            "risk_type": self.risk_type,
            "budget": float(self.budget),
            "used": float(self.used),
            "remaining": self.remaining(),
            "timestamp": self.timestamp,
            "data_version": self.data_version,
            "model_version": self.model_version,
            "inputs": dict(self.inputs),
            "params": dict(self.params),
            "output": dict(self.output),
            "confidence": float(self.confidence),
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RiskBudget:
        return cls(
            name=str(data.get("name", "")),
            scope=str(data.get("scope", "portfolio")),
            risk_type=str(data.get("risk_type", "volatility")),
            budget=float(data.get("budget", 0.0)),
            used=float(data.get("used", 0.0)),
            timestamp=str(data.get("timestamp", _utc_now())),
            data_version=str(data.get("data_version", "1.0.0")),
            model_version=str(data.get("model_version", "1.0.0")),
            inputs=dict(data.get("inputs") or {}),
            params=dict(data.get("params") or {}),
            output=dict(data.get("output") or {}),
            confidence=float(data.get("confidence", 1.0)),
            reasons=list(data.get("reasons") or []),
        )


@dataclass(slots=True)
class StrategyAllocation:
    """Per-strategy capital and risk constraints after allocation."""

    name: str
    capital_budget: float
    risk_budget: float
    weight: float
    max_gross: float
    max_net: float
    max_position: float
    max_leverage: float
    max_turnover: float
    max_participation: float
    capacity_scale: float = 1.0
    correlation_scale: float = 1.0
    drawdown_scale: float = 1.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "capital_budget": float(self.capital_budget),
            "risk_budget": float(self.risk_budget),
            "weight": float(self.weight),
            "max_gross": float(self.max_gross),
            "max_net": float(self.max_net),
            "max_position": float(self.max_position),
            "max_leverage": float(self.max_leverage),
            "max_turnover": float(self.max_turnover),
            "max_participation": float(self.max_participation),
            "capacity_scale": float(self.capacity_scale),
            "correlation_scale": float(self.correlation_scale),
            "drawdown_scale": float(self.drawdown_scale),
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyAllocation:
        return cls(
            name=str(data.get("name", "")),
            capital_budget=float(data.get("capital_budget", 0.0)),
            risk_budget=float(data.get("risk_budget", 0.0)),
            weight=float(data.get("weight", 0.0)),
            max_gross=float(data.get("max_gross", 1.0)),
            max_net=float(data.get("max_net", 1.0)),
            max_position=float(data.get("max_position", 0.1)),
            max_leverage=float(data.get("max_leverage", 1.0)),
            max_turnover=float(data.get("max_turnover", 0.5)),
            max_participation=float(data.get("max_participation", 0.1)),
            capacity_scale=float(data.get("capacity_scale", 1.0)),
            correlation_scale=float(data.get("correlation_scale", 1.0)),
            drawdown_scale=float(data.get("drawdown_scale", 1.0)),
            reasons=list(data.get("reasons") or []),
        )


@dataclass(slots=True)
class CapitalAllocation:
    """Full capital allocation result with audit trail."""

    timestamp: str = field(default_factory=_utc_now)
    data_version: str = "1.0.0"
    model_version: str = "1.0.0"
    method: str = "risk_parity"
    names: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    capital_amounts: dict[str, float] = field(default_factory=dict)
    risk_budgets_used: dict[str, float] = field(default_factory=dict)
    strategies: dict[str, StrategyAllocation] = field(default_factory=dict)
    risk_budgets: list[RiskBudget] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    constraints_applied: list[str] = field(default_factory=list)
    correlation_adjustment: dict[str, float] = field(default_factory=dict)
    capacity_adjustment: dict[str, float] = field(default_factory=dict)
    drawdown_adjustment: dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "data_version": self.data_version,
            "model_version": self.model_version,
            "method": self.method,
            "names": list(self.names),
            "weights": {k: float(v) for k, v in self.weights.items()},
            "capital_amounts": {k: float(v) for k, v in self.capital_amounts.items()},
            "risk_budgets_used": {k: float(v) for k, v in self.risk_budgets_used.items()},
            "strategies": {k: v.to_dict() for k, v in self.strategies.items()},
            "risk_budgets": [b.to_dict() for b in self.risk_budgets],
            "inputs": dict(self.inputs),
            "params": dict(self.params),
            "output": dict(self.output),
            "constraints_applied": list(self.constraints_applied),
            "correlation_adjustment": {k: float(v) for k, v in self.correlation_adjustment.items()},
            "capacity_adjustment": {k: float(v) for k, v in self.capacity_adjustment.items()},
            "drawdown_adjustment": {k: float(v) for k, v in self.drawdown_adjustment.items()},
            "confidence": float(self.confidence),
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapitalAllocation:
        strategies_raw = data.get("strategies") or {}
        strategies = {
            str(k): StrategyAllocation.from_dict(v) if isinstance(v, dict) else v
            for k, v in strategies_raw.items()
        }
        budgets_raw = data.get("risk_budgets") or []
        budgets = [RiskBudget.from_dict(b) if isinstance(b, dict) else b for b in budgets_raw]
        return cls(
            timestamp=str(data.get("timestamp", _utc_now())),
            data_version=str(data.get("data_version", "1.0.0")),
            model_version=str(data.get("model_version", "1.0.0")),
            method=str(data.get("method", "risk_parity")),
            names=list(data.get("names") or []),
            weights={str(k): float(v) for k, v in (data.get("weights") or {}).items()},
            capital_amounts={
                str(k): float(v) for k, v in (data.get("capital_amounts") or {}).items()
            },
            risk_budgets_used={
                str(k): float(v) for k, v in (data.get("risk_budgets_used") or {}).items()
            },
            strategies=strategies,
            risk_budgets=budgets,
            inputs=dict(data.get("inputs") or {}),
            params=dict(data.get("params") or {}),
            output=dict(data.get("output") or {}),
            constraints_applied=list(data.get("constraints_applied") or []),
            correlation_adjustment={
                str(k): float(v) for k, v in (data.get("correlation_adjustment") or {}).items()
            },
            capacity_adjustment={
                str(k): float(v) for k, v in (data.get("capacity_adjustment") or {}).items()
            },
            drawdown_adjustment={
                str(k): float(v) for k, v in (data.get("drawdown_adjustment") or {}).items()
            },
            confidence=float(data.get("confidence", 1.0)),
            reasons=list(data.get("reasons") or []),
        )

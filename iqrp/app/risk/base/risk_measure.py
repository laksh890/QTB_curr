"""Shared types for Institutional Risk Intelligence.

Risk never generates alpha. Hard limits cannot be overridden by forecast confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class RiskState(str, Enum):
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    REDUCED_RISK = "REDUCED_RISK"
    CAPITAL_PRESERVATION = "CAPITAL_PRESERVATION"
    TRADING_HALT = "TRADING_HALT"


class LimitSeverity(str, Enum):
    WARNING = "WARNING"
    SOFT = "SOFT"
    HARD = "HARD"


@dataclass(slots=True)
class RiskMeasure:
    name: str
    value: float
    unit: str = ""
    confidence: float | None = None
    horizon: int | None = None
    method: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": float(self.value) if np.isfinite(self.value) else None,
            "unit": self.unit,
            "confidence": self.confidence,
            "horizon": self.horizon,
            "method": self.method,
            "parameters": dict(self.parameters),
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class LimitBreach:
    limit_name: str
    severity: LimitSeverity
    observed: float
    threshold: float
    reason: str
    scope: str = "portfolio"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "limit_name": self.limit_name,
            "severity": self.severity.value,
            "observed": self.observed,
            "threshold": self.threshold,
            "reason": self.reason,
            "scope": self.scope,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class RiskDecision:
    approved: bool
    reason: str
    risk_state: RiskState
    breaches: list[LimitBreach] = field(default_factory=list)
    recommended_size: float | None = None
    recommended_leverage: float | None = None
    audit: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "reason": self.reason,
            "risk_state": self.risk_state.value,
            "breaches": [b.to_dict() for b in self.breaches],
            "recommended_size": self.recommended_size,
            "recommended_leverage": self.recommended_leverage,
            "audit": dict(self.audit),
        }


@dataclass(slots=True)
class RiskReport:
    portfolio_risk: dict[str, Any]
    position_risk: dict[str, Any]
    tail_risk: dict[str, Any]
    liquidity_risk: dict[str, Any]
    concentration: dict[str, Any]
    factor_exposure: dict[str, Any]
    drawdown: dict[str, Any]
    stress: dict[str, Any]
    limits: dict[str, Any]
    breaches: list[dict[str, Any]]
    risk_state: RiskState
    timestamp: Any = None
    data_version: str = "1.0.0"
    model_version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "portfolio_risk": dict(self.portfolio_risk),
            "position_risk": dict(self.position_risk),
            "tail_risk": dict(self.tail_risk),
            "liquidity_risk": dict(self.liquidity_risk),
            "concentration": dict(self.concentration),
            "factor_exposure": dict(self.factor_exposure),
            "drawdown": dict(self.drawdown),
            "stress": dict(self.stress),
            "limits": dict(self.limits),
            "breaches": list(self.breaches),
            "risk_state": self.risk_state.value,
            "timestamp": self.timestamp,
            "data_version": self.data_version,
            "model_version": self.model_version,
            "metadata": dict(self.metadata),
        }


def as_returns(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def as_weights(w: Any, n: int | None = None) -> np.ndarray:
    arr = np.asarray(w, dtype=np.float64).reshape(-1)
    if n is not None and arr.size != n:
        if arr.size == 1:
            arr = np.full(n, float(arr[0]) / max(n, 1))
        else:
            out = np.zeros(n, dtype=np.float64)
            m = min(n, arr.size)
            out[:m] = arr[:m]
            arr = out
    return arr

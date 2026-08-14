"""Portfolio optimization objective specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ObjectiveType(str, Enum):
    MIN_VARIANCE = "min_variance"
    MEAN_VARIANCE = "mean_variance"
    MAX_SHARPE = "max_sharpe"
    MAX_DIVERSIFICATION = "max_diversification"
    RISK_PARITY = "risk_parity"
    ERC = "erc"
    HRP = "hrp"
    MIN_CVAR = "min_cvar"
    DRAWDOWN = "drawdown"
    TURNOVER_AWARE = "turnover_aware"
    ROBUST = "robust"
    RISK_BUDGET = "risk_budget"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass(slots=True)
class ObjectiveSpec:
    """Configurable objective for portfolio optimization."""

    objective_type: ObjectiveType | str = ObjectiveType.MEAN_VARIANCE
    risk_aversion: float = 1.0
    target_return: float | None = None
    target_volatility: float | None = None
    risk_free_rate: float = 0.0
    turnover_penalty: float = 0.0
    transaction_cost_penalty: float = 0.0
    cvar_confidence: float = 0.95
    drawdown_penalty: float = 0.0
    robust_kappa: float = 0.0
    risk_budgets: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)  # multi-objective component weights
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.objective_type, str):
            self.objective_type = ObjectiveType(self.objective_type)

    @property
    def name(self) -> str:
        ot = self.objective_type
        return ot.value if isinstance(ot, ObjectiveType) else str(ot)

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective_type": self.name,
            "risk_aversion": float(self.risk_aversion),
            "target_return": float(self.target_return) if self.target_return is not None else None,
            "target_volatility": (
                float(self.target_volatility) if self.target_volatility is not None else None
            ),
            "risk_free_rate": float(self.risk_free_rate),
            "turnover_penalty": float(self.turnover_penalty),
            "transaction_cost_penalty": float(self.transaction_cost_penalty),
            "cvar_confidence": float(self.cvar_confidence),
            "drawdown_penalty": float(self.drawdown_penalty),
            "robust_kappa": float(self.robust_kappa),
            "risk_budgets": {str(k): float(v) for k, v in self.risk_budgets.items()},
            "weights": {str(k): float(v) for k, v in self.weights.items()},
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectiveSpec:
        return cls(
            objective_type=data.get("objective_type", ObjectiveType.MEAN_VARIANCE),
            risk_aversion=float(data.get("risk_aversion", 1.0)),
            target_return=(
                float(data["target_return"]) if data.get("target_return") is not None else None
            ),
            target_volatility=(
                float(data["target_volatility"])
                if data.get("target_volatility") is not None
                else None
            ),
            risk_free_rate=float(data.get("risk_free_rate", 0.0)),
            turnover_penalty=float(data.get("turnover_penalty", 0.0)),
            transaction_cost_penalty=float(data.get("transaction_cost_penalty", 0.0)),
            cvar_confidence=float(data.get("cvar_confidence", 0.95)),
            drawdown_penalty=float(data.get("drawdown_penalty", 0.0)),
            robust_kappa=float(data.get("robust_kappa", 0.0)),
            risk_budgets={str(k): float(v) for k, v in (data.get("risk_budgets") or {}).items()},
            weights={str(k): float(v) for k, v in (data.get("weights") or {}).items()},
            params=dict(data.get("params") or {}),
        )

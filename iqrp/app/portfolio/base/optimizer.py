"""Abstract portfolio optimizer interface and result types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

import numpy as np

from iqrp.app.portfolio.base.constraints import ConstraintSet, ConstraintViolation, conflicting_constraints
from iqrp.app.portfolio.base.objective import ObjectiveSpec
from iqrp.app.portfolio.base.portfolio import Portfolio


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class OptimizationFailureError(Exception):
    """Raised when optimization is infeasible and no silent constraint relaxation is allowed."""

    def __init__(
        self,
        message: str,
        *,
        conflicting: Sequence[str] | None = None,
        diagnostics: dict[str, Any] | None = None,
        result: OptimizationResult | None = None,
    ) -> None:
        super().__init__(message)
        self.conflicting = list(conflicting or [])
        self.diagnostics = dict(diagnostics or {})
        self.result = result


@dataclass(slots=True)
class OptimizationResult:
    """Structured optimizer output with audit and failure metadata."""

    success: bool
    weights: list[float] = field(default_factory=list)
    names: list[str] = field(default_factory=list)
    status: str = "ok"
    failure_reason: str | None = None
    conflicting_constraints: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    fallback_used: bool = False
    fallback_kind: str | None = None
    objective_value: float | None = None
    expected_return: float | None = None
    expected_variance: float | None = None
    violations: list[ConstraintViolation] = field(default_factory=list)
    objective: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_utc_now)
    data_version: str = "1.0.0"
    model_version: str = "1.0.0"
    method: str = ""
    seed: int | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)

    def weight_array(self) -> np.ndarray:
        return np.asarray(self.weights, dtype=np.float64)

    def to_portfolio(
        self,
        *,
        currency: str = "USD",
        portfolio_type: str = "long_only",
        cash: float = 0.0,
    ) -> Portfolio:
        return Portfolio(
            names=list(self.names),
            weights=[float(w) for w in self.weights],
            cash=float(cash),
            currency=currency,
            portfolio_type=portfolio_type,
            meta={
                "optimization_status": self.status,
                "method": self.method,
                "fallback_used": self.fallback_used,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": bool(self.success),
            "weights": [float(w) for w in self.weights],
            "names": list(self.names),
            "status": self.status,
            "failure_reason": self.failure_reason,
            "conflicting_constraints": list(self.conflicting_constraints),
            "diagnostics": dict(self.diagnostics),
            "fallback_used": bool(self.fallback_used),
            "fallback_kind": self.fallback_kind,
            "objective_value": float(self.objective_value) if self.objective_value is not None else None,
            "expected_return": float(self.expected_return) if self.expected_return is not None else None,
            "expected_variance": float(self.expected_variance) if self.expected_variance is not None else None,
            "violations": [v.to_dict() for v in self.violations],
            "objective": dict(self.objective),
            "constraints": dict(self.constraints),
            "timestamp": self.timestamp,
            "data_version": self.data_version,
            "model_version": self.model_version,
            "method": self.method,
            "seed": self.seed,
            "inputs": dict(self.inputs),
            "params": dict(self.params),
            "audit": dict(self.audit),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimizationResult:
        return cls(
            success=bool(data.get("success", False)),
            weights=[float(w) for w in (data.get("weights") or [])],
            names=list(data.get("names") or []),
            status=str(data.get("status", "unknown")),
            failure_reason=data.get("failure_reason"),
            conflicting_constraints=list(data.get("conflicting_constraints") or []),
            diagnostics=dict(data.get("diagnostics") or {}),
            fallback_used=bool(data.get("fallback_used", False)),
            fallback_kind=data.get("fallback_kind"),
            objective_value=float(data["objective_value"]) if data.get("objective_value") is not None else None,
            expected_return=float(data["expected_return"]) if data.get("expected_return") is not None else None,
            expected_variance=(
                float(data["expected_variance"]) if data.get("expected_variance") is not None else None
            ),
            violations=[ConstraintViolation.from_dict(v) for v in (data.get("violations") or [])],
            objective=dict(data.get("objective") or {}),
            constraints=dict(data.get("constraints") or {}),
            timestamp=str(data.get("timestamp", _utc_now())),
            data_version=str(data.get("data_version", "1.0.0")),
            model_version=str(data.get("model_version", "1.0.0")),
            method=str(data.get("method", "")),
            seed=int(data["seed"]) if data.get("seed") is not None else None,
            inputs=dict(data.get("inputs") or {}),
            params=dict(data.get("params") or {}),
            audit=dict(data.get("audit") or {}),
        )

    @classmethod
    def failure(
        cls,
        *,
        reason: str,
        names: Sequence[str] | None = None,
        weights: Sequence[float] | None = None,
        violations: Sequence[ConstraintViolation] | None = None,
        diagnostics: dict[str, Any] | None = None,
        method: str = "",
        data_version: str = "1.0.0",
        model_version: str = "1.0.0",
        fallback_used: bool = False,
        fallback_kind: str | None = None,
        success: bool = False,
    ) -> OptimizationResult:
        viols = list(violations or [])
        return cls(
            success=success,
            weights=[float(w) for w in (weights or [])],
            names=list(names or []),
            status="failed" if not success else "fallback",
            failure_reason=reason,
            conflicting_constraints=conflicting_constraints(viols),
            diagnostics=dict(diagnostics or {}),
            fallback_used=fallback_used,
            fallback_kind=fallback_kind,
            violations=viols,
            method=method,
            data_version=data_version,
            model_version=model_version,
            audit={"failure_reason": reason},
        )


class PortfolioOptimizer(ABC):
    """Abstract base for portfolio optimizers."""

    name: str = "portfolio_optimizer"

    def __init__(
        self,
        *,
        objective: ObjectiveSpec | None = None,
        constraints: ConstraintSet | None = None,
        data_version: str = "1.0.0",
        model_version: str = "1.0.0",
        seed: int | None = None,
    ) -> None:
        self.objective = objective or ObjectiveSpec()
        self.constraints = constraints or ConstraintSet()
        self.data_version = data_version
        self.model_version = model_version
        self.seed = seed

    @abstractmethod
    def optimize(
        self,
        *,
        mu: Sequence[float] | np.ndarray | None = None,
        cov: Sequence[Sequence[float]] | np.ndarray | None = None,
        names: Sequence[str] | None = None,
        current_weights: Sequence[float] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Solve for target weights under objective and constraints."""

    def validate_weights(
        self,
        weights: Sequence[float] | np.ndarray,
        *,
        names: Sequence[str] | None = None,
        current_weights: Sequence[float] | np.ndarray | None = None,
    ) -> list[ConstraintViolation]:
        return self.constraints.evaluate(weights, names=names, current_weights=current_weights)

    def raise_on_hard_violations(self, result: OptimizationResult) -> OptimizationResult:
        """Do not silently relax hard constraints; raise or return structured failure."""
        hard = [v for v in result.violations if v.hard]
        if hard and not result.fallback_used:
            raise OptimizationFailureError(
                result.failure_reason or "Hard constraint violations",
                conflicting=conflicting_constraints(hard),
                diagnostics=result.diagnostics,
                result=result,
            )
        return result

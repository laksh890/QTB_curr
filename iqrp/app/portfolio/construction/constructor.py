"""High-level PortfolioConstructor: signals/forecasts → constrained portfolio."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from iqrp.app.portfolio.base.optimizer import OptimizationResult
from iqrp.app.portfolio.base.portfolio import Portfolio
from iqrp.app.portfolio.base.position import Position
from iqrp.app.portfolio.config import PortfolioSettings
from iqrp.app.portfolio.construction.signal_to_weight import signals_to_raw_weights
from iqrp.app.portfolio.construction.target_positions import TargetPositions, weights_to_positions
from iqrp.app.portfolio.construction.target_weights import TargetWeights, build_target_weights


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class PortfolioResult:
    """Full construction output with risk / cost / audit fields."""

    portfolio_weights: TargetWeights | None = None
    target_positions: TargetPositions | list[Position] | None = None
    expected_return: float | None = None
    expected_volatility: float | None = None
    expected_sharpe: float | None = None
    expected_cvar: float | None = None
    expected_drawdown: float | None = None
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    turnover: float = 0.0
    transaction_cost: dict[str, Any] = field(default_factory=dict)
    risk_contribution: dict[str, Any] = field(default_factory=dict)
    factor_exposure: dict[str, Any] = field(default_factory=dict)
    liquidity_exposure: dict[str, Any] = field(default_factory=dict)
    optimization: OptimizationResult | None = None
    fallback_used: bool = False
    fallback_kind: str | None = None
    fallback_reasons: list[str] = field(default_factory=list)
    risk_validation: dict[str, Any] | None = None
    method: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_utc_now)
    data_version: str = "1.0.0"
    model_version: str = "1.0.0"
    seed: int | None = None
    audit: dict[str, Any] = field(default_factory=dict)
    names: list[str] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    success: bool = True
    status: str = "ok"
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        pos = self.target_positions
        if isinstance(pos, TargetPositions):
            pos_payload: Any = pos.to_dict()
        elif isinstance(pos, list):
            pos_payload = [p.to_dict() if hasattr(p, "to_dict") else p for p in pos]
        else:
            pos_payload = None
        return {
            "portfolio_weights": (
                self.portfolio_weights.to_dict() if self.portfolio_weights else None
            ),
            "target_positions": pos_payload,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "expected_sharpe": self.expected_sharpe,
            "expected_cvar": self.expected_cvar,
            "expected_drawdown": self.expected_drawdown,
            "gross_exposure": float(self.gross_exposure),
            "net_exposure": float(self.net_exposure),
            "turnover": float(self.turnover),
            "transaction_cost": dict(self.transaction_cost),
            "risk_contribution": dict(self.risk_contribution),
            "factor_exposure": dict(self.factor_exposure),
            "liquidity_exposure": dict(self.liquidity_exposure),
            "optimization": self.optimization.to_dict() if self.optimization else None,
            "fallback_used": bool(self.fallback_used),
            "fallback_kind": self.fallback_kind,
            "fallback_reasons": list(self.fallback_reasons),
            "risk_validation": self.risk_validation,
            "method": self.method,
            "constraints": dict(self.constraints),
            "timestamp": self.timestamp,
            "data_version": self.data_version,
            "model_version": self.model_version,
            "seed": self.seed,
            "audit": dict(self.audit),
            "names": list(self.names),
            "weights": [float(w) for w in self.weights],
            "success": bool(self.success),
            "status": self.status,
            "messages": list(self.messages),
        }

    def to_portfolio(self, *, currency: str = "USD", cash: float = 0.0) -> Portfolio:
        positions: list[Position] = []
        if isinstance(self.target_positions, TargetPositions):
            positions = list(self.target_positions.positions)
        elif isinstance(self.target_positions, list):
            positions = list(self.target_positions)
        return Portfolio(
            names=list(self.names),
            weights=[float(w) for w in self.weights],
            positions=positions,
            cash=float(cash),
            currency=currency,
            meta={"method": self.method, "fallback_used": self.fallback_used},
        )


class PortfolioConstructor:
    """Build a portfolio from provided signals / forecasts under constraints.

    Does **not** generate alpha — only expresses caller-supplied forecasts.
    """

    def __init__(self, settings: PortfolioSettings | None = None) -> None:
        self.settings = settings or PortfolioSettings.default()

    def signals_to_weights(
        self,
        signals: Sequence[float] | np.ndarray,
        *,
        method: str = "zscore",
        names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> TargetWeights:
        out = signals_to_raw_weights(
            signals,
            method=method,
            long_only=bool(kwargs.get("long_only", self.settings.long_only)),
            temperature=float(kwargs.get("temperature", 1.0)),
            budget=float(kwargs.get("budget", 1.0)),
            names=names,
        )
        return build_target_weights(
            out["weights"],
            names=out["names"],
            method=f"signal:{out['method']}",
            source="signals",
            long_only=bool(out["long_only"]),
            meta={"raw": out.get("raw")},
        )

    def build_target_weights(
        self,
        weights: Any,
        *,
        names: Sequence[str] | None = None,
        method: str = "",
        **kwargs: Any,
    ) -> TargetWeights:
        return build_target_weights(
            weights,
            names=names,
            method=method or self.settings.method,
            long_only=bool(kwargs.get("long_only", self.settings.long_only)),
            meta=kwargs.get("meta"),
        )

    def build_positions(
        self,
        weights: Any,
        *,
        capital: float,
        prices: Any,
        names: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> TargetPositions:
        return weights_to_positions(weights, capital=capital, prices=prices, names=names, **kwargs)

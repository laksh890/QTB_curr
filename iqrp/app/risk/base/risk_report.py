"""Risk report builders."""

from __future__ import annotations

from typing import Any

from iqrp.app.risk.base.risk_measure import RiskReport, RiskState


def build_report(
    *,
    portfolio_risk: dict[str, Any] | None = None,
    position_risk: dict[str, Any] | None = None,
    tail_risk: dict[str, Any] | None = None,
    liquidity_risk: dict[str, Any] | None = None,
    concentration: dict[str, Any] | None = None,
    factor_exposure: dict[str, Any] | None = None,
    drawdown: dict[str, Any] | None = None,
    stress: dict[str, Any] | None = None,
    limits: dict[str, Any] | None = None,
    breaches: list[dict[str, Any]] | None = None,
    risk_state: RiskState = RiskState.NORMAL,
    timestamp: Any = None,
    data_version: str = "1.0.0",
    model_version: str = "1.0.0",
    metadata: dict[str, Any] | None = None,
) -> RiskReport:
    return RiskReport(
        portfolio_risk=dict(portfolio_risk or {}),
        position_risk=dict(position_risk or {}),
        tail_risk=dict(tail_risk or {}),
        liquidity_risk=dict(liquidity_risk or {}),
        concentration=dict(concentration or {}),
        factor_exposure=dict(factor_exposure or {}),
        drawdown=dict(drawdown or {}),
        stress=dict(stress or {}),
        limits=dict(limits or {}),
        breaches=list(breaches or []),
        risk_state=risk_state,
        timestamp=timestamp,
        data_version=data_version,
        model_version=model_version,
        metadata=dict(metadata or {}),
    )

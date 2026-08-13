"""Dashboard payload builder for risk UI / reports."""

from __future__ import annotations

from typing import Any

from iqrp.app.risk.base import LimitBreach, RiskState
from iqrp.app.risk.monitoring.alerts import build_alerts
from iqrp.app.risk.monitoring.breaches import summarize_breaches


def dashboard_payload(
    *,
    risk_state: RiskState | str = RiskState.NORMAL,
    portfolio_risk: dict[str, Any] | None = None,
    tail_risk: dict[str, Any] | None = None,
    liquidity_risk: dict[str, Any] | None = None,
    concentration: dict[str, Any] | None = None,
    drawdown: dict[str, Any] | None = None,
    leverage: dict[str, Any] | None = None,
    stress: dict[str, Any] | None = None,
    model_risk: dict[str, Any] | None = None,
    breaches: list[Any] | None = None,
    measures: dict[str, Any] | None = None,
    timestamp: Any = None,
) -> dict[str, Any]:
    """Assemble a to_dict-friendly dashboard payload."""
    state = risk_state if isinstance(risk_state, RiskState) else RiskState(str(risk_state))
    breach_list = list(breaches or [])

    typed: list[LimitBreach] = [b for b in breach_list if isinstance(b, LimitBreach)]
    dict_breaches = [b for b in breach_list if isinstance(b, dict)]
    summary = summarize_breaches(breach_list)

    alerts = build_alerts(breaches=typed, risk_state=state, measures=measures)
    for b in dict_breaches:
        alerts.append(
            {
                "type": "limit_breach",
                "severity": b.get("severity", "WARNING"),
                "message": b.get("reason", ""),
                "limit_name": b.get("limit_name"),
                "observed": b.get("observed"),
                "threshold": b.get("threshold"),
                "scope": b.get("scope", "portfolio"),
                "metadata": dict(b.get("metadata") or {}),
            }
        )

    return {
        "name": "risk_dashboard",
        "timestamp": timestamp,
        "risk_state": state.value,
        "panels": {
            "portfolio_risk": dict(portfolio_risk or {}),
            "tail_risk": dict(tail_risk or {}),
            "liquidity_risk": dict(liquidity_risk or {}),
            "concentration": dict(concentration or {}),
            "drawdown": dict(drawdown or {}),
            "leverage": dict(leverage or {}),
            "stress": dict(stress or {}),
            "model_risk": dict(model_risk or {}),
        },
        "breaches": summary,
        "alerts": alerts,
        "measures": dict(measures or {}),
    }

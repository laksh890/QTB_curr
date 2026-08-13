"""Alert construction from breaches and risk state."""

from __future__ import annotations

from typing import Any

from iqrp.app.risk.base import LimitBreach, LimitSeverity, RiskState


def build_alerts(
    *,
    breaches: list[LimitBreach] | None = None,
    risk_state: RiskState | str = RiskState.NORMAL,
    measures: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build prioritized alert payloads."""
    alerts: list[dict[str, Any]] = []
    state = risk_state if isinstance(risk_state, RiskState) else RiskState(str(risk_state))

    if state != RiskState.NORMAL:
        severity = {
            RiskState.CAUTION: "WARNING",
            RiskState.REDUCED_RISK: "SOFT",
            RiskState.CAPITAL_PRESERVATION: "HARD",
            RiskState.TRADING_HALT: "HARD",
        }.get(state, "WARNING")
        alerts.append(
            {
                "type": "risk_state",
                "severity": severity,
                "message": f"Risk state elevated to {state.value}",
                "risk_state": state.value,
            }
        )

    for b in breaches or []:
        alerts.append(
            {
                "type": "limit_breach",
                "severity": b.severity.value,
                "message": b.reason,
                "limit_name": b.limit_name,
                "observed": b.observed,
                "threshold": b.threshold,
                "scope": b.scope,
                "metadata": dict(b.metadata),
            }
        )

    if measures:
        for name, payload in measures.items():
            if isinstance(payload, dict) and payload.get("value") is not None:
                # Optional soft alerts for extreme named measures
                if name in ("model_drift", "forecast_uncertainty", "model_disagreement"):
                    val = float(payload["value"])
                    if val > 2.0:
                        alerts.append(
                            {
                                "type": "model_risk",
                                "severity": LimitSeverity.WARNING.value,
                                "message": f"{name} elevated: {val:.4g}",
                                "measure": name,
                                "value": val,
                            }
                        )

    # Priority: HARD > SOFT > WARNING
    order = {"HARD": 0, "SOFT": 1, "WARNING": 2}
    alerts.sort(key=lambda a: order.get(str(a.get("severity")), 9))
    return alerts

"""Alert payloads for alpha monitoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_SEVERITY_ORDER = {"CRITICAL": 0, "HARD": 1, "HIGH": 2, "MEDIUM": 3, "WARNING": 4, "INFO": 5}


def build_alpha_alerts(
    *,
    retirement: Mapping[str, Any] | None = None,
    ic_decay: Mapping[str, Any] | None = None,
    drift: Mapping[str, Any] | None = None,
    performance: Mapping[str, Any] | None = None,
    signal_name: str | None = None,
    extra: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build prioritized alert list from monitor outputs."""
    alerts: list[dict[str, Any]] = []
    prefix = f"[{signal_name}] " if signal_name else ""

    if retirement is not None:
        status = str(retirement.get("status", retirement.get("recommend", "ACTIVE")))
        if status == "RETIRED":
            alerts.append(
                {
                    "type": "retirement",
                    "severity": "CRITICAL",
                    "message": f"{prefix}Signal recommended RETIRED: {retirement.get('reasons', [])}",
                    "status": status,
                    "reasons": list(retirement.get("reasons", [])),
                }
            )
        elif status == "DEGRADED":
            alerts.append(
                {
                    "type": "retirement",
                    "severity": "HIGH",
                    "message": f"{prefix}Signal recommended DEGRADED: {retirement.get('reasons', [])}",
                    "status": status,
                    "reasons": list(retirement.get("reasons", [])),
                }
            )

    if ic_decay is not None:
        st = str(ic_decay.get("status", "HEALTHY"))
        if st == "COLLAPSED":
            alerts.append(
                {
                    "type": "ic_decay",
                    "severity": "CRITICAL",
                    "message": f"{prefix}IC collapsed (ratio={ic_decay.get('ratio')})",
                    "status": st,
                }
            )
        elif st == "DECAYING":
            alerts.append(
                {
                    "type": "ic_decay",
                    "severity": "MEDIUM",
                    "message": f"{prefix}IC decaying (ratio={ic_decay.get('ratio')})",
                    "status": st,
                }
            )

    if drift is not None and drift.get("drifted"):
        sev = str(drift.get("severity", "medium")).upper()
        if sev == "NONE":
            sev = "MEDIUM"
        alerts.append(
            {
                "type": "drift",
                "severity": sev if sev in _SEVERITY_ORDER else "MEDIUM",
                "message": f"{prefix}Signal drift detected",
                "detail": {k: drift[k] for k in drift if k != "name"},
            }
        )

    if performance is not None:
        st = str(performance.get("status", "HEALTHY"))
        if st in ("COLLAPSED", "DEGRADED"):
            alerts.append(
                {
                    "type": "performance_decay",
                    "severity": "HIGH" if st == "COLLAPSED" else "MEDIUM",
                    "message": f"{prefix}Performance {st.lower()}",
                    "status": st,
                }
            )

    for item in extra or []:
        alerts.append(dict(item))

    alerts.sort(key=lambda a: _SEVERITY_ORDER.get(str(a.get("severity", "INFO")).upper(), 9))
    return alerts


def summarize_alerts(alerts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for a in alerts:
        sev = str(a.get("severity", "INFO")).upper()
        counts[sev] = counts.get(sev, 0) + 1
    return {
        "name": "alert_summary",
        "n_alerts": len(alerts),
        "by_severity": counts,
        "has_critical": counts.get("CRITICAL", 0) > 0,
        "has_actionable": any(counts.get(s, 0) > 0 for s in ("CRITICAL", "HARD", "HIGH")),
    }

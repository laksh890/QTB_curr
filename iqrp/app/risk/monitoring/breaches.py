"""Breach summarization utilities."""

from __future__ import annotations

from typing import Any

from iqrp.app.risk.base import LimitBreach, LimitSeverity


def summarize_breaches(breaches: list[LimitBreach] | list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize breach list by severity and limit name."""
    items: list[dict[str, Any]] = []
    for b in breaches:
        if isinstance(b, LimitBreach):
            items.append(b.to_dict())
        else:
            items.append(dict(b))

    by_severity = {s.value: 0 for s in LimitSeverity}
    by_limit: dict[str, int] = {}
    hard = False
    for it in items:
        sev = str(it.get("severity", "WARNING"))
        by_severity[sev] = by_severity.get(sev, 0) + 1
        name = str(it.get("limit_name", "unknown"))
        by_limit[name] = by_limit.get(name, 0) + 1
        if sev == LimitSeverity.HARD.value:
            hard = True

    return {
        "name": "breach_summary",
        "count": len(items),
        "has_hard_breach": hard,
        "by_severity": by_severity,
        "by_limit": by_limit,
        "breaches": items,
    }

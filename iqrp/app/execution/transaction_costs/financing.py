"""Financing / funding cost for leveraged or overnight positions."""

from __future__ import annotations

from typing import Any


def financing_cost(
    *,
    notional: float,
    rate: float = 0.05,
    days: float = 1.0,
    day_count: float = 360.0,
    side: str | None = None,
) -> dict[str, Any]:
    """Simple financing cost = notional * rate * days / day_count."""
    notion = abs(float(notional))
    r = float(rate)
    d = max(float(days), 0.0)
    dc = max(float(day_count), 1.0)
    # Longs typically pay; shorts may earn/pay depending on policy — magnitude returned
    total = notion * r * d / dc
    _ = side
    return {
        "name": "financing_cost",
        "total": float(total),
        "notional": notion,
        "rate": r,
        "days": d,
        "day_count": dc,
    }


__all__ = ["financing_cost"]

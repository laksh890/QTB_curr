"""Stock-borrow cost for short positions."""

from __future__ import annotations

from typing import Any


def borrow_cost(
    *,
    notional: float,
    borrow_rate: float = 0.01,
    days: float = 1.0,
    day_count: float = 360.0,
    is_short: bool = True,
) -> dict[str, Any]:
    """Borrow fee on short notional; zero when not short."""
    if not is_short:
        return {
            "name": "borrow_cost",
            "total": 0.0,
            "notional": abs(float(notional)),
            "borrow_rate": float(borrow_rate),
            "days": float(days),
            "is_short": False,
        }
    notion = abs(float(notional))
    rate = max(float(borrow_rate), 0.0)
    d = max(float(days), 0.0)
    dc = max(float(day_count), 1.0)
    total = notion * rate * d / dc
    return {
        "name": "borrow_cost",
        "total": float(total),
        "notional": notion,
        "borrow_rate": rate,
        "days": d,
        "day_count": dc,
        "is_short": True,
    }


__all__ = ["borrow_cost"]

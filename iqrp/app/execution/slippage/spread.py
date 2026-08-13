"""Bid-ask spread contribution to slippage."""

from __future__ import annotations

from typing import Any


def spread_slippage(
    *,
    mid: float,
    spread: float,
    half_spread: bool = True,
    side: str | None = None,
) -> dict[str, Any]:
    """Return spread cost in price units and bps.

    By default charges half-spread (crossing to touch). ``side`` is accepted for
    API symmetry but does not change the magnitude (cost is adverse either way).
    """
    mid_f = max(float(mid), 1e-12)
    spr = max(float(spread), 0.0)
    px = 0.5 * spr if half_spread else spr
    _ = side  # reserved for asymmetric spread models
    return {
        "name": "spread_slippage",
        "slippage": float(px),
        "slippage_bps": float(px / mid_f * 1e4),
        "half_spread": bool(half_spread),
        "spread": spr,
        "mid": mid_f,
    }


def effective_spread_bps(
    *,
    side: str,
    fill_price: float,
    mid: float,
) -> float:
    """Realized effective spread in bps vs contemporaneous mid."""
    mid_f = max(float(mid), 1e-12)
    fill = float(fill_price)
    s = str(side).lower()
    if s in {"sell", "short", "s"}:
        return float((mid_f - fill) / mid_f * 1e4)
    return float((fill - mid_f) / mid_f * 1e4)


__all__ = ["effective_spread_bps", "spread_slippage"]

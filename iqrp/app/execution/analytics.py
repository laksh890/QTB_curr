"""Execution quality analytics: IS, arrival/VWAP/TWAP slippage, fill rate, latency.

CRITICAL RULES
--------------
- Analytics never invent fills or positions.
- Uses only observed fills and provided reference prices (no future leakage).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _fill_qty(f: Mapping[str, Any]) -> float:
    return abs(float(f.get("fill_qty", f.get("quantity", f.get("qty", 0.0)))))


def _fill_price(f: Mapping[str, Any]) -> float:
    return float(f.get("fill_price", f.get("price", 0.0)))


def _vwap(fills: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    notional = 0.0
    qty = 0.0
    for f in fills:
        q = _fill_qty(f)
        p = _fill_price(f)
        notional += q * p
        qty += q
    if qty <= 0.0:
        return 0.0, 0.0
    return notional / qty, qty


def _signed_slippage_bps(side: str, arrival: float, exec_px: float) -> float:
    """Positive = adverse vs arrival (buy paid more / sell received less)."""
    if arrival <= 0.0:
        return 0.0
    side_l = str(side).strip().lower()
    is_buy = side_l in {"buy", "b", "cover", "long"}
    raw = (exec_px - arrival) / arrival * 1e4
    return float(raw if is_buy else -raw)


def implementation_shortfall(
    *,
    side: str,
    arrival_price: float,
    fills: Sequence[Mapping[str, Any]],
    decision_price: float | None = None,
) -> dict[str, Any]:
    """Implementation shortfall vs arrival (and optional decision) price."""
    vwap, qty = _vwap(fills)
    arrival = float(arrival_price)
    decision = float(decision_price) if decision_price is not None else arrival
    is_bps = _signed_slippage_bps(side, arrival, vwap) if qty > 0 else 0.0
    decision_bps = _signed_slippage_bps(side, decision, vwap) if qty > 0 else 0.0
    notional = qty * arrival
    is_cost = notional * is_bps / 1e4
    return {
        "arrival_price": arrival,
        "decision_price": decision,
        "exec_vwap": vwap,
        "filled_qty": qty,
        "implementation_shortfall_bps": is_bps,
        "decision_slippage_bps": decision_bps,
        "implementation_shortfall_cost": float(is_cost),
        "notional_at_arrival": float(notional),
    }


def arrival_slippage(
    *,
    side: str,
    arrival_price: float,
    fills: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    vwap, qty = _vwap(fills)
    bps = _signed_slippage_bps(side, float(arrival_price), vwap) if qty > 0 else 0.0
    return {
        "arrival_price": float(arrival_price),
        "exec_vwap": vwap,
        "filled_qty": qty,
        "arrival_slippage_bps": bps,
    }


def benchmark_slippage(
    *,
    side: str,
    benchmark_price: float,
    fills: Sequence[Mapping[str, Any]],
    label: str = "benchmark",
) -> dict[str, Any]:
    vwap, qty = _vwap(fills)
    bps = _signed_slippage_bps(side, float(benchmark_price), vwap) if qty > 0 else 0.0
    return {
        "label": label,
        "benchmark_price": float(benchmark_price),
        "exec_vwap": vwap,
        "filled_qty": qty,
        "slippage_bps": bps,
    }


def fill_rate(*, ordered_qty: float, filled_qty: float) -> dict[str, Any]:
    ordered = abs(float(ordered_qty))
    filled = abs(float(filled_qty))
    rate = (filled / ordered) if ordered > 0 else 0.0
    return {
        "ordered_qty": ordered,
        "filled_qty": filled,
        "residual_qty": max(ordered - filled, 0.0),
        "fill_rate": float(min(max(rate, 0.0), 1.0)),
    }


def execution_quality_report(
    *,
    side: str,
    ordered_qty: float,
    fills: Sequence[Mapping[str, Any]],
    arrival_price: float,
    vwap_benchmark: float | None = None,
    twap_benchmark: float | None = None,
    latency: Mapping[str, Any] | None = None,
    pre_trade_estimate: Mapping[str, Any] | None = None,
    post_trade_costs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose a post-trade execution quality report."""
    vwap, filled = _vwap(fills)
    is_report = implementation_shortfall(side=side, arrival_price=arrival_price, fills=fills)
    arr = arrival_slippage(side=side, arrival_price=arrival_price, fills=fills)
    fr = fill_rate(ordered_qty=ordered_qty, filled_qty=filled)
    report: dict[str, Any] = {
        "side": str(side),
        "ordered_qty": abs(float(ordered_qty)),
        "filled_qty": filled,
        "exec_vwap": vwap,
        "fill_rate": fr,
        "implementation_shortfall": is_report,
        "arrival_slippage": arr,
        "latency": dict(latency or {}),
        "pre_trade_estimate": dict(pre_trade_estimate or {}),
        "post_trade_costs": dict(post_trade_costs or {}),
    }
    if vwap_benchmark is not None:
        report["vwap_slippage"] = benchmark_slippage(
            side=side,
            benchmark_price=float(vwap_benchmark),
            fills=fills,
            label="vwap",
        )
    if twap_benchmark is not None:
        report["twap_slippage"] = benchmark_slippage(
            side=side,
            benchmark_price=float(twap_benchmark),
            fills=fills,
            label="twap",
        )
    return report


__all__ = [
    "arrival_slippage",
    "benchmark_slippage",
    "execution_quality_report",
    "fill_rate",
    "implementation_shortfall",
]

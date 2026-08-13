"""Post-trade realized slippage vs expected."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from iqrp.app.execution.slippage.estimator import estimate_slippage


def _vwap(fills: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    notionals = 0.0
    qty = 0.0
    for f in fills:
        q = abs(float(f.get("quantity", f.get("qty", 0.0))))
        p = float(f.get("price", f.get("fill_price", 0.0)))
        notionals += q * p
        qty += q
    if qty <= 0.0:
        return 0.0, 0.0
    return notionals / qty, qty


def realized_slippage(
    fills: Sequence[Mapping[str, Any]],
    *,
    side: str,
    arrival_price: float,
    decision_price: float | None = None,
    mid: float | None = None,
) -> dict[str, Any]:
    """Compute realized slippage vs arrival / decision / mid benchmarks."""
    vwap, qty = _vwap(fills)
    arr = max(float(arrival_price), 1e-12)
    decision = float(decision_price) if decision_price is not None else arr
    mid_f = float(mid) if mid is not None else arr
    s = str(side).lower()
    buy = s not in {"sell", "short", "s"}

    def _bps(fill: float, bench: float) -> float:
        if buy:
            return float((fill - bench) / max(bench, 1e-12) * 1e4)
        return float((bench - fill) / max(bench, 1e-12) * 1e4)

    def _px(fill: float, bench: float) -> float:
        if buy:
            return float(fill - bench)
        return float(bench - fill)

    arrival_px = _px(vwap, arr) if vwap > 0 else 0.0
    decision_px = _px(vwap, decision) if vwap > 0 else 0.0
    mid_px = _px(vwap, mid_f) if vwap > 0 else 0.0

    return {
        "name": "realized_slippage",
        "side": s,
        "vwap": float(vwap),
        "filled_quantity": float(qty),
        "arrival_price": arr,
        "decision_price": float(decision),
        "mid": mid_f,
        "realized_slippage": arrival_px,
        "realized_slippage_bps": _bps(vwap, arr) if vwap > 0 else 0.0,
        "decision_slippage": decision_px,
        "decision_slippage_bps": _bps(vwap, decision) if vwap > 0 else 0.0,
        "mid_slippage": mid_px,
        "mid_slippage_bps": _bps(vwap, mid_f) if vwap > 0 else 0.0,
        "realized_slippage_notional": arrival_px * qty,
    }


def compare_expected_realized(
    fills: Sequence[Mapping[str, Any]],
    *,
    side: str,
    quantity: float,
    mid: float,
    arrival_price: float | None = None,
    spread: float = 0.0,
    adv: float = 1e6,
    volatility: float = 0.02,
    **estimate_kwargs: Any,
) -> dict[str, Any]:
    """Compare pre-trade expected slippage to post-trade realized."""
    arr = float(arrival_price) if arrival_price is not None else float(mid)
    expected = estimate_slippage(
        side=side,
        quantity=quantity,
        mid=mid,
        spread=spread,
        adv=adv,
        volatility=volatility,
        **estimate_kwargs,
    )
    realized = realized_slippage(
        fills,
        side=side,
        arrival_price=arr,
        mid=mid,
    )
    exp_bps = float(expected["expected_slippage_bps"])
    real_bps = float(realized["realized_slippage_bps"])
    return {
        "name": "compare_expected_realized",
        "expected": expected,
        "realized": realized,
        "forecast_error_bps": real_bps - exp_bps,
        "forecast_error": float(realized["realized_slippage"]) - float(expected["expected_slippage"]),
        "expected_slippage_bps": exp_bps,
        "realized_slippage_bps": real_bps,
    }


__all__ = ["compare_expected_realized", "realized_slippage"]

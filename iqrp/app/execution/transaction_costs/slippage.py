"""Slippage transaction cost wrapper."""

from __future__ import annotations

from typing import Any

from iqrp.app.execution.slippage.estimator import estimate_slippage


def slippage_cost(
    *,
    side: str,
    quantity: float,
    mid: float,
    spread: float = 0.0,
    adv: float = 1e6,
    volatility: float = 0.02,
    liquidity: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any]:
    """Currency slippage cost from the execution slippage estimator."""
    est = estimate_slippage(
        side=side,
        quantity=quantity,
        mid=mid,
        spread=spread,
        adv=adv,
        volatility=volatility,
        liquidity=liquidity,
        **kwargs,
    )
    return {
        "name": "slippage_cost",
        "total": float(est["expected_slippage_notional"]),
        "slippage_px": float(est["expected_slippage"]),
        "slippage_bps": float(est["expected_slippage_bps"]),
        "notional": float(est["notional"]),
        "components": est["components"],
        "detail": est,
    }


__all__ = ["slippage_cost"]

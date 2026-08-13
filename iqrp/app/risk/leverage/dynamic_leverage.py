"""Dynamic leverage recommendation."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure


def recommended_leverage(
    *,
    realized_vol: float,
    target_vol: float = 0.10,
    current_drawdown: float = 0.0,
    max_drawdown: float = 0.20,
    confidence: float = 1.0,
    liquidity_score: float = 1.0,
    regime: str | int = "normal",
    base_leverage: float = 1.0,
    max_leverage: float = 2.0,
    min_leverage: float = 0.0,
    confidence_cap: float = 1.25,
    regime_scales: dict[str, float] | None = None,
) -> RiskMeasure:
    """Combine vol, drawdown, confidence, liquidity, and regime into leverage.

    Hard constraints:
    - Final leverage clipped to [min_leverage, max_leverage]
    - Confidence may scale up to ``confidence_cap`` only (never unbounded)
    - Drawdown at/above ``max_drawdown`` forces min_leverage (hard limit)
    """
    rv = max(float(realized_vol), 1e-12)
    tv = max(float(target_vol), 0.0)
    base = max(float(base_leverage), 0.0)
    max_lev = max(float(max_leverage), 0.0)
    min_lev = float(np.clip(min_leverage, 0.0, max_lev))
    conf_cap = max(float(confidence_cap), 1.0)

    vol_scalar = tv / rv
    dd = max(float(current_drawdown), 0.0)
    dd_limit = max(float(max_drawdown), 1e-12)
    dd_scalar = float(np.clip(1.0 - dd / dd_limit, 0.0, 1.0))

    conf = float(np.clip(confidence, 0.0, 1.0))
    # Map confidence in [0,1] to scale in [1/conf_cap, conf_cap] but never above conf_cap
    conf_scalar = 1.0 / conf_cap + conf * (conf_cap - 1.0 / conf_cap)
    conf_scalar = float(np.clip(conf_scalar, 0.0, conf_cap))

    liq = float(np.clip(liquidity_score, 0.0, 1.0))

    defaults = {
        "normal": 1.0,
        "low_vol": 1.0,
        "high_vol": 0.6,
        "crisis": 0.25,
        "stress": 0.35,
        "transition": 0.7,
    }
    scales = dict(defaults)
    if regime_scales:
        scales.update({str(k): float(v) for k, v in regime_scales.items()})
    regime_key = str(regime).lower()
    regime_scalar = float(np.clip(scales.get(regime_key, 0.5), 0.0, 1.0))

    raw = base * vol_scalar * dd_scalar * conf_scalar * liq * regime_scalar

    if dd >= dd_limit:
        # Hard drawdown limit cannot be overridden by confidence
        lev = min_lev
        hard_halt = True
    else:
        lev = float(np.clip(raw, min_lev, max_lev))
        hard_halt = False

    return RiskMeasure(
        name="recommended_leverage",
        value=lev,
        unit="leverage",
        method="dynamic_leverage",
        confidence=conf,
        parameters={
            "realized_vol": float(realized_vol),
            "target_vol": tv,
            "current_drawdown": dd,
            "max_drawdown": dd_limit,
            "liquidity_score": liq,
            "regime": regime_key,
            "base_leverage": base,
            "max_leverage": max_lev,
            "min_leverage": min_lev,
            "confidence_cap": conf_cap,
            "vol_scalar": vol_scalar,
            "dd_scalar": dd_scalar,
            "conf_scalar": conf_scalar,
            "regime_scalar": regime_scalar,
            "uncapped": float(raw),
            "hard_halt": hard_halt,
        },
    )

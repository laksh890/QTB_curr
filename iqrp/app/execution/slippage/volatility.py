"""Volatility-driven slippage component."""

from __future__ import annotations

from typing import Any

import numpy as np


def volatility_slippage(
    *,
    mid: float,
    volatility: float,
    horizon_seconds: float = 0.0,
    trading_day_seconds: float = 23400.0,
    coeff: float = 0.25,
    delay_seconds: float = 0.0,
) -> dict[str, Any]:
    """Price-unit volatility impact over an execution horizon / delay."""
    mid_f = max(float(mid), 1e-12)
    vol = max(float(volatility), 0.0)
    day = max(float(trading_day_seconds), 1.0)
    horizon = max(float(horizon_seconds), max(float(delay_seconds), 0.0))
    time_frac = float(np.sqrt(horizon / day)) if horizon > 0 else 0.0
    px = float(coeff) * vol * mid_f * time_frac
    return {
        "name": "volatility_slippage",
        "slippage": px,
        "slippage_bps": float(px / mid_f * 1e4),
        "volatility": vol,
        "time_fraction": time_frac,
        "coeff": float(coeff),
    }


__all__ = ["volatility_slippage"]

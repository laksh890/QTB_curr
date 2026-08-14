"""Volatility targeting position sizing."""

from __future__ import annotations

import numpy as np

from iqrp.app.risk.base import RiskMeasure


def volatility_target_size(
    *,
    realized_vol: float,
    target_vol: float = 0.10,
    max_leverage: float = 2.0,
    base_size: float = 1.0,
) -> RiskMeasure:
    """Size = base * target_vol / realized_vol, clipped by max_leverage."""
    rv = max(float(realized_vol), 1e-12)
    tv = max(float(target_vol), 0.0)
    max_lev = max(float(max_leverage), 0.0)
    raw = float(base_size) * tv / rv
    size = float(np.clip(raw, 0.0, max_lev * abs(float(base_size))))
    return RiskMeasure(
        name="volatility_target_size",
        value=size,
        unit="size",
        method="vol_target",
        parameters={
            "realized_vol": float(realized_vol),
            "target_vol": tv,
            "max_leverage": max_lev,
            "base_size": float(base_size),
            "uncapped": raw,
        },
    )


def fixed_fractional_size(
    *,
    equity: float,
    risk_fraction: float = 0.01,
    stop_distance: float = 0.02,
    max_size: float | None = None,
) -> RiskMeasure:
    """Fixed-fractional position size: (equity * f) / stop_distance."""
    eq = max(float(equity), 0.0)
    f = float(np.clip(risk_fraction, 0.0, 1.0))
    stop = max(float(stop_distance), 1e-12)
    raw = (eq * f) / stop
    if max_size is not None:
        size = float(min(raw, max(float(max_size), 0.0)))
    else:
        size = float(raw)
    return RiskMeasure(
        name="fixed_fractional_size",
        value=size,
        unit="notional",
        method="fixed_fractional",
        parameters={
            "equity": eq,
            "risk_fraction": f,
            "stop_distance": stop,
            "max_size": max_size,
        },
    )


def confidence_adjusted_size(
    *,
    base_size: float,
    confidence: float,
    min_scale: float = 0.25,
    max_scale: float = 1.0,
) -> RiskMeasure:
    """Scale size by forecast confidence in [min_scale, max_scale].

    Confidence cannot expand size beyond max_scale (hard ceiling).
    """
    conf = float(np.clip(confidence, 0.0, 1.0))
    lo = float(np.clip(min_scale, 0.0, 1.0))
    hi = float(np.clip(max_scale, lo, 1.0))
    scale = lo + conf * (hi - lo)
    size = float(base_size) * scale
    return RiskMeasure(
        name="confidence_adjusted_size",
        value=size,
        unit="size",
        method="confidence_scale",
        confidence=conf,
        parameters={
            "base_size": float(base_size),
            "scale": scale,
            "min_scale": lo,
            "max_scale": hi,
        },
    )


def regime_adjusted_size(
    *,
    base_size: float,
    regime: str | int = "normal",
    regime_scales: dict[str, float] | None = None,
) -> RiskMeasure:
    """Scale size by regime label; unknown regimes default to conservative scale."""
    defaults = {
        "normal": 1.0,
        "low_vol": 1.0,
        "high_vol": 0.5,
        "crisis": 0.25,
        "stress": 0.35,
        "transition": 0.6,
        "0": 1.0,
        "1": 0.7,
        "2": 0.4,
    }
    scales = dict(defaults)
    if regime_scales:
        scales.update({str(k): float(v) for k, v in regime_scales.items()})
    key = str(regime).lower()
    scale = float(scales.get(key, 0.5))
    scale = float(np.clip(scale, 0.0, 1.0))
    size = float(base_size) * scale
    return RiskMeasure(
        name="regime_adjusted_size",
        value=size,
        unit="size",
        method="regime_scale",
        parameters={"base_size": float(base_size), "regime": key, "scale": scale},
    )

"""Capacity and liquidity constraints for capital allocation.

Calls ``iqrp.app.risk.market.liquidity.liquidity_risk``. Missing ADV/spread
never assumes unlimited capacity — applies conservative downscale.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.market.liquidity import liquidity_risk


def _as_float_array(
    values: np.ndarray | list[float] | None,
    n: int,
    *,
    fill: float | None,
) -> tuple[np.ndarray, bool]:
    """Return length-n array and whether inputs were missing/incomplete."""
    if values is None:
        if fill is None:
            return np.full(n, np.nan), True
        return np.full(n, float(fill), dtype=np.float64), True
    arr = np.asarray(values, dtype=np.float64).ravel()
    missing = False
    if arr.size != n:
        missing = True
        out = np.full(n, float(fill) if fill is not None else np.nan, dtype=np.float64)
        m = min(arr.size, n)
        if m > 0:
            out[:m] = arr[:m]
        return out, True
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        missing = True
        out = arr.copy()
        bad = ~np.isfinite(out) | (out <= 0)
        if fill is not None:
            out[bad] = float(fill)
        else:
            out[bad] = np.nan
        return out, missing
    return arr, False


def estimate_capacity(
    names: list[str],
    *,
    capital: float = 1.0,
    weights: np.ndarray | list[float] | None = None,
    adv: np.ndarray | list[float] | None = None,
    spreads: np.ndarray | list[float] | None = None,
    vols: np.ndarray | list[float] | None = None,
    max_participation: float = 0.10,
    impact_coeff: float = 0.10,
    ttl_days: float = 5.0,
    missing_capacity_scale: float = 0.50,
    missing_liquidity_scale: float = 0.50,
    default_adv: float = 1.0e6,
    default_spread: float = 0.002,
) -> dict[str, Any]:
    """Per-name capacity scales from ADV, participation, spread, impact, slippage, TTL."""
    n = len(names)
    if n == 0:
        return {
            "name": "capacity",
            "scales": {},
            "scores": {},
            "measures": {},
            "missing_capacity": True,
            "missing_liquidity": True,
        }

    w = np.asarray(weights if weights is not None else np.full(n, 1.0 / n), dtype=np.float64)
    if w.size != n:
        w = np.full(n, 1.0 / n)
    w = np.maximum(w, 0.0)
    s = float(np.sum(w))
    if s > 0:
        w = w / s

    adv_arr, adv_missing = _as_float_array(adv, n, fill=default_adv)
    spr_arr, spr_missing = _as_float_array(spreads, n, fill=default_spread)
    vol_arr, _ = _as_float_array(vols, n, fill=0.01)
    missing_liquidity = adv_missing or spr_missing

    cap = max(float(capital), 0.0)
    part_cap = float(np.clip(max_participation, 1e-6, 1.0))
    ttl = max(float(ttl_days), 1e-6)

    scales: dict[str, float] = {}
    scores: dict[str, float] = {}
    measures: dict[str, Any] = {}
    max_caps: dict[str, float] = {}

    for i, name in enumerate(names):
        notional = cap * float(w[i])
        # Max notional tradable within TTL at participation cap
        daily_cap = part_cap * float(adv_arr[i])
        max_notional = daily_cap * ttl
        max_caps[name] = float(max_notional)

        lr = liquidity_risk(
            position_size=max(notional, 1e-12),
            adv=float(adv_arr[i]),
            spread=float(spr_arr[i]),
            price=1.0,
            volatility=float(vol_arr[i]),
            max_participation=part_cap,
            impact_coeff=float(impact_coeff),
        )
        score = float(lr.get("score", 0.5))
        scores[name] = score
        measures[name] = lr.get("measures", {})

        # Capacity scale: if position exceeds TTL capacity, downscale
        if max_notional <= 1e-12:
            cap_scale = float(missing_capacity_scale)
        else:
            util = notional / max_notional if notional > 0 else 0.0
            # Soft: full scale if util <= 1; inverse beyond
            cap_scale = float(np.clip(1.0 / max(util, 1.0), 0.0, 1.0))
            # Also fold liquidity score (higher = safer)
            cap_scale *= float(np.clip(0.25 + 0.75 * score, 0.0, 1.0))

        if missing_liquidity:
            cap_scale *= float(np.clip(missing_liquidity_scale, 0.0, 1.0))
        if adv_missing:
            cap_scale = min(cap_scale, float(missing_capacity_scale))

        scales[name] = float(np.clip(cap_scale, 0.0, 1.0))

    return {
        "name": "capacity",
        "scales": scales,
        "scores": scores,
        "measures": measures,
        "max_notional": max_caps,
        "missing_capacity": bool(adv_missing),
        "missing_liquidity": bool(missing_liquidity),
        "parameters": {
            "max_participation": part_cap,
            "impact_coeff": float(impact_coeff),
            "ttl_days": ttl,
            "missing_capacity_scale": float(missing_capacity_scale),
            "missing_liquidity_scale": float(missing_liquidity_scale),
            "default_adv": float(default_adv),
            "default_spread": float(default_spread),
        },
    }


def apply_capacity_scales(
    weights: np.ndarray | list[float],
    scales: dict[str, float] | np.ndarray | list[float],
    *,
    names: list[str] | None = None,
) -> np.ndarray:
    """Elementwise capacity downscale then renormalize (if mass remains)."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    n = w.size
    if isinstance(scales, dict):
        keys = names if names and len(names) == n else [f"s{i}" for i in range(n)]
        s = np.asarray([float(scales.get(k, 1.0)) for k in keys], dtype=np.float64)
    else:
        s = np.asarray(scales, dtype=np.float64).ravel()
        if s.size != n:
            s = np.ones(n, dtype=np.float64)
    s = np.clip(s, 0.0, 1.0)
    out = np.maximum(w, 0.0) * s
    tot = float(np.sum(out))
    if tot > 1e-12:
        out = out / tot
    return out

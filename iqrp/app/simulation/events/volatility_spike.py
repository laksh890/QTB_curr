"""Volatility explosion and liquidity collapse injectors."""

from __future__ import annotations

import numpy as np


def inject_volatility_spikes(
    prices: np.ndarray,
    volatility: np.ndarray,
    *,
    probability: float,
    rng: np.random.Generator,
    mult: float = 3.0,
    duration: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Amplify local returns during spike windows.

    Returns (prices, volatility, mask).
    """
    out_p = np.asarray(prices, dtype=np.float64).copy()
    out_v = np.asarray(volatility, dtype=np.float64).copy()
    n = len(out_p)
    mask = np.zeros(n, dtype=np.bool_)
    # Work on returns then rebuild
    if n < 3:
        return out_p, out_v, mask
    log_p = np.log(np.clip(out_p, 1e-12, None))
    rets = np.diff(log_p)
    vol = (
        out_v
        if len(out_v) == len(rets)
        else (out_v[1:] if len(out_v) == n else np.full(len(rets), float(np.mean(out_v))))
    )
    t = 0
    while t < len(rets) - duration:
        if rng.random() < probability:
            mask[t + 1 : t + 1 + duration] = True
            rets[t : t + duration] *= mult * (0.8 + 0.4 * rng.random())
            vol[t : t + duration] *= mult
            t += duration
        else:
            t += 1
    new_log = log_p[0] + np.concatenate([[0.0], np.cumsum(rets)])
    out_p = np.exp(new_log)
    if len(out_v) == n:
        out_v[1:] = vol[: n - 1]
        out_v[0] = vol[0]
    else:
        out_v = vol
    return out_p, out_v, mask


def inject_liquidity_collapse(
    spreads_bps: np.ndarray,
    volumes: np.ndarray,
    *,
    probability: float,
    rng: np.random.Generator,
    duration: int = 5,
    spread_mult: float = 8.0,
    volume_mult: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_s = np.asarray(spreads_bps, dtype=np.float64).copy()
    out_v = np.asarray(volumes, dtype=np.float64).copy()
    n = min(len(out_s), len(out_v))
    mask = np.zeros(n, dtype=np.bool_)
    t = 0
    while t < n - duration:
        if rng.random() < probability:
            mask[t : t + duration] = True
            out_s[t : t + duration] *= spread_mult
            out_v[t : t + duration] *= volume_mult
            t += duration
        else:
            t += 1
    return out_s, out_v, mask

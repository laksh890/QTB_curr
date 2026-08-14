"""Signal half-life / forward-return decay analysis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from iqrp.app.backtesting.performance.returns import as_returns


DEFAULT_FORWARD_BARS: tuple[int, ...] = (1, 2, 3, 5, 10, 20)


def forward_returns(prices: Any, horizons: Sequence[int] | None = None) -> dict[int, np.ndarray]:
    """Compute simple forward returns for each horizon in bars."""
    px = np.asarray(prices, dtype=np.float64).reshape(-1)
    hs = list(horizons or DEFAULT_FORWARD_BARS)
    out: dict[int, np.ndarray] = {}
    n = px.size
    for h in hs:
        h = int(h)
        fr = np.full(n, np.nan, dtype=np.float64)
        if h <= 0 or n <= h:
            out[h] = fr
            continue
        fr[: n - h] = px[h:] / px[: n - h] - 1.0
        out[h] = fr
    return out


def information_coefficient(signal: Any, forward: Any) -> float:
    s = np.asarray(signal, dtype=np.float64).reshape(-1)
    f = np.asarray(forward, dtype=np.float64).reshape(-1)
    n = min(s.size, f.size)
    if n < 3:
        return float("nan")
    mask = np.isfinite(s[:n]) & np.isfinite(f[:n])
    if int(mask.sum()) < 3:
        return float("nan")
    return float(np.corrcoef(s[:n][mask], f[:n][mask])[0, 1])


def signal_half_life_report(
    signal: Any,
    prices: Any,
    *,
    horizons: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Measure how long predictive information remains useful.

    Returns per-horizon mean/median forward return, volatility, hit rate, IC,
    and a coarse decay profile (IC vs horizon).
    """
    sig = np.asarray(signal, dtype=np.float64).reshape(-1)
    hs = list(horizons or DEFAULT_FORWARD_BARS)
    fr_map = forward_returns(prices, hs)
    by_h: dict[str, dict[str, Any]] = {}
    ics: list[float] = []
    for h in hs:
        fr = fr_map[int(h)]
        n = min(sig.size, fr.size)
        s = sig[:n]
        f = fr[:n]
        mask = np.isfinite(s) & np.isfinite(f) & (np.abs(s) > 1e-15)
        if int(mask.sum()) == 0:
            by_h[str(h)] = {
                "mean_forward_return": None,
                "median_forward_return": None,
                "volatility": None,
                "hit_rate": None,
                "information_coefficient": None,
                "n": 0,
            }
            ics.append(float("nan"))
            continue
        signed = np.sign(s[mask]) * f[mask]
        hit = float(np.mean(signed > 0))
        ic = information_coefficient(s[mask], f[mask])
        ics.append(ic)
        by_h[str(h)] = {
            "mean_forward_return": float(np.mean(signed)),
            "median_forward_return": float(np.median(signed)),
            "volatility": float(np.std(signed, ddof=1)) if signed.size > 1 else 0.0,
            "hit_rate": hit,
            "information_coefficient": ic if np.isfinite(ic) else None,
            "n": int(mask.sum()),
        }

    # Half-life estimate: first horizon where |IC| drops below half of peak |IC|
    finite_ics = [(h, ic) for h, ic in zip(hs, ics, strict=False) if np.isfinite(ic)]
    half_life_bars: int | None = None
    if finite_ics:
        peak_h, peak_ic = max(finite_ics, key=lambda x: abs(x[1]))
        thresh = abs(peak_ic) * 0.5
        for h, ic in finite_ics:
            if h >= peak_h and abs(ic) <= thresh + 1e-12:
                half_life_bars = int(h)
                break
        if half_life_bars is None and finite_ics:
            half_life_bars = int(finite_ics[-1][0])

    return {
        "horizons_bars": [int(h) for h in hs],
        "by_horizon": by_h,
        "decay_profile_ic": {str(h): (ic if np.isfinite(ic) else None) for h, ic in zip(hs, ics, strict=False)},
        "estimated_half_life_bars": half_life_bars,
        "note": "Half-life is research/diagnostic; not a live-edge claim.",
    }


def position_signal_from_returns(returns: Any, lookback: int = 1) -> np.ndarray:
    """Simple signed momentum signal for half-life demos."""
    r = as_returns(returns)
    sig = np.zeros_like(r)
    lb = max(int(lookback), 1)
    for i in range(lb, r.size):
        sig[i] = float(np.sign(np.sum(r[i - lb : i])))
    return sig


__all__ = [
    "DEFAULT_FORWARD_BARS",
    "forward_returns",
    "information_coefficient",
    "position_signal_from_returns",
    "signal_half_life_report",
]

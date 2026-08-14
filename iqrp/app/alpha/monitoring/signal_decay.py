"""Live / research IC decay monitoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return float("nan")
    a, b = x[m], y[m]
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rolling_ic(
    signal: Any,
    forward_returns: Any,
    *,
    window: int = 60,
    step: int = 1,
    rank: bool = False,
) -> dict[str, Any]:
    """Rolling IC time series for live monitoring."""
    sig = np.asarray(signal, dtype=np.float64)
    ret = np.asarray(forward_returns, dtype=np.float64)
    if sig.shape != ret.shape:
        raise ValueError("signal and forward_returns shape mismatch")
    w = max(5, int(window))
    ics: list[float] = []
    indices: list[int] = []

    def _ic_slice(s: np.ndarray, r: np.ndarray) -> float:
        if s.ndim == 1:
            if rank:
                m = np.isfinite(s) & np.isfinite(r)
                if m.sum() < 3:
                    return float("nan")
                a = s[m].argsort().argsort().astype(np.float64)
                b = r[m].argsort().argsort().astype(np.float64)
                return (
                    float(np.corrcoef(a, b)[0, 1])
                    if np.std(a) > 0 and np.std(b) > 0
                    else float("nan")
                )
            return _pearson(s, r)
        daily = []
        for i in range(s.shape[0]):
            daily.append(
                _pearson(s[i], r[i])
                if not rank
                else _pearson(
                    (
                        s[i].argsort().argsort().astype(np.float64)
                        if np.isfinite(s[i]).any()
                        else s[i]
                    ),
                    (
                        r[i].argsort().argsort().astype(np.float64)
                        if np.isfinite(r[i]).any()
                        else r[i]
                    ),
                )
            )
        arr = np.asarray(daily, dtype=np.float64)
        return float(np.nanmean(arr)) if arr.size else float("nan")

    n = sig.shape[0]
    for start in range(0, max(0, n - w + 1), max(1, int(step))):
        sl = slice(start, start + w)
        ics.append(_ic_slice(sig[sl], ret[sl]))
        indices.append(start + w - 1)

    arr = np.asarray(ics, dtype=np.float64)
    return {
        "name": "rolling_ic",
        "window": w,
        "step": int(step),
        "indices": indices,
        "ic": ics,
        "mean": float(np.nanmean(arr)) if arr.size else float("nan"),
        "std": float(np.nanstd(arr)) if arr.size else float("nan"),
        "last": float(arr[-1]) if arr.size and np.isfinite(arr[-1]) else float("nan"),
    }


def ic_decay_curve(
    signal: Any,
    returns: Any,
    *,
    horizons: Sequence[int] = (1, 2, 3, 5, 10, 21),
) -> dict[str, Any]:
    """IC at multiple forward horizons. ``returns`` is period returns ``(T,)`` or ``(T, N)``."""
    sig = np.asarray(signal, dtype=np.float64)
    ret = np.asarray(returns, dtype=np.float64)
    hs = [int(h) for h in horizons if int(h) > 0]
    curve: list[dict[str, Any]] = []

    for h in hs:
        if ret.ndim == 1:
            # compound forward return
            fwd = np.full(ret.shape[0], np.nan, dtype=np.float64)
            for t in range(ret.shape[0] - h):
                window = ret[t + 1 : t + 1 + h]
                if np.all(np.isfinite(window)):
                    fwd[t] = float(np.prod(1.0 + window) - 1.0)
            if sig.ndim == 1:
                ic = _pearson(sig, fwd)
            else:
                raise ValueError("1D returns require 1D signal for horizon decay")
        else:
            # assume returns already aligned as h-period forward returns panel when h==1;
            # otherwise shift
            if h == 1:
                fwd = ret
            else:
                fwd = np.full_like(ret, np.nan)
                for t in range(ret.shape[0] - h):
                    block = ret[t + 1 : t + 1 + h]
                    fwd[t] = np.nanprod(1.0 + block, axis=0) - 1.0
            if sig.ndim == 1:
                ic = _pearson(sig, np.nanmean(fwd, axis=1))
            else:
                daily = [_pearson(sig[i], fwd[i]) for i in range(min(sig.shape[0], fwd.shape[0]))]
                ic = float(np.nanmean(daily))
        curve.append({"horizon": h, "ic": float(ic) if np.isfinite(ic) else float("nan")})

    ics = np.asarray([c["ic"] for c in curve], dtype=np.float64)
    half_life = estimate_ic_half_life(hs, ics)
    return {
        "name": "ic_decay_curve",
        "curve": curve,
        "half_life": half_life,
        "decay_rate": _decay_rate(hs, ics),
    }


def estimate_ic_half_life(horizons: Sequence[int], ics: np.ndarray) -> float:
    """Half-life where |IC| falls to half of first horizon |IC|."""
    h = np.asarray(list(horizons), dtype=np.float64)
    y = np.abs(np.asarray(ics, dtype=np.float64))
    if h.size == 0 or not np.isfinite(y[0]) or y[0] < 1e-12:
        return float("nan")
    target = 0.5 * y[0]
    for i in range(1, h.size):
        if np.isfinite(y[i]) and y[i] <= target:
            # linear interpolate in horizon space
            y0, y1 = y[i - 1], y[i]
            h0, h1 = h[i - 1], h[i]
            if abs(y1 - y0) < 1e-15:
                return float(h1)
            return float(h0 + (target - y0) * (h1 - h0) / (y1 - y0))
    return float("nan")


def _decay_rate(horizons: Sequence[int], ics: np.ndarray) -> float:
    """Fit log|IC| ~ a - b * h ; return b (higher = faster decay)."""
    h = np.asarray(list(horizons), dtype=np.float64)
    y = np.abs(np.asarray(ics, dtype=np.float64))
    m = np.isfinite(h) & np.isfinite(y) & (y > 1e-12)
    if m.sum() < 2:
        return float("nan")
    yh = np.log(y[m])
    x = h[m]
    b = float(np.polyfit(x, yh, 1)[0])
    return float(-b)  # positive => decay


def monitor_ic_decay(
    rolling: Mapping[str, Any] | Any,
    *,
    baseline_ic: float,
    collapse_ratio: float = 0.3,
    warn_ratio: float = 0.6,
) -> dict[str, Any]:
    """Compare recent rolling IC to baseline; flag decay."""
    if isinstance(rolling, Mapping) and "ic" in rolling:
        series = np.asarray(rolling["ic"], dtype=np.float64)
        last = float(rolling.get("last", series[-1] if series.size else np.nan))
    else:
        series = np.asarray(rolling, dtype=np.float64)
        last = float(series[-1]) if series.size else float("nan")

    base = abs(float(baseline_ic)) + 1e-12
    ratio = abs(last) / base if np.isfinite(last) else 0.0
    if ratio <= collapse_ratio or (
        np.isfinite(last) and np.sign(last) != np.sign(baseline_ic) and abs(baseline_ic) > 1e-6
    ):
        status = "COLLAPSED"
    elif ratio <= warn_ratio:
        status = "DECAYING"
    else:
        status = "HEALTHY"
    return {
        "name": "monitor_ic_decay",
        "status": status,
        "recent_ic": last,
        "baseline_ic": float(baseline_ic),
        "ratio": float(ratio),
        "collapse_ratio": float(collapse_ratio),
        "warn_ratio": float(warn_ratio),
    }

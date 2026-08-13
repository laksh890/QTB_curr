"""Simple symbolic operations on arrays for candidate signal formulas.

CRITICAL — Point-in-time: no future leakage in signal computation helpers.
All rolling / lag operations use only past windows (including the current bar
when specified). Statistical significance of a formula alone ≠ alpha.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats  # type: ignore[import-untyped]


def as_float1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D array, got shape {arr.shape}")
    return arr


def lag(x: np.ndarray, periods: int = 1) -> np.ndarray:
    """Shift series forward in time by ``periods`` (past values only).

    ``out[t] = x[t - periods]``; leading entries are NaN.
    Point-in-time: never peeks at future observations.
    """
    if periods < 0:
        raise ValueError("lag periods must be >= 0 (negative lag would leak future)")
    x = as_float1d(x)
    if periods == 0:
        return x.copy()
    out = np.full_like(x, np.nan)
    if periods < len(x):
        out[periods:] = x[:-periods]
    return out


def diff(x: np.ndarray, periods: int = 1) -> np.ndarray:
    """First difference using only past values: ``x[t] - x[t - periods]``."""
    x = as_float1d(x)
    return x - lag(x, periods)


def ratio(numerator: np.ndarray, denominator: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Element-wise ratio with safe denominator."""
    num = as_float1d(numerator)
    den = as_float1d(denominator)
    if len(num) != len(den):
        raise ValueError("ratio operands must have equal length")
    out = np.full_like(num, np.nan)
    mask = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > eps)
    out[mask] = num[mask] / den[mask]
    return out


def rolling_apply(
    x: np.ndarray,
    window: int,
    fn: Callable[[np.ndarray], float],
    *,
    min_periods: int | None = None,
) -> np.ndarray:
    """Apply ``fn`` to each trailing window ending at t (inclusive).

    Point-in-time: window is ``x[t - window + 1 : t + 1]`` — past only.
    """
    x = as_float1d(x)
    if window < 1:
        raise ValueError("window must be >= 1")
    min_p = window if min_periods is None else max(1, int(min_periods))
    out = np.full(len(x), np.nan, dtype=np.float64)
    for t in range(len(x)):
        start = max(0, t - window + 1)
        chunk = x[start : t + 1]
        finite = chunk[np.isfinite(chunk)]
        if finite.size < min_p:
            continue
        out[t] = float(fn(finite))
    return out


def rolling_mean(x: np.ndarray, window: int, *, min_periods: int | None = None) -> np.ndarray:
    return rolling_apply(x, window, lambda a: float(np.mean(a)), min_periods=min_periods)


def rolling_std(x: np.ndarray, window: int, *, min_periods: int | None = None) -> np.ndarray:
    def _std(a: np.ndarray) -> float:
        if a.size < 2:
            return float("nan")
        return float(np.std(a, ddof=1))

    return rolling_apply(x, window, _std, min_periods=min_periods)


def rolling_sum(x: np.ndarray, window: int, *, min_periods: int | None = None) -> np.ndarray:
    return rolling_apply(x, window, lambda a: float(np.sum(a)), min_periods=min_periods)


def zscore(x: np.ndarray, window: int, *, min_periods: int | None = None) -> np.ndarray:
    """Trailing z-score: ``(x[t] - mean_past) / std_past`` (past window only)."""
    x = as_float1d(x)
    mu = rolling_mean(x, window, min_periods=min_periods)
    sd = rolling_std(x, window, min_periods=min_periods)
    out = np.full_like(x, np.nan)
    mask = np.isfinite(x) & np.isfinite(mu) & np.isfinite(sd) & (sd > 1e-12)
    out[mask] = (x[mask] - mu[mask]) / sd[mask]
    return out


def rank(x: np.ndarray, window: int | None = None) -> np.ndarray:
    """Percentile rank in trailing window, or full-sample rank if window is None.

    When ``window`` is set, rank at t uses only ``x[t-window+1:t+1]`` (PIT).
    Full-sample rank is for cross-sectional / research diagnostics only and
    must not be used as a live causal signal without a clear PIT window.
    """
    x = as_float1d(x)
    if window is None:
        out = np.full_like(x, np.nan)
        m = np.isfinite(x)
        if m.sum() == 0:
            return out
        ranks = stats.rankdata(x[m], method="average")
        out[m] = (ranks - 1.0) / max(m.sum() - 1, 1)
        return out

    if window < 1:
        raise ValueError("window must be >= 1")
    out = np.full_like(x, np.nan)
    for t in range(len(x)):
        start = max(0, t - window + 1)
        chunk = x[start : t + 1]
        if not np.isfinite(chunk[-1]):
            continue
        finite_idx = np.where(np.isfinite(chunk))[0]
        if finite_idx.size == 0:
            continue
        vals = chunk[finite_idx]
        ranks = stats.rankdata(vals, method="average")
        # rank of the last finite observation in the window (current bar if finite)
        last_pos = int(np.where(finite_idx == len(chunk) - 1)[0][0]) if np.isfinite(chunk[-1]) else -1
        if last_pos < 0:
            continue
        out[t] = (ranks[last_pos] - 1.0) / max(len(vals) - 1, 1)
    return out


def delay(x: np.ndarray, periods: int = 1) -> np.ndarray:
    """Alias for :func:`lag` (common quant naming)."""
    return lag(x, periods)


def ts_max(x: np.ndarray, window: int) -> np.ndarray:
    return rolling_apply(x, window, lambda a: float(np.max(a)))


def ts_min(x: np.ndarray, window: int) -> np.ndarray:
    return rolling_apply(x, window, lambda a: float(np.min(a)))


def signed_power(x: np.ndarray, power: float = 2.0) -> np.ndarray:
    x = as_float1d(x)
    out = np.full_like(x, np.nan)
    m = np.isfinite(x)
    out[m] = np.sign(x[m]) * np.abs(x[m]) ** power
    return out


def evaluate_expression(
    ops: list[tuple[str, dict]],
    series: dict[str, np.ndarray],
) -> np.ndarray:
    """Evaluate a simple stack of named ops for formula prototyping.

    Each op is ``(op_name, kwargs)``. Supported: lag, diff, ratio, zscore, rank,
    rolling_mean, rolling_std, rolling_sum. ``kwargs`` may reference series names
    via ``input`` / ``numerator`` / ``denominator``.
    """
    stack: list[np.ndarray] = []
    for op_name, kwargs in ops:
        kw = dict(kwargs)
        if op_name == "load":
            name = str(kw["name"])
            stack.append(as_float1d(series[name]))
        elif op_name == "lag":
            stack.append(lag(stack.pop(), int(kw.get("periods", 1))))
        elif op_name == "diff":
            stack.append(diff(stack.pop(), int(kw.get("periods", 1))))
        elif op_name == "zscore":
            stack.append(zscore(stack.pop(), int(kw["window"])))
        elif op_name == "rank":
            stack.append(rank(stack.pop(), kw.get("window")))
        elif op_name == "rolling_mean":
            stack.append(rolling_mean(stack.pop(), int(kw["window"])))
        elif op_name == "rolling_std":
            stack.append(rolling_std(stack.pop(), int(kw["window"])))
        elif op_name == "rolling_sum":
            stack.append(rolling_sum(stack.pop(), int(kw["window"])))
        elif op_name == "ratio":
            den = stack.pop()
            num = stack.pop()
            stack.append(ratio(num, den, eps=float(kw.get("eps", 1e-12))))
        elif op_name == "neg":
            stack.append(-stack.pop())
        else:
            raise ValueError(f"Unknown symbolic op: {op_name}")
    if len(stack) != 1:
        raise ValueError(f"expression stack must end with 1 value, got {len(stack)}")
    return stack[0]

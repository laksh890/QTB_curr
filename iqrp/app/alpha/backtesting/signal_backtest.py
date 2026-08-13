"""Signal-driven long-short / long-only backtests.

Look-ahead prevention
---------------------
* Positions at ``t`` are formed from ``signal[t]`` only.
* PnL uses ``forward_returns[t]`` which must be the return from ``t`` to
  ``t+1`` (or a longer horizon) that is **not** known when the signal is
  formed. If you pass close-to-close returns contemporaneous with the signal,
  set ``returns_are_forward=False`` to shift returns by one bar so
  ``weight[t]`` multiplies ``return[t+1]``.
* No future signal values enter weights.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

Mode = Literal["long_short", "long_only", "sign"]


def _as_1d(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _align_returns(signal: np.ndarray, returns: np.ndarray, *, returns_are_forward: bool) -> tuple[np.ndarray, np.ndarray]:
    n = min(signal.size, returns.size)
    sig = signal[:n].copy()
    ret = returns[:n].copy()
    if returns_are_forward:
        return sig, ret
    # Shift: weight[t] earns return[t+1]; drop last signal bar
    if n < 2:
        return sig[:0], ret[:0]
    return sig[:-1], ret[1:]


def signal_to_weights(
    signal: np.ndarray,
    *,
    mode: Mode = "long_short",
) -> np.ndarray:
    """Map a time-series signal to portfolio exposure in [-1, 1] or [0, 1]."""
    s = signal.astype(np.float64, copy=True)
    w = np.zeros_like(s)
    finite = np.isfinite(s)
    if not np.any(finite):
        return w
    x = s[finite]
    if mode == "sign":
        out = np.sign(x)
        out[out == 0.0] = 0.0
        w[finite] = out
        return w
    if mode == "long_only":
        x = np.maximum(x - np.nanmedian(s), 0.0)
        # rank within positive set via soft ranks of raw signal
        ranks = _rank01(s)
        w = np.where(finite, ranks, 0.0)
        # zero below-median for long-only tilt
        med = float(np.nanmedian(s[finite]))
        w = np.where(s >= med, w, 0.0)
        total = float(np.nansum(w))
        if total > 1e-15:
            w = w / total
        return w
    # long_short: demeaned ranks in [-1, 1]
    ranks = _rank01(s)
    ranks = ranks - 0.5
    # scale to unit gross
    gross = float(np.nansum(np.abs(ranks)))
    if gross > 1e-15:
        ranks = ranks / (gross / 2.0)  # long gross ≈ 1, short gross ≈ 1
    w = np.where(finite, ranks, 0.0)
    return w


def _rank01(s: np.ndarray) -> np.ndarray:
    """Average ranks mapped to (0, 1]; NaNs → NaN."""
    out = np.full(s.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(s)
    n = int(np.sum(finite))
    if n == 0:
        return out
    order = np.argsort(np.argsort(s[finite]))
    out[finite] = (order.astype(np.float64) + 1.0) / float(n)
    return out


def _ann_sharpe(r: np.ndarray, *, periods_per_year: float = 252.0) -> float:
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    sd = float(np.std(r, ddof=1))
    if sd < 1e-15:
        return 0.0 if abs(float(np.mean(r))) < 1e-15 else float("nan")
    return float(np.mean(r) / sd * np.sqrt(periods_per_year))


def signal_backtest(
    signal: Any,
    returns: Any,
    *,
    cost_bps: float = 0.0,
    mode: Mode = "long_short",
    returns_are_forward: bool = True,
    periods_per_year: float = 252.0,
    weights: Any | None = None,
) -> dict[str, Any]:
    """Backtest a 1-D signal against aligned forward returns.

    Parameters
    ----------
    signal, returns:
        Equal-length (or truncated to min length) time series.
    cost_bps:
        One-way transaction cost in basis points applied to turnover
        ``|w_t - w_{t-1}|``.
    mode:
        ``long_short`` (demeaned ranks), ``long_only``, or ``sign``.
    returns_are_forward:
        If False, returns are shifted by +1 to prevent look-ahead.
    """
    sig = _as_1d(signal)
    ret = _as_1d(returns)
    sig, ret = _align_returns(sig, ret, returns_are_forward=returns_are_forward)
    n = int(sig.size)
    if n == 0:
        return {
            "gross_returns": np.asarray([], dtype=np.float64),
            "net_returns": np.asarray([], dtype=np.float64),
            "weights": np.asarray([], dtype=np.float64),
            "turnover": np.asarray([], dtype=np.float64),
            "gross_sharpe": float("nan"),
            "net_sharpe": float("nan"),
            "gross_mean": float("nan"),
            "net_mean": float("nan"),
            "total_cost": 0.0,
            "avg_turnover": 0.0,
            "n": 0,
            "mode": mode,
            "cost_bps": float(cost_bps),
            "look_ahead_guard": "returns_are_forward" if returns_are_forward else "shifted_returns",
        }

    if weights is None:
        w = signal_to_weights(sig, mode=mode)
    else:
        w = _as_1d(weights)[:n]
        if w.size < n:
            ww = np.zeros(n, dtype=np.float64)
            ww[: w.size] = w
            w = ww

    gross = w * ret
    turnover = np.zeros(n, dtype=np.float64)
    turnover[0] = abs(float(w[0]))
    turnover[1:] = np.abs(np.diff(w))
    costs = turnover * (float(cost_bps) / 1e4)
    net = gross - costs

    return {
        "gross_returns": gross,
        "net_returns": net,
        "weights": w,
        "turnover": turnover,
        "gross_sharpe": _ann_sharpe(gross, periods_per_year=periods_per_year),
        "net_sharpe": _ann_sharpe(net, periods_per_year=periods_per_year),
        "gross_mean": float(np.nanmean(gross)),
        "net_mean": float(np.nanmean(net)),
        "total_cost": float(np.nansum(costs)),
        "avg_turnover": float(np.nanmean(turnover)),
        "n": n,
        "mode": mode,
        "cost_bps": float(cost_bps),
        "look_ahead_guard": "returns_are_forward" if returns_are_forward else "shifted_returns",
        "cumulative_net": float(np.nansum(net)),
        "hit_rate": float(np.nanmean(net > 0)) if n else float("nan"),
    }

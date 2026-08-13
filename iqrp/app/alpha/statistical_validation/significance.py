"""IC significance tests (t-test and Newey–West HAC).

Look-ahead prevention
---------------------
``signal`` and ``forward_returns`` must already be time-aligned so that
``signal[t]`` is known strictly before ``forward_returns[t]`` realizes.
Do not pass contemporaneous returns; use lagged signal or forward returns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def _finite_pair(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def _pearson_ic(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    if float(np.std(x)) < 1e-15 or float(np.std(y)) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ic_significance(
    signal: Any,
    forward_returns: Any,
    *,
    alternative: str = "two-sided",
) -> dict[str, float | str | int]:
    """Pearson IC with classical t-test against H0: IC = 0.

    ``t = IC * sqrt((n-2)/(1-IC^2))``, ``df = n-2``.
    """
    x, y = _finite_pair(signal, forward_returns)
    n = int(x.size)
    ic = _pearson_ic(x, y)
    if n < 3 or not np.isfinite(ic) or abs(ic) >= 1.0:
        return {
            "ic": float(ic) if np.isfinite(ic) else float("nan"),
            "t_stat": float("nan"),
            "pvalue": float("nan"),
            "n": n,
            "method": "ttest",
            "alternative": alternative,
            "stderr": float("nan"),
        }
    denom = max(1.0 - ic * ic, 1e-15)
    t_stat = ic * np.sqrt((n - 2) / denom)
    df = n - 2
    if alternative == "greater":
        pvalue = float(1.0 - stats.t.cdf(t_stat, df))
    elif alternative == "less":
        pvalue = float(stats.t.cdf(t_stat, df))
    else:
        pvalue = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df)))
    stderr = float(np.sqrt(denom / max(n - 2, 1)))
    return {
        "ic": float(ic),
        "t_stat": float(t_stat),
        "pvalue": float(np.clip(pvalue, 0.0, 1.0)),
        "n": n,
        "method": "ttest",
        "alternative": alternative,
        "stderr": stderr,
        "df": float(df),
    }


def newey_west_variance(x: np.ndarray, *, lags: int | None = None) -> float:
    """Newey–West long-run variance of a centered series."""
    z = np.asarray(x, dtype=np.float64).reshape(-1)
    z = z[np.isfinite(z)]
    n = z.size
    if n < 2:
        return float("nan")
    z = z - float(np.mean(z))
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** 0.25))
    lags = int(np.clip(lags, 0, max(n - 1, 0)))
    gamma0 = float(np.dot(z, z) / n)
    lrv = gamma0
    for h in range(1, lags + 1):
        w = 1.0 - h / (lags + 1.0)
        gamma = float(np.dot(z[h:], z[:-h]) / n)
        lrv += 2.0 * w * gamma
    return float(max(lrv, 0.0))


def rolling_ic_series(
    signal: Any,
    forward_returns: Any,
    *,
    window: int = 60,
) -> np.ndarray:
    """Rolling Pearson IC series (time-ordered; no shuffle)."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    y = np.asarray(forward_returns, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    x, y = x[:n], y[:n]
    w = max(int(window), 5)
    out = np.full(n, np.nan, dtype=np.float64)
    for t in range(w - 1, n):
        out[t] = _pearson_ic(x[t - w + 1 : t + 1], y[t - w + 1 : t + 1])
    return out


def newey_west_ic_significance(
    signal: Any,
    forward_returns: Any,
    *,
    window: int = 60,
    lags: int | None = None,
    alternative: str = "two-sided",
) -> dict[str, float | str | int]:
    """Test mean rolling IC ≠ 0 using Newey–West HAC standard errors."""
    ics = rolling_ic_series(signal, forward_returns, window=window)
    z = ics[np.isfinite(ics)]
    n = int(z.size)
    if n < 3:
        return {
            "ic": float("nan"),
            "t_stat": float("nan"),
            "pvalue": float("nan"),
            "n": n,
            "method": "newey_west",
            "alternative": alternative,
            "stderr": float("nan"),
            "lags": float(lags or 0),
        }
    mean_ic = float(np.mean(z))
    lrv = newey_west_variance(z, lags=lags)
    se = float(np.sqrt(lrv / n)) if lrv > 0 and np.isfinite(lrv) else float("nan")
    if not np.isfinite(se) or se <= 0:
        t_stat = float("nan")
        pvalue = float("nan")
    else:
        t_stat = mean_ic / se
        if alternative == "greater":
            pvalue = float(1.0 - stats.norm.cdf(t_stat))
        elif alternative == "less":
            pvalue = float(stats.norm.cdf(t_stat))
        else:
            pvalue = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))
        pvalue = float(np.clip(pvalue, 0.0, 1.0))
    return {
        "ic": mean_ic,
        "t_stat": float(t_stat),
        "pvalue": pvalue,
        "n": n,
        "method": "newey_west",
        "alternative": alternative,
        "stderr": se,
        "lags": float(lags if lags is not None else int(np.floor(4.0 * (n / 100.0) ** 0.25))),
        "window": int(window),
    }

"""IID and block bootstrap confidence intervals for IC and Sharpe."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

StatName = Literal["ic", "sharpe"]


def _finite_pair(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def _ic(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    sx, sy = float(np.std(x)), float(np.std(y))
    if sx < 1e-15 or sy < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _sharpe(r: np.ndarray, *, periods_per_year: float = 252.0) -> float:
    if r.size < 2:
        return float("nan")
    sd = float(np.std(r, ddof=1))
    if sd < 1e-15:
        return float("nan")
    return float(np.mean(r) / sd * np.sqrt(periods_per_year))


def _point_stat(
    x: np.ndarray,
    y: np.ndarray | None,
    stat: StatName,
    *,
    periods_per_year: float,
) -> float:
    if stat == "ic":
        if y is None:
            raise ValueError("y (forward returns) required for IC")
        return _ic(x, y)
    return _sharpe(x if y is None else x * y, periods_per_year=periods_per_year)


def iid_bootstrap_ci(
    x: Any,
    y: Any | None = None,
    *,
    stat: StatName = "ic",
    n_boot: int = 1000,
    alpha: float = 0.05,
    periods_per_year: float = 252.0,
    seed: int | None = None,
) -> dict[str, float | int | str | list[float]]:
    """IID bootstrap percentile CI for IC (x,y) or Sharpe (returns in x)."""
    rng = np.random.default_rng(seed)
    if stat == "ic":
        a, b = _finite_pair(x, y if y is not None else x)
    else:
        a = np.asarray(x, dtype=np.float64).reshape(-1)
        a = a[np.isfinite(a)]
        b = None
    n = int(a.size)
    point = _point_stat(a, b, stat, periods_per_year=periods_per_year)
    boots = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        if b is None:
            boots[i] = _point_stat(a[idx], None, stat, periods_per_year=periods_per_year)
        else:
            boots[i] = _point_stat(a[idx], b[idx], stat, periods_per_year=periods_per_year)
    boots = boots[np.isfinite(boots)]
    lo = float(np.quantile(boots, alpha / 2.0)) if boots.size else float("nan")
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0)) if boots.size else float("nan")
    return {
        "stat": stat,
        "method": "iid_bootstrap",
        "estimate": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "alpha": float(alpha),
        "n": n,
        "n_boot": int(n_boot),
        "boot_mean": float(np.mean(boots)) if boots.size else float("nan"),
        "boot_std": float(np.std(boots, ddof=1)) if boots.size > 1 else float("nan"),
    }


def block_bootstrap_ci(
    x: Any,
    y: Any | None = None,
    *,
    stat: StatName = "ic",
    block_size: int = 20,
    n_boot: int = 1000,
    alpha: float = 0.05,
    periods_per_year: float = 252.0,
    seed: int | None = None,
) -> dict[str, float | int | str]:
    """Moving-block bootstrap CI preserving serial dependence."""
    rng = np.random.default_rng(seed)
    if stat == "ic":
        a, b = _finite_pair(x, y if y is not None else x)
    else:
        a = np.asarray(x, dtype=np.float64).reshape(-1)
        a = a[np.isfinite(a)]
        b = None
    n = int(a.size)
    bs = int(np.clip(block_size, 1, max(n, 1)))
    n_blocks = int(np.ceil(n / bs))
    point = _point_stat(a, b, stat, periods_per_year=periods_per_year)
    boots = np.empty(int(n_boot), dtype=np.float64)
    max_start = max(n - bs, 0)
    for i in range(int(n_boot)):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + bs) for s in starts])[:n]
        idx = np.clip(idx, 0, n - 1)
        if b is None:
            boots[i] = _point_stat(a[idx], None, stat, periods_per_year=periods_per_year)
        else:
            boots[i] = _point_stat(a[idx], b[idx], stat, periods_per_year=periods_per_year)
    boots = boots[np.isfinite(boots)]
    lo = float(np.quantile(boots, alpha / 2.0)) if boots.size else float("nan")
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0)) if boots.size else float("nan")
    return {
        "stat": stat,
        "method": "block_bootstrap",
        "estimate": float(point),
        "ci_low": lo,
        "ci_high": hi,
        "alpha": float(alpha),
        "n": n,
        "n_boot": int(n_boot),
        "block_size": bs,
        "boot_mean": float(np.mean(boots)) if boots.size else float("nan"),
        "boot_std": float(np.std(boots, ddof=1)) if boots.size > 1 else float("nan"),
    }

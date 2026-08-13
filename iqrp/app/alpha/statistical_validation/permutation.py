"""Permutation null for information coefficient.

Look-ahead prevention
---------------------
Only ``forward_returns`` are shuffled. The signal time-order is preserved so
the null destroys predictive association without inventing future information
in the feature path.
"""

from __future__ import annotations

from typing import Any

import numpy as np


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
    if float(np.std(x)) < 1e-15 or float(np.std(y)) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def permutation_ic_test(
    signal: Any,
    forward_returns: Any,
    *,
    n_perm: int = 1000,
    alternative: str = "two-sided",
    seed: int | None = None,
) -> dict[str, float | int | str | list[float]]:
    """Permute forward returns to build a null IC distribution."""
    x, y = _finite_pair(signal, forward_returns)
    n = int(x.size)
    obs = _ic(x, y)
    rng = np.random.default_rng(seed)
    null = np.empty(int(n_perm), dtype=np.float64)
    for i in range(int(n_perm)):
        yp = rng.permutation(y)
        null[i] = _ic(x, yp)
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or null.size == 0:
        pvalue = float("nan")
    elif alternative == "greater":
        pvalue = float(np.mean(null >= obs))
    elif alternative == "less":
        pvalue = float(np.mean(null <= obs))
    else:
        pvalue = float(np.mean(np.abs(null) >= abs(obs)))
    return {
        "ic": float(obs) if np.isfinite(obs) else float("nan"),
        "pvalue": float(np.clip(pvalue, 0.0, 1.0)) if np.isfinite(pvalue) else float("nan"),
        "n": n,
        "n_perm": int(n_perm),
        "null_mean": float(np.mean(null)) if null.size else float("nan"),
        "null_std": float(np.std(null, ddof=1)) if null.size > 1 else float("nan"),
        "null_ci_low": float(np.quantile(null, 0.025)) if null.size else float("nan"),
        "null_ci_high": float(np.quantile(null, 0.975)) if null.size else float("nan"),
        "method": "permutation_ic",
        "alternative": alternative,
    }

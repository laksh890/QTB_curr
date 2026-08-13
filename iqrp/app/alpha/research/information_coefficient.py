"""Information coefficient wrappers for alpha research.

May import ``iqrp.app.features.research._numeric.information_coefficient``.

CRITICAL:
- IC measures association, not economic alpha.
- Statistical significance alone ≠ alpha.
- Historical Sharpe alone cannot approve.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.features.research._numeric import information_coefficient, safe_nanmean


def compute_ic(signal: np.ndarray, forward_returns: np.ndarray) -> float:
    """Pearson information coefficient between signal and forward returns."""
    x = np.asarray(signal, dtype=np.float64)
    y = np.asarray(forward_returns, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError(f"length mismatch: signal={len(x)} forward={len(y)}")
    return float(information_coefficient(x, y))


def rolling_ic(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    *,
    window: int = 60,
    step: int = 1,
    min_obs: int = 20,
) -> np.ndarray:
    """Rolling IC series (research diagnostic)."""
    x = np.asarray(signal, dtype=np.float64)
    y = np.asarray(forward_returns, dtype=np.float64)
    if len(x) != len(y):
        raise ValueError("length mismatch")
    if window < 3:
        raise ValueError("window must be >= 3")
    out: list[float] = []
    for start in range(0, max(0, len(x) - window + 1), step):
        sl = slice(start, start + window)
        xs, ys = x[sl], y[sl]
        if np.isfinite(xs).sum() < min_obs:
            out.append(float("nan"))
            continue
        out.append(compute_ic(xs, ys))
    return np.asarray(out, dtype=np.float64)


def ic_summary(
    signal: np.ndarray,
    forward_returns: np.ndarray,
    *,
    window: int = 60,
    step: int = 20,
) -> dict[str, Any]:
    """Summary IC statistics. Not an approval metric."""
    ic = compute_ic(signal, forward_returns)
    roll = rolling_ic(signal, forward_returns, window=window, step=step)
    finite = roll[np.isfinite(roll)]
    return {
        "ic": ic,
        "rolling_ic_mean": safe_nanmean(roll),
        "rolling_ic_std": float(np.std(finite)) if finite.size else float("nan"),
        "rolling_ic_ir": (
            float(np.mean(finite) / (np.std(finite) + 1e-12)) if finite.size else float("nan")
        ),
        "n_rolling": int(finite.size),
        "disclaimer": "IC ≠ alpha. Statistical significance alone ≠ alpha.",
    }

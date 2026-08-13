"""Historical mean expected returns.

Research-only helper. Prefer forecast-derived expected returns for live
portfolio construction — historical means invent persistence that may not
exist out of sample.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__VERSION__ = "1.0.0"


def _as_matrix(returns: Any) -> np.ndarray:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("returns must be 1-D or 2-D (T x N)")
    return arr


def historical_expected_returns(
    returns: Any,
    *,
    window: int | None = None,
    annualization: float = 1.0,
    names: Sequence[str] | None = None,
    version: str = __VERSION__,
) -> dict[str, Any]:
    """Sample mean returns.

    Documented for research / backtests only. Prefer
    ``forecast_expected_returns`` for live construction so that forecast
    uncertainty is not replaced by historical certainty.
    """
    x = _as_matrix(returns)
    t, n = x.shape
    if window is not None and window > 0:
        w = min(int(window), t)
        x = x[-w:]
        t = w

    if t == 0 or n == 0:
        mu = np.zeros(max(n, 0), dtype=np.float64)
        n_obs = 0
    else:
        mask = np.all(np.isfinite(x), axis=1)
        clean = x[mask]
        n_obs = int(clean.shape[0])
        if n_obs == 0:
            mu = np.zeros(n, dtype=np.float64)
        else:
            mu = np.nanmean(clean, axis=0) * float(annualization)
            mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "name": "historical_expected_returns",
        "method": "historical_mean",
        "mu": mu.tolist(),
        "vector": mu.tolist(),
        "shape": [int(mu.size)],
        "n_obs": int(n_obs),
        "window": window,
        "annualization": float(annualization),
        "names": list(names) if names is not None else None,
        "research_only": True,
        "warning": "Prefer forecast_expected_returns for live portfolios",
        "version": version,
    }

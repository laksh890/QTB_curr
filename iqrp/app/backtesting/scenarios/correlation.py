"""Correlation shock scenarios."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["apply_correlation_shock", "run_correlation_scenario", "stress_correlation"]


def stress_correlation(
    cov: Any,
    *,
    shift: float = 0.5,
) -> np.ndarray:
    """Blend covariance toward perfect correlation (``shift>0``) or independence.

    ``shift`` in [-1, 1].
    """
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be square")
    vol = np.sqrt(np.clip(np.diag(c), 0.0, None))
    perfect = np.outer(vol, vol)
    indep = np.diag(np.diag(c))
    alpha = float(np.clip(shift, -1.0, 1.0))
    if alpha >= 0:
        out = (1.0 - alpha) * c + alpha * perfect
    else:
        out = (1.0 + alpha) * c + (-alpha) * indep
    return 0.5 * (out + out.T)


def apply_correlation_shock(
    returns: Any,
    *,
    shift: float = 0.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Re-simulate returns under a shocked correlation structure."""
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] < 2:
        raise ValueError("correlation shock requires multivariate returns (T, N>=2)")
    mu = np.mean(r, axis=0)
    cov = np.cov(r, rowvar=False)
    stressed_cov = stress_correlation(cov, shift=shift)
    rng = np.random.default_rng(int(seed))
    jitter = 1e-10 * np.eye(stressed_cov.shape[0])
    try:
        sim = rng.multivariate_normal(mu, stressed_cov + jitter, size=r.shape[0])
    except np.linalg.LinAlgError:
        sd = np.sqrt(np.clip(np.diag(stressed_cov), 1e-12, None))
        sim = rng.normal(mu, sd, size=r.shape)
    return {
        "name": "correlation",
        "kind": "correlation",
        "shift": float(shift),
        "cov": stressed_cov,
        "returns": sim,
    }


def run_correlation_scenario(
    returns: Any,
    *,
    shifts: list[float] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Grid of correlation shifts."""
    grid = [-0.5, 0.0, 0.5, 1.0] if shifts is None else list(shifts)
    results = []
    for s in grid:
        out = apply_correlation_shock(returns, shift=float(s), seed=seed)
        port = np.mean(out["returns"], axis=1)
        corr = out["cov"] / np.outer(
            np.sqrt(np.clip(np.diag(out["cov"]), 1e-12, None)),
            np.sqrt(np.clip(np.diag(out["cov"]), 1e-12, None)),
        )
        off = corr - np.eye(corr.shape[0])
        results.append(
            {
                "shift": float(s),
                "port_vol": float(np.std(port, ddof=1)),
                "mean_abs_corr": float(np.mean(np.abs(off))),
            }
        )
    return {"name": "correlation_grid", "kind": "correlation", "results": results}

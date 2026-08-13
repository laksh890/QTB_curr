"""Parametric and correlated Monte Carlo simulation."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import as_returns


def parametric_monte_carlo(
    returns: Any,
    *,
    n_simulations: int = 5000,
    horizon: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate i.i.d. normal paths calibrated to historical mean/vol."""
    r = as_returns(returns)
    n_sim = max(int(n_simulations), 1)
    h = max(int(horizon), 1)
    rng = np.random.default_rng(int(seed))
    if r.size == 0:
        mu, sigma = 0.0, 0.0
        paths = np.zeros((n_sim, h), dtype=np.float64)
    else:
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
        paths = rng.normal(mu, max(sigma, 1e-12), size=(n_sim, h))

    terminal = paths.sum(axis=1)
    return {
        "name": "parametric_monte_carlo",
        "paths": paths,
        "terminal": terminal,
        "mu": mu,
        "sigma": sigma,
        "n_simulations": n_sim,
        "horizon": h,
        "seed": int(seed),
        "mean_terminal": float(np.mean(terminal)),
        "std_terminal": float(np.std(terminal, ddof=1)) if terminal.size > 1 else 0.0,
    }


def correlated_monte_carlo(
    mean: Any,
    cov: Any,
    *,
    n_simulations: int = 5000,
    horizon: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate correlated multivariate normal returns over a horizon.

    Returns shape (n_simulations, horizon, n_assets) and terminal sums.
    """
    mu = np.asarray(mean, dtype=np.float64).reshape(-1)
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be square")
    n = c.shape[0]
    if mu.size != n:
        tmp = np.zeros(n, dtype=np.float64)
        m = min(n, mu.size)
        tmp[:m] = mu[:m]
        mu = tmp

    n_sim = max(int(n_simulations), 1)
    h = max(int(horizon), 1)
    rng = np.random.default_rng(int(seed))

    # PSD via eigen clip
    c_sym = 0.5 * (c + c.T)
    eigvals, eigvecs = np.linalg.eigh(c_sym)
    eigvals = np.maximum(eigvals, 0.0)
    c_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    draws = rng.multivariate_normal(mu, c_psd, size=(n_sim, h))
    terminal = draws.sum(axis=1)  # (n_sim, n)

    return {
        "name": "correlated_monte_carlo",
        "paths": draws,
        "terminal": terminal,
        "mean": mu.tolist(),
        "n_simulations": n_sim,
        "horizon": h,
        "n_assets": n,
        "seed": int(seed),
    }

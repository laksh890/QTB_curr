"""Gaussian copula simulation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from iqrp.app.risk.base import as_returns


def gaussian_copula_simulate(
    returns: Any,
    *,
    n_simulations: int = 5000,
    seed: int = 42,
    correlation: Any | None = None,
) -> dict[str, Any]:
    """Simulate dependent returns via Gaussian copula + empirical margins.

    ``returns`` is (T, N). Margins are inverted from the empirical CDF of each
    column (past data only). Dependence uses the sample correlation unless
    ``correlation`` is provided.
    """
    x = np.asarray(returns, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError("returns must be 1-D or 2-D")

    t, n = x.shape
    n_sim = max(int(n_simulations), 1)
    rng = np.random.default_rng(int(seed))

    if t < 2 or n == 0:
        return {
            "name": "gaussian_copula_simulate",
            "samples": np.zeros((n_sim, max(n, 0)), dtype=np.float64),
            "n_simulations": n_sim,
            "n_assets": n,
            "seed": int(seed),
        }

    clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if correlation is not None:
        corr = np.asarray(correlation, dtype=np.float64)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.corrcoef(clean, rowvar=False)
        if corr.ndim == 0:
            corr = np.array([[1.0]])
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

    corr = 0.5 * (corr + corr.T)
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-10)
    corr_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    z = rng.multivariate_normal(np.zeros(n), corr_psd, size=n_sim)
    u = stats.norm.cdf(z)
    u = np.clip(u, 1e-6, 1.0 - 1e-6)

    samples = np.empty((n_sim, n), dtype=np.float64)
    for j in range(n):
        col = np.sort(clean[:, j])
        # Empirical quantile function
        positions = u[:, j] * (t - 1)
        lo = np.floor(positions).astype(int)
        hi = np.ceil(positions).astype(int)
        w = positions - lo
        samples[:, j] = (1.0 - w) * col[lo] + w * col[hi]

    return {
        "name": "gaussian_copula_simulate",
        "samples": samples,
        "correlation": corr_psd.tolist(),
        "n_simulations": n_sim,
        "n_assets": n,
        "n_obs": t,
        "seed": int(seed),
    }

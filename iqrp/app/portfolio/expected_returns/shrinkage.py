"""James–Stein / grand-mean shrinkage for expected returns."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from iqrp.app.portfolio.expected_returns.historical import historical_expected_returns

__VERSION__ = "1.0.0"


def james_stein_shrinkage(
    mu: Sequence[float] | np.ndarray,
    *,
    prior: Sequence[float] | np.ndarray | None = None,
    intensity: float | None = None,
    cov: Sequence[Sequence[float]] | np.ndarray | None = None,
    n_obs: int | None = None,
    names: Sequence[str] | None = None,
    version: str = __VERSION__,
) -> dict[str, Any]:
    """Shrink a mean vector toward a grand-mean (or provided) prior.

    When ``intensity`` is omitted and ``cov`` / ``n_obs`` are available, uses a
    James–Stein style factor ``(N-2) / (T * ||mu - prior||^2_Sigma^{-1})``
    clipped to [0, 1]. Otherwise shrinks halfway toward the prior (0.5).
    """
    m = np.asarray(mu, dtype=np.float64).reshape(-1)
    n = int(m.size)
    if prior is None:
        grand = float(np.mean(m)) if n else 0.0
        p = np.full(n, grand, dtype=np.float64)
        prior_method = "grand_mean"
    else:
        p = np.asarray(prior, dtype=np.float64).reshape(-1)
        if p.size != n:
            raise ValueError(f"prior length {p.size} != mu length {n}")
        prior_method = "provided"

    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)

    if intensity is not None:
        alpha = float(np.clip(intensity, 0.0, 1.0))
        intensity_method = "provided"
    elif cov is not None and n_obs is not None and n_obs > 0 and n > 2:
        Sigma = np.asarray(cov, dtype=np.float64)
        if Sigma.shape != (n, n):
            raise ValueError(f"cov shape {Sigma.shape} incompatible with mu length {n}")
        diff = m - p
        try:
            inv = np.linalg.pinv(Sigma + 1e-12 * np.eye(n))
            quad = float(diff @ inv @ diff)
        except np.linalg.LinAlgError:
            quad = float(np.dot(diff, diff))
        if quad <= 1e-18:
            alpha = 1.0
        else:
            alpha = float(np.clip((n - 2) / (float(n_obs) * quad), 0.0, 1.0))
        intensity_method = "james_stein"
    else:
        alpha = 0.5
        intensity_method = "default_half"

    shrunk = (1.0 - alpha) * m + alpha * p
    return {
        "name": "james_stein_shrinkage",
        "method": "james_stein_grand_mean",
        "mu": shrunk.tolist(),
        "vector": shrunk.tolist(),
        "shape": [n],
        "n_obs": int(n_obs) if n_obs is not None else n,
        "intensity": alpha,
        "prior": p.tolist(),
        "raw_mu": m.tolist(),
        "prior_method": prior_method,
        "intensity_method": intensity_method,
        "names": list(names) if names is not None else None,
        "version": version,
    }


def shrinkage_expected_returns(
    returns: Any | None = None,
    *,
    mu: Sequence[float] | np.ndarray | None = None,
    prior: Sequence[float] | np.ndarray | None = None,
    intensity: float | None = None,
    cov: Sequence[Sequence[float]] | np.ndarray | None = None,
    window: int | None = None,
    names: Sequence[str] | None = None,
    version: str = __VERSION__,
) -> dict[str, Any]:
    """James–Stein / grand-mean shrinkage of expected returns.

    Provide either a mean vector ``mu`` or asset ``returns`` (historical mean
    is computed first — research path only).
    """
    n_obs: int | None = None
    if mu is None:
        if returns is None:
            raise ValueError("Provide mu or returns")
        hist = historical_expected_returns(returns, window=window, names=names, version=version)
        mu_vec = np.asarray(hist["mu"], dtype=np.float64)
        n_obs = int(hist["n_obs"])
        source = "historical"
    else:
        mu_vec = np.asarray(mu, dtype=np.float64).reshape(-1)
        source = "provided_mu"

    out = james_stein_shrinkage(
        mu_vec,
        prior=prior,
        intensity=intensity,
        cov=cov,
        n_obs=n_obs,
        names=names,
        version=version,
    )
    out["name"] = "shrinkage_expected_returns"
    out["source"] = source
    return out

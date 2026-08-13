"""Multi-step forecasting from particle clouds."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.regimes.particle.particle import ParticleCloud
from iqrp.app.regimes.particle.propagation import TransitionModel, propagate_cloud


def forecast_particles(
    cloud: ParticleCloud,
    model: TransitionModel,
    *,
    horizon: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[ParticleCloud]]:
    """
    Propagate particles ``horizon`` steps without observation updates.

    Returns ``(means (H,d), covs (H,d,d), clouds)``.
    """
    h = max(1, int(horizon))
    cur = cloud
    means = np.empty((h, cloud.dim), dtype=np.float64)
    covs = np.empty((h, cloud.dim, cloud.dim), dtype=np.float64)
    clouds: list[ParticleCloud] = []
    for t in range(h):
        cur = propagate_cloud(cur, model, rng=rng, t=t)
        means[t] = cur.mean()
        covs[t] = cur.covariance()
        clouds.append(cur)
    return means, covs, clouds


def credible_interval(
    cloud: ParticleCloud,
    *,
    level: float = 0.95,
    dim: int = 0,
) -> tuple[float, float]:
    """Weighted quantile credible interval for a state dimension."""
    w = cloud.weights
    x = cloud.states[:, int(dim)]
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cdf = np.cumsum(ws)
    cdf = cdf / max(float(cdf[-1]), 1e-300)
    alpha = (1.0 - level) / 2.0
    lo = float(np.interp(alpha, cdf, xs))
    hi = float(np.interp(1.0 - alpha, cdf, xs))
    return lo, hi


def posterior_summary(cloud: ParticleCloud, *, level: float = 0.95) -> dict[str, Any]:
    mean = cloud.mean()
    cov = cloud.covariance()
    intervals = [credible_interval(cloud, level=level, dim=i) for i in range(cloud.dim)]
    return {
        "mean": mean,
        "covariance": cov,
        "std": np.sqrt(np.clip(np.diag(cov), 0, None)),
        "credible_intervals": intervals,
        "ess": cloud.ess(),
        "n_particles": cloud.n_particles,
    }


def particle_diversity(cloud: ParticleCloud) -> float:
    """Fraction of unique states (rounded) among particles."""
    rounded = np.round(cloud.states, decimals=8)
    # view as void for unique rows
    uniq = np.unique(rounded, axis=0)
    return float(uniq.shape[0] / max(cloud.n_particles, 1))

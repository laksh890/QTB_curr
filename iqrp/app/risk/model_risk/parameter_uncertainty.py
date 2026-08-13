"""Parameter uncertainty via sampling variability."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_returns


def parameter_uncertainty(
    returns: Any,
    *,
    n_bootstrap: int = 500,
    seed: int = 42,
    statistic: str = "mean",
) -> RiskMeasure:
    """Bootstrap standard error of a simple sample statistic (mean or vol).

    Uses resampling of past returns only.
    """
    r = as_returns(returns)
    n_boot = max(int(n_bootstrap), 10)
    rng = np.random.default_rng(int(seed))
    stat = str(statistic).lower()

    if r.size < 2:
        return RiskMeasure(
            name="parameter_uncertainty",
            value=0.0,
            unit="stderr",
            method="bootstrap_se",
            parameters={"n_obs": int(r.size), "n_bootstrap": n_boot, "statistic": stat},
        )

    n = r.size
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = r[idx]
    if stat in ("vol", "std", "volatility", "sigma"):
        stats = np.std(samples, axis=1, ddof=1)
        point = float(np.std(r, ddof=1))
    else:
        stats = np.mean(samples, axis=1)
        point = float(np.mean(r))
        stat = "mean"

    se = float(np.std(stats, ddof=1))
    return RiskMeasure(
        name="parameter_uncertainty",
        value=se,
        unit="stderr",
        method="bootstrap_se",
        parameters={
            "n_obs": n,
            "n_bootstrap": n_boot,
            "statistic": stat,
            "point_estimate": point,
            "seed": int(seed),
            "ci_low": float(np.quantile(stats, 0.025)),
            "ci_high": float(np.quantile(stats, 0.975)),
        },
    )

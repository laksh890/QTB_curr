"""Confidence interval constructors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from iqrp.app.math._array import as_vector
from iqrp.app.math.probability.sampling import bootstrap_sample


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    low: float
    high: float
    level: float
    method: str
    estimate: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "low": self.low,
            "high": self.high,
            "level": self.level,
            "method": self.method,
            "estimate": self.estimate,
        }


def normal_ci(x: Any, *, level: float = 0.95) -> ConfidenceInterval:
    v = as_vector(x)
    n = len(v)
    m = float(np.mean(v))
    se = float(np.std(v, ddof=1) / np.sqrt(max(n, 1)))
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    return ConfidenceInterval(m - z * se, m + z * se, level, "normal", m)


def bootstrap_ci(
    x: Any,
    *,
    level: float = 0.95,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> ConfidenceInterval:
    v = as_vector(x)
    stats_boot = bootstrap_sample(
        v, n_bootstrap=n_bootstrap, statistic=lambda a: float(np.mean(a)), rng=rng
    )
    alpha = 1.0 - level
    low, high = np.quantile(stats_boot, [alpha / 2, 1 - alpha / 2])
    return ConfidenceInterval(float(low), float(high), level, "bootstrap", float(np.mean(v)))


def wilson_ci(successes: int, n: int, *, level: float = 0.95) -> ConfidenceInterval:
    """Wilson score interval for binomial proportion."""
    if n <= 0:
        return ConfidenceInterval(0.0, 1.0, level, "wilson", 0.0)
    z = float(stats.norm.ppf(0.5 + level / 2.0))
    phat = successes / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return ConfidenceInterval(
        float(max(0.0, center - margin)),
        float(min(1.0, center + margin)),
        level,
        "wilson",
        float(phat),
    )


def bayesian_ci(
    successes: int,
    n: int,
    *,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    level: float = 0.95,
) -> ConfidenceInterval:
    """Equal-tail Beta posterior credible interval (Beta-Binomial)."""
    a = alpha_prior + successes
    b = beta_prior + (n - successes)
    low = float(stats.beta.ppf((1 - level) / 2, a, b))
    high = float(stats.beta.ppf(0.5 + level / 2, a, b))
    mean = float(a / (a + b))
    return ConfidenceInterval(low, high, level, "bayesian", mean)


def ci(
    x: Any,
    *,
    method: Literal["normal", "bootstrap", "wilson", "bayesian"] = "normal",
    level: float = 0.95,
    **kwargs: Any,
) -> ConfidenceInterval:
    if method == "normal":
        return normal_ci(x, level=level)
    if method == "bootstrap":
        return bootstrap_ci(x, level=level, **kwargs)
    if method == "wilson":
        v = as_vector(x)
        return wilson_ci(int(np.sum(v)), len(v), level=level)
    if method == "bayesian":
        v = as_vector(x)
        return bayesian_ci(int(np.sum(v)), len(v), level=level, **kwargs)
    raise ValueError(f"Unknown CI method '{method}'")

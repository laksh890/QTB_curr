"""Prior distributions for Bayesian regime-switching models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from iqrp.app.math.probability.distributions import (
    SciPyDistribution,
    dirichlet,
    gamma,
    gaussian,
    multivariate_gaussian,
)
from iqrp.app.regimes.bayesian.config import PriorsConfig


class Prior(Protocol):
    def logpdf(self, x: Any) -> float | np.ndarray: ...

    def rvs(self, rng: np.random.Generator | None = None) -> np.ndarray: ...


def beta_prior(a: float = 1.0, b: float = 1.0) -> SciPyDistribution:
    from iqrp.app.math.probability.distributions import beta

    return beta(a=a, b=b)


def inverse_gamma_prior(shape: float = 2.0, scale: float = 1.0) -> SciPyDistribution:
    """Inverse-Gamma(shape, scale) with SciPy parameterization ``invgamma(a, scale=b)``."""
    from iqrp.app.math.probability.distributions import DistMeta

    a = max(float(shape), 1e-6)
    b = max(float(scale), 1e-12)
    return SciPyDistribution(
        stats.invgamma(a, scale=b), DistMeta("inverse_gamma", "positive", ("shape", "scale"))
    )


def wishart_prior(df: float, scale: np.ndarray) -> SciPyDistribution:
    from iqrp.app.math.probability.distributions import DistMeta

    scale_arr = np.asarray(scale, dtype=np.float64)
    d = scale_arr.shape[0]
    df_use = max(float(df), float(d))
    return SciPyDistribution(
        stats.wishart(df=df_use, scale=scale_arr),
        DistMeta("wishart", "SPD", ("df", "scale")),
    )


def normal_prior(mu: float = 0.0, sigma: float = 1.0) -> SciPyDistribution:
    return gaussian(mu, sigma)


def mvn_prior(mean: Any, cov: Any) -> SciPyDistribution:
    return multivariate_gaussian(mean, cov)


def dirichlet_prior(alpha: Any) -> SciPyDistribution:
    return dirichlet(alpha)


def gamma_prior(a: float = 2.0, scale: float = 1.0) -> SciPyDistribution:
    return gamma(a=a, scale=scale)


@dataclass
class UserDefinedPrior:
    """Callable prior with optional sampler."""

    log_density: Callable[[Any], float]
    sampler: Callable[[np.random.Generator], np.ndarray] | None = None

    def logpdf(self, x: Any) -> float:
        return float(self.log_density(x))

    def rvs(self, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or np.random.default_rng()
        if self.sampler is None:
            raise ValueError("UserDefinedPrior has no sampler")
        return np.asarray(self.sampler(rng), dtype=np.float64)


@dataclass
class ModelPriors:
    """Collection of conjugate priors for Bayesian HMM / Markov switching."""

    transition_alpha: np.ndarray
    initial_alpha: np.ndarray
    mean_loc: np.ndarray
    mean_strength: float
    invgamma_shape: float
    invgamma_scale: float
    wishart_df: float
    wishart_scale: np.ndarray
    user_priors: dict[str, UserDefinedPrior] | None = None

    @classmethod
    def from_config(
        cls,
        config: PriorsConfig,
        n_states: int,
        n_features: int,
        *,
        user_priors: dict[str, UserDefinedPrior] | None = None,
    ) -> ModelPriors:
        k = int(n_states)
        d = int(n_features)
        ta = np.full((k, k), float(config.transition_alpha), dtype=np.float64)
        ia = np.full(k, float(config.initial_alpha), dtype=np.float64)
        mean_loc = np.full((k, d), float(config.mean_prior_location), dtype=np.float64)
        scale = float(config.wishart_scale) * np.eye(d)
        return cls(
            transition_alpha=ta,
            initial_alpha=ia,
            mean_loc=mean_loc,
            mean_strength=float(config.mean_prior_strength),
            invgamma_shape=float(config.invgamma_shape),
            invgamma_scale=float(config.invgamma_scale),
            wishart_df=float(config.wishart_df),
            wishart_scale=scale,
            user_priors=user_priors,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "transition_alpha": self.transition_alpha.tolist(),
            "initial_alpha": self.initial_alpha.tolist(),
            "mean_loc": self.mean_loc.tolist(),
            "mean_strength": self.mean_strength,
            "invgamma_shape": self.invgamma_shape,
            "invgamma_scale": self.invgamma_scale,
            "wishart_df": self.wishart_df,
            "wishart_scale": self.wishart_scale.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelPriors:
        return cls(
            transition_alpha=np.asarray(data["transition_alpha"], dtype=np.float64),
            initial_alpha=np.asarray(data["initial_alpha"], dtype=np.float64),
            mean_loc=np.asarray(data["mean_loc"], dtype=np.float64),
            mean_strength=float(data["mean_strength"]),
            invgamma_shape=float(data["invgamma_shape"]),
            invgamma_scale=float(data["invgamma_scale"]),
            wishart_df=float(data["wishart_df"]),
            wishart_scale=np.asarray(data["wishart_scale"], dtype=np.float64),
        )


def sample_dirichlet_rows(alpha: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    a = np.asarray(alpha, dtype=np.float64)
    out = np.empty_like(a)
    for i in range(a.shape[0]):
        out[i] = rng.dirichlet(np.clip(a[i], 1e-6, None))
    return out


def sample_invgamma(shape: float, scale: float, size: int, rng: np.random.Generator) -> np.ndarray:
    # InvGamma(shape, scale) via 1/Gamma(shape, scale=1/scale)  [rate parameterization]
    a = max(shape, 1e-6)
    b = max(scale, 1e-12)
    return 1.0 / rng.gamma(a, 1.0 / b, size=size)


def sample_wishart(df: float, scale: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    scale_arr = np.asarray(scale, dtype=np.float64)
    d = scale_arr.shape[0]
    df_use = max(float(df), float(d))
    # Bartlett decomposition
    chol = np.linalg.cholesky(scale_arr + 1e-12 * np.eye(d))
    a = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        a[i, i] = np.sqrt(rng.chisquare(df_use - i))
        for j in range(i):
            a[i, j] = rng.normal()
    m = chol @ a
    return m @ m.T

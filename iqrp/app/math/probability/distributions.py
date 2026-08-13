"""Probability distribution primitives (SciPy-backed, NumPy-compatible)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats  # type: ignore[import-untyped]

from iqrp.app.math._array import as_array, as_vector


@dataclass(frozen=True, slots=True)
class DistMeta:
    name: str
    support: str
    parameters: tuple[str, ...]


class Distribution(ABC):
    """Common interface for univariate / multivariate distributions."""

    meta: DistMeta

    @abstractmethod
    def pdf(self, x: Any) -> np.ndarray:
        """Density or mass function."""

    @abstractmethod
    def logpdf(self, x: Any) -> np.ndarray:
        """Log density / mass."""

    @abstractmethod
    def cdf(self, x: Any) -> np.ndarray:
        """CDF."""

    @abstractmethod
    def rvs(
        self, size: int | tuple[int, ...] = 1, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Random variates."""

    def mean(self) -> float | np.ndarray:
        raise NotImplementedError

    def var(self) -> float | np.ndarray:
        raise NotImplementedError


class SciPyDistribution(Distribution):
    """Thin wrapper around a SciPy frozen distribution."""

    def __init__(self, frozen: Any, meta: DistMeta) -> None:
        self._dist = frozen
        self.meta = meta

    def pdf(self, x: Any) -> np.ndarray:
        x_arr = as_array(x)
        out = self._dist.pdf(x_arr) if hasattr(self._dist, "pdf") else self._dist.pmf(x_arr)
        return np.atleast_1d(np.asarray(out, dtype=np.float64))

    def logpdf(self, x: Any) -> np.ndarray:
        x_arr = as_array(x)
        out = (
            self._dist.logpdf(x_arr) if hasattr(self._dist, "logpdf") else self._dist.logpmf(x_arr)
        )
        return np.atleast_1d(np.asarray(out, dtype=np.float64))

    def cdf(self, x: Any) -> np.ndarray:
        return np.asarray(self._dist.cdf(as_array(x)), dtype=np.float64)

    def rvs(
        self, size: int | tuple[int, ...] = 1, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        # SciPy uses RandomState-like; seed via numpy Generator bit generator
        seed = None if rng is None else int(rng.integers(0, 2**31 - 1))
        return np.asarray(self._dist.rvs(size=size, random_state=seed), dtype=np.float64)

    def mean(self) -> float | np.ndarray:
        return np.asarray(self._dist.mean(), dtype=np.float64)

    def var(self) -> float | np.ndarray:
        return np.asarray(self._dist.var(), dtype=np.float64)

    def ppf(self, q: Any) -> np.ndarray:
        return np.asarray(self._dist.ppf(as_array(q)), dtype=np.float64)


def gaussian(mu: float = 0.0, sigma: float = 1.0) -> SciPyDistribution:
    return SciPyDistribution(
        stats.norm(loc=mu, scale=sigma),
        DistMeta("gaussian", "real", ("mu", "sigma")),
    )


def multivariate_gaussian(
    mean: Any,
    cov: Any,
) -> SciPyDistribution:
    mean_arr = as_vector(mean)
    cov_arr = as_array(cov)
    return SciPyDistribution(
        stats.multivariate_normal(mean=mean_arr, cov=cov_arr, allow_singular=True),
        DistMeta("multivariate_gaussian", "R^d", ("mean", "cov")),
    )


def student_t(df: float = 5.0, mu: float = 0.0, sigma: float = 1.0) -> SciPyDistribution:
    return SciPyDistribution(
        stats.t(df=df, loc=mu, scale=sigma),
        DistMeta("student_t", "real", ("df", "mu", "sigma")),
    )


def bernoulli(p: float = 0.5) -> SciPyDistribution:
    return SciPyDistribution(
        stats.bernoulli(p=p),
        DistMeta("bernoulli", "{0,1}", ("p",)),
    )


def binomial(n: int = 10, p: float = 0.5) -> SciPyDistribution:
    return SciPyDistribution(
        stats.binom(n=n, p=p),
        DistMeta("binomial", "0..n", ("n", "p")),
    )


def poisson(mu: float = 1.0) -> SciPyDistribution:
    return SciPyDistribution(
        stats.poisson(mu=mu),
        DistMeta("poisson", "N0", ("mu",)),
    )


def exponential(scale: float = 1.0) -> SciPyDistribution:
    return SciPyDistribution(
        stats.expon(scale=scale),
        DistMeta("exponential", "R+", ("scale",)),
    )


def gamma(a: float = 2.0, scale: float = 1.0) -> SciPyDistribution:
    return SciPyDistribution(
        stats.gamma(a=a, scale=scale),
        DistMeta("gamma", "R+", ("a", "scale")),
    )


def beta(a: float = 2.0, b: float = 2.0) -> SciPyDistribution:
    return SciPyDistribution(
        stats.beta(a=a, b=b),
        DistMeta("beta", "[0,1]", ("a", "b")),
    )


def dirichlet(alpha: Any) -> SciPyDistribution:
    alpha_arr = as_vector(alpha)
    return SciPyDistribution(
        stats.dirichlet(alpha=alpha_arr),
        DistMeta("dirichlet", "simplex", ("alpha",)),
    )


def uniform(low: float = 0.0, high: float = 1.0) -> SciPyDistribution:
    return SciPyDistribution(
        stats.uniform(loc=low, scale=high - low),
        DistMeta("uniform", "[low,high]", ("low", "high")),
    )


def laplace(mu: float = 0.0, b: float = 1.0) -> SciPyDistribution:
    return SciPyDistribution(
        stats.laplace(loc=mu, scale=b),
        DistMeta("laplace", "real", ("mu", "b")),
    )


def lognormal(mu: float = 0.0, sigma: float = 1.0) -> SciPyDistribution:
    return SciPyDistribution(
        stats.lognorm(s=sigma, scale=np.exp(mu)),
        DistMeta("lognormal", "R+", ("mu", "sigma")),
    )


def chi_square(df: float = 2.0) -> SciPyDistribution:
    return SciPyDistribution(
        stats.chi2(df=df),
        DistMeta("chi_square", "R+", ("df",)),
    )


def f_distribution(dfn: float = 5.0, dfd: float = 10.0) -> SciPyDistribution:
    return SciPyDistribution(
        stats.f(dfn=dfn, dfd=dfd),
        DistMeta("f", "R+", ("dfn", "dfd")),
    )


def weibull(c: float = 1.5, scale: float = 1.0) -> SciPyDistribution:
    return SciPyDistribution(
        stats.weibull_min(c=c, scale=scale),
        DistMeta("weibull", "R+", ("c", "scale")),
    )


def cauchy(mu: float = 0.0, gamma: float = 1.0) -> SciPyDistribution:
    return SciPyDistribution(
        stats.cauchy(loc=mu, scale=gamma),
        DistMeta("cauchy", "real", ("mu", "gamma")),
    )


class MixtureDistribution(Distribution):
    """Finite mixture of univariate SciPy distributions."""

    meta = DistMeta("mixture", "union", ("weights", "components"))

    def __init__(self, weights: Any, components: list[Distribution]) -> None:
        w = as_vector(weights)
        if len(w) != len(components):
            raise ValueError("weights and components length mismatch")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        self.weights = w / w.sum()
        self.components = list(components)

    def pdf(self, x: Any) -> np.ndarray:
        x_arr = as_array(x)
        dens = np.zeros_like(x_arr, dtype=np.float64)
        for w, comp in zip(self.weights, self.components, strict=True):
            dens = dens + w * comp.pdf(x_arr)
        return dens

    def logpdf(self, x: Any) -> np.ndarray:
        from iqrp.app.math.utils.numerical_stability import logsumexp

        x_arr = as_array(x)
        logs = np.stack(
            [
                np.log(w) + comp.logpdf(x_arr)
                for w, comp in zip(self.weights, self.components, strict=True)
            ],
            axis=0,
        )
        # logsumexp over mixture axis
        if x_arr.ndim == 0:
            return np.asarray(logsumexp(logs), dtype=np.float64)
        return np.asarray(logsumexp(logs, axis=0), dtype=np.float64)

    def cdf(self, x: Any) -> np.ndarray:
        x_arr = as_array(x)
        out = np.zeros_like(x_arr, dtype=np.float64)
        for w, comp in zip(self.weights, self.components, strict=True):
            out = out + w * comp.cdf(x_arr)
        return out

    def rvs(
        self, size: int | tuple[int, ...] = 1, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        flat = int(np.prod(size)) if isinstance(size, tuple) else int(size)
        comps = rng.choice(len(self.weights), size=flat, p=self.weights)
        out = np.empty(flat, dtype=np.float64)
        for i, comp in enumerate(self.components):
            mask = comps == i
            k = int(mask.sum())
            if k:
                out[mask] = comp.rvs(size=k, rng=rng).ravel()[:k]
        return out.reshape(size)


def get_distribution(name: str, **params: Any) -> Distribution:
    """Factory by distribution name."""
    factories: dict[str, Any] = {
        "gaussian": gaussian,
        "normal": gaussian,
        "multivariate_gaussian": multivariate_gaussian,
        "student_t": student_t,
        "bernoulli": bernoulli,
        "binomial": binomial,
        "poisson": poisson,
        "exponential": exponential,
        "gamma": gamma,
        "beta": beta,
        "dirichlet": dirichlet,
        "uniform": uniform,
        "laplace": laplace,
        "lognormal": lognormal,
        "chi_square": chi_square,
        "f": f_distribution,
        "weibull": weibull,
        "cauchy": cauchy,
    }
    if name not in factories:
        from iqrp.app.core.exceptions import ValidationError

        raise ValidationError(f"Unknown distribution '{name}'", code="MATH_DIST_UNKNOWN")
    return factories[name](**params)  # type: ignore[no-any-return]

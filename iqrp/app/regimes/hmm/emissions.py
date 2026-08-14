"""Emission models for Hidden Markov Models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np

from iqrp.app.math.matrices.matrix import normalize_rows
from iqrp.app.math.utils.numerical_stability import logsumexp

CovarianceType = Literal["diag", "full"]


class EmissionModel(ABC):
    """Maps hidden state to observation likelihood."""

    n_states: int
    n_features: int

    @abstractmethod
    def log_prob(self, observations: np.ndarray) -> np.ndarray:
        """Return ``(T, K)`` log emission densities."""

    @abstractmethod
    def sample(
        self,
        states: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw observations for a state sequence."""

    @abstractmethod
    def m_step(
        self,
        observations: np.ndarray,
        responsibilities: np.ndarray,
        *,
        min_covar: float = 1e-6,
    ) -> None:
        """Update emission parameters from posterior responsibilities ``(T, K)``."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize parameters."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> EmissionModel:
        """Deserialize parameters."""


class DiscreteEmissionModel(EmissionModel):
    """Categorical emissions over a finite observation alphabet."""

    def __init__(
        self,
        n_states: int,
        n_symbols: int,
        *,
        probs: np.ndarray | None = None,
        alpha: float = 1.0,
    ) -> None:
        self.n_states = int(n_states)
        self.n_features = 1
        self.n_symbols = int(n_symbols)
        self.alpha = float(alpha)
        if probs is None:
            self.probs = np.full((self.n_states, self.n_symbols), 1.0 / self.n_symbols)
        else:
            self.probs = normalize_rows(np.asarray(probs, dtype=np.float64))

    def log_prob(self, observations: np.ndarray) -> np.ndarray:
        y = np.asarray(observations, dtype=np.int64).reshape(-1)
        out = np.empty((y.size, self.n_states), dtype=np.float64)
        log_p = np.log(np.clip(self.probs, 1e-300, None))
        for t, sym in enumerate(y):
            s = int(np.clip(sym, 0, self.n_symbols - 1))
            out[t] = log_p[:, s]
        return out

    def sample(
        self,
        states: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        s = np.asarray(states, dtype=np.int64).reshape(-1)
        out = np.empty(s.size, dtype=np.int64)
        for t, st in enumerate(s):
            out[t] = int(rng.choice(self.n_symbols, p=self.probs[int(st)]))
        return out.reshape(-1, 1)

    def m_step(
        self,
        observations: np.ndarray,
        responsibilities: np.ndarray,
        *,
        min_covar: float = 1e-6,
    ) -> None:
        del min_covar
        y = np.asarray(observations, dtype=np.int64).reshape(-1)
        gamma = np.asarray(responsibilities, dtype=np.float64)
        counts = np.full((self.n_states, self.n_symbols), self.alpha, dtype=np.float64)
        for t, sym in enumerate(y):
            s = int(np.clip(sym, 0, self.n_symbols - 1))
            counts[:, s] += gamma[t]
        self.probs = normalize_rows(counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "discrete",
            "n_states": self.n_states,
            "n_symbols": self.n_symbols,
            "alpha": self.alpha,
            "probs": self.probs.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscreteEmissionModel:
        return cls(
            int(data["n_states"]),
            int(data["n_symbols"]),
            probs=np.asarray(data["probs"], dtype=np.float64),
            alpha=float(data.get("alpha", 1.0)),
        )


class GaussianEmissionModel(EmissionModel):
    """Univariate or multivariate Gaussian emissions (diag or full covariance)."""

    def __init__(
        self,
        n_states: int,
        n_features: int,
        *,
        means: np.ndarray | None = None,
        covars: np.ndarray | None = None,
        covariance_type: CovarianceType = "diag",
    ) -> None:
        self.n_states = int(n_states)
        self.n_features = int(n_features)
        self.covariance_type: CovarianceType = covariance_type
        if means is None:
            self.means = np.zeros((self.n_states, self.n_features), dtype=np.float64)
        else:
            self.means = np.asarray(means, dtype=np.float64).reshape(self.n_states, self.n_features)
        if covars is None:
            if covariance_type == "diag":
                self.covars = np.ones((self.n_states, self.n_features), dtype=np.float64)
            else:
                self.covars = np.array(
                    [np.eye(self.n_features) for _ in range(self.n_states)], dtype=np.float64
                )
        else:
            self.covars = np.asarray(covars, dtype=np.float64)

    def log_prob(self, observations: np.ndarray) -> np.ndarray:
        y = np.asarray(observations, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        t, _d = y.shape
        out = np.empty((t, self.n_states), dtype=np.float64)
        for k in range(self.n_states):
            out[:, k] = _gaussian_logpdf(y, self.means[k], self.covars[k], self.covariance_type)
        return out

    def sample(
        self,
        states: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        s = np.asarray(states, dtype=np.int64).reshape(-1)
        out = np.empty((s.size, self.n_features), dtype=np.float64)
        for t, st in enumerate(s):
            k = int(st)
            if self.covariance_type == "diag":
                out[t] = rng.normal(self.means[k], np.sqrt(np.clip(self.covars[k], 1e-12, None)))
            else:
                out[t] = rng.multivariate_normal(self.means[k], self.covars[k])
        return out

    def m_step(
        self,
        observations: np.ndarray,
        responsibilities: np.ndarray,
        *,
        min_covar: float = 1e-6,
    ) -> None:
        y = np.asarray(observations, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        gamma = np.asarray(responsibilities, dtype=np.float64)
        nk = np.clip(gamma.sum(axis=0), 1e-12, None)
        self.means = (gamma.T @ y) / nk[:, None]
        if self.covariance_type == "diag":
            cov = np.empty((self.n_states, self.n_features), dtype=np.float64)
            for k in range(self.n_states):
                diff = y - self.means[k]
                cov[k] = np.clip(
                    (gamma[:, k][:, None] * diff**2).sum(axis=0) / nk[k], min_covar, None
                )
            self.covars = cov
        else:
            cov = np.empty((self.n_states, self.n_features, self.n_features), dtype=np.float64)
            for k in range(self.n_states):
                diff = y - self.means[k]
                weighted = diff * np.sqrt(gamma[:, k])[:, None]
                c = (weighted.T @ weighted) / nk[k]
                c = c + min_covar * np.eye(self.n_features)
                cov[k] = 0.5 * (c + c.T)
            self.covars = cov

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "gaussian",
            "n_states": self.n_states,
            "n_features": self.n_features,
            "covariance_type": self.covariance_type,
            "means": self.means.tolist(),
            "covars": self.covars.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GaussianEmissionModel:
        return cls(
            int(data["n_states"]),
            int(data["n_features"]),
            means=np.asarray(data["means"], dtype=np.float64),
            covars=np.asarray(data["covars"], dtype=np.float64),
            covariance_type=data.get("covariance_type", "diag"),
        )


def build_emission(
    kind: str,
    n_states: int,
    n_features: int,
    *,
    covariance_type: CovarianceType = "diag",
    n_symbols: int | None = None,
) -> EmissionModel:
    if kind == "discrete":
        return DiscreteEmissionModel(n_states, n_symbols or max(2, n_features))
    return GaussianEmissionModel(n_states, n_features, covariance_type=covariance_type)


def emission_from_dict(data: dict[str, Any]) -> EmissionModel:
    kind = data.get("type", "gaussian")
    if kind == "discrete":
        return DiscreteEmissionModel.from_dict(data)
    return GaussianEmissionModel.from_dict(data)


def _gaussian_logpdf(
    y: np.ndarray,
    mean: np.ndarray,
    covar: np.ndarray,
    covariance_type: CovarianceType,
) -> np.ndarray:
    diff = y - mean
    d = y.shape[1]
    if covariance_type == "diag":
        var = np.clip(np.asarray(covar, dtype=np.float64).reshape(-1), 1e-12, None)
        out = -0.5 * (d * np.log(2 * np.pi) + np.sum(np.log(var)) + np.sum(diff**2 / var, axis=1))
        return np.asarray(out, dtype=np.float64)
    cov = np.asarray(covar, dtype=np.float64)
    cov = cov + 1e-9 * np.eye(d)
    try:
        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise np.linalg.LinAlgError
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov = cov + 1e-3 * np.eye(d)
        _, logdet = np.linalg.slogdet(cov)
        inv = np.linalg.pinv(cov)
    quad = np.einsum("ti,ij,tj->t", diff, inv, diff)
    return np.asarray(-0.5 * (d * np.log(2 * np.pi) + logdet + quad), dtype=np.float64)


def mixture_log_likelihood(log_emissions: np.ndarray, weights: np.ndarray) -> float:
    """Marginal mixture LL helper (not the HMM sequence LL)."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / max(float(w.sum()), 1e-300)
    log_w = np.log(np.clip(w, 1e-300, None))
    return float(np.sum(logsumexp(log_emissions + log_w[None, :], axis=1)))

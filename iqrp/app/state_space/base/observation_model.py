"""Abstract observation (emission) model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from iqrp.app.math.probability.distributions import gaussian
from iqrp.app.math.utils.numerical_stability import logsumexp, stable_softmax


class ObservationModel(ABC):
    """Maps latent state to observation likelihood / generative draw."""

    @abstractmethod
    def emission_probability(self, observation: Any, state: int) -> float:
        """``p(y | s = state)`` (density or mass)."""

    @abstractmethod
    def sample_observation(
        self,
        state: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw ``y | s = state``."""

    @abstractmethod
    def expected_observation(self, state: int) -> np.ndarray:
        """``E[y | s = state]``."""

    def log_emission(self, observation: Any, state: int) -> float:
        p = self.emission_probability(observation, state)
        return float(np.log(max(p, 1e-300)))

    def emission_matrix(self, observations: np.ndarray) -> np.ndarray:
        """Likelihood matrix ``(T, K)`` for a batch of observations."""
        y = np.asarray(observations, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        t, _ = y.shape
        k = self.n_states
        out = np.empty((t, k), dtype=np.float64)
        for i in range(t):
            for s in range(k):
                out[i, s] = self.emission_probability(y[i], s)
        return out

    def log_emission_matrix(self, observations: np.ndarray) -> np.ndarray:
        e = self.emission_matrix(observations)
        return np.asarray(np.log(np.clip(e, 1e-300, None)), dtype=np.float64)

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Number of latent states ``K``."""


class DiagonalGaussianObservationModel(ObservationModel):
    """Independent Gaussian emissions per latent state (framework utility)."""

    def __init__(self, means: Any, variances: Any) -> None:
        mu = np.asarray(means, dtype=np.float64)
        var = np.asarray(variances, dtype=np.float64)
        if mu.ndim == 1:
            mu = mu.reshape(-1, 1)
        if var.ndim == 1:
            var = var.reshape(-1, 1)
        if mu.shape != var.shape:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(
                "means and variances shape mismatch",
                code="SS_OBS_SHAPE",
            )
        self._means = mu
        self._variances = np.clip(var, 1e-12, None)

    @property
    def n_states(self) -> int:
        return int(self._means.shape[0])

    @property
    def obs_dim(self) -> int:
        return int(self._means.shape[1])

    def emission_probability(self, observation: Any, state: int) -> float:
        y = np.asarray(observation, dtype=np.float64).reshape(-1)
        mu = self._means[int(state)]
        var = self._variances[int(state)]
        # Product of univariate Gaussians via math-engine Distribution
        log_p = 0.0
        for d in range(len(mu)):
            dist = gaussian(float(mu[d]), float(np.sqrt(var[d])))
            x = float(y[d]) if d < len(y) else float(mu[d])
            log_p += float(np.asarray(dist.logpdf(x), dtype=np.float64).reshape(-1)[0])
        return float(np.exp(min(log_p, 700.0)))

    def log_emission(self, observation: Any, state: int) -> float:
        y = np.asarray(observation, dtype=np.float64).reshape(-1)
        mu = self._means[int(state)]
        var = self._variances[int(state)]
        log_p = 0.0
        for d in range(len(mu)):
            dist = gaussian(float(mu[d]), float(np.sqrt(var[d])))
            x = float(y[d]) if d < len(y) else float(mu[d])
            log_p += float(np.asarray(dist.logpdf(x), dtype=np.float64).reshape(-1)[0])
        return float(log_p)

    def log_emission_matrix(self, observations: np.ndarray) -> np.ndarray:
        y = np.asarray(observations, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        t = y.shape[0]
        k = self.n_states
        out = np.zeros((t, k), dtype=np.float64)
        for s in range(k):
            mu = self._means[s]
            var = self._variances[s]
            for d in range(self.obs_dim):
                dist = gaussian(float(mu[d]), float(np.sqrt(var[d])))
                col = y[:, d] if d < y.shape[1] else np.full(t, mu[d])
                out[:, s] += np.asarray(dist.logpdf(col), dtype=np.float64)
        return out

    def sample_observation(
        self,
        state: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        rng = rng or np.random.default_rng()
        mu = self._means[int(state)]
        std = np.sqrt(self._variances[int(state)])
        return np.asarray(rng.normal(mu, std), dtype=np.float64)

    def expected_observation(self, state: int) -> np.ndarray:
        return np.asarray(self._means[int(state)].copy(), dtype=np.float64)

    def predictive_density(self, observation: Any, state_probs: Any) -> float:
        """Mixture predictive ``sum_k π_k p(y|k)`` in log-space."""
        pi = np.asarray(state_probs, dtype=np.float64).reshape(-1)
        pi = pi / max(float(pi.sum()), 1e-300)
        logs = np.array(
            [
                self.log_emission(observation, s) + np.log(max(pi[s], 1e-300))
                for s in range(len(pi))
            ],
            dtype=np.float64,
        )
        return float(np.exp(min(float(logsumexp(logs)), 700.0)))

    def soft_responsibilities(
        self, observations: np.ndarray, prior: Any | None = None
    ) -> np.ndarray:
        log_e = self.log_emission_matrix(observations)
        if prior is None:
            return stable_softmax(log_e, axis=1)
        log_pi = np.log(np.clip(np.asarray(prior, dtype=np.float64).reshape(-1), 1e-300, None))
        return stable_softmax(log_e + log_pi[None, :], axis=1)

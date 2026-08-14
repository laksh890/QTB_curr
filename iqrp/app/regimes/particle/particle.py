"""Particle representation and cloud utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.math.utils.numerical_stability import logsumexp, stable_softmax


@dataclass
class Particle:
    """Single particle with state and importance weight."""

    state: np.ndarray
    weight: float = 1.0
    log_weight: float = 0.0
    likelihood: float = 1.0
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.state = np.asarray(self.state, dtype=np.float64).reshape(-1)

    def copy(self) -> Particle:
        return Particle(
            state=self.state.copy(),
            weight=float(self.weight),
            log_weight=float(self.log_weight),
            likelihood=float(self.likelihood),
            timestamp=self.timestamp,
            metadata=dict(self.metadata),
        )


@dataclass
class ParticleCloud:
    """Collection of weighted particles."""

    states: np.ndarray  # (N, d)
    log_weights: np.ndarray  # (N,)
    likelihoods: np.ndarray  # (N,)
    timestamps: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states, dtype=np.float64)
        if self.states.ndim == 1:
            self.states = self.states.reshape(-1, 1)
        n = self.states.shape[0]
        self.log_weights = np.asarray(self.log_weights, dtype=np.float64).reshape(n)
        self.likelihoods = np.asarray(self.likelihoods, dtype=np.float64).reshape(n)
        if self.timestamps is not None:
            self.timestamps = np.asarray(self.timestamps, dtype=np.float64).reshape(n)

    @property
    def n_particles(self) -> int:
        return int(self.states.shape[0])

    @property
    def dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def weights(self) -> np.ndarray:
        return stable_softmax(self.log_weights)

    def normalize(self) -> ParticleCloud:
        w = self.weights
        log_w = np.log(np.clip(w, 1e-300, None))
        return ParticleCloud(
            states=self.states.copy(),
            log_weights=log_w,
            likelihoods=self.likelihoods.copy(),
            timestamps=None if self.timestamps is None else self.timestamps.copy(),
            metadata=dict(self.metadata),
        )

    def mean(self) -> np.ndarray:
        w = self.weights
        return np.sum(w[:, None] * self.states, axis=0)

    def covariance(self) -> np.ndarray:
        w = self.weights
        mu = self.mean()
        diff = self.states - mu
        return (w[:, None, None] * (diff[:, :, None] @ diff[:, None, :])).sum(axis=0)

    def ess(self) -> float:
        w = self.weights
        return float(1.0 / np.sum(w**2))

    def log_likelihood_increment(self) -> float:
        return float(logsumexp(self.log_weights) - np.log(self.n_particles))

    def to_particles(self) -> list[Particle]:
        w = self.weights
        out: list[Particle] = []
        for i in range(self.n_particles):
            out.append(
                Particle(
                    state=self.states[i],
                    weight=float(w[i]),
                    log_weight=float(self.log_weights[i]),
                    likelihood=float(self.likelihoods[i]),
                    timestamp=(None if self.timestamps is None else float(self.timestamps[i])),
                )
            )
        return out

    @classmethod
    def from_particles(cls, particles: list[Particle]) -> ParticleCloud:
        states = np.vstack([p.state for p in particles])
        log_w = np.array([p.log_weight for p in particles], dtype=np.float64)
        likes = np.array([p.likelihood for p in particles], dtype=np.float64)
        ts = np.array(
            [np.nan if p.timestamp is None else p.timestamp for p in particles],
            dtype=np.float64,
        )
        return cls(states=states, log_weights=log_w, likelihoods=likes, timestamps=ts)

    @classmethod
    def equal_weight(
        cls,
        states: np.ndarray,
        *,
        timestamp: float | None = None,
    ) -> ParticleCloud:
        x = np.asarray(states, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        n = x.shape[0]
        log_w = np.full(n, -np.log(n), dtype=np.float64)
        ts = None if timestamp is None else np.full(n, timestamp, dtype=np.float64)
        return cls(states=x, log_weights=log_w, likelihoods=np.ones(n), timestamps=ts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "states": self.states.tolist(),
            "log_weights": self.log_weights.tolist(),
            "likelihoods": self.likelihoods.tolist(),
            "timestamps": None if self.timestamps is None else self.timestamps.tolist(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParticleCloud:
        return cls(
            states=np.asarray(data["states"], dtype=np.float64),
            log_weights=np.asarray(data["log_weights"], dtype=np.float64),
            likelihoods=np.asarray(data["likelihoods"], dtype=np.float64),
            timestamps=(
                None
                if data.get("timestamps") is None
                else np.asarray(data["timestamps"], dtype=np.float64)
            ),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class FilterTrace:
    """Batch particle filter output."""

    means: np.ndarray
    covs: np.ndarray
    clouds: list[ParticleCloud]
    ess: np.ndarray
    resampled: np.ndarray
    log_likelihood: float
    metadata: dict[str, Any] = field(default_factory=dict)

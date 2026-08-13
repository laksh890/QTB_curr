"""State transition / particle propagation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

from iqrp.app.regimes.particle.config import ParticleSettings
from iqrp.app.regimes.particle.particle import ParticleCloud

Application = Literal[
    "custom",
    "nonlinear_trend",
    "volatility",
    "liquidity",
    "dynamic_corr",
    "market_stress",
    "risk_factors",
]


@dataclass
class TransitionModel:
    """Latent-state transition model."""

    f: np.ndarray | None = None
    q_scale: float = 0.01
    dt: float = 1.0
    application: str = "custom"
    transition_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None
    observe_fn: Callable[[np.ndarray], np.ndarray] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        if self.f is not None:
            return int(np.asarray(self.f).shape[0])
        return int(self.metadata.get("n_states", 1))

    def propagate(
        self,
        states: np.ndarray,
        *,
        rng: np.random.Generator,
        t: int = 0,
    ) -> np.ndarray:
        x = np.asarray(states, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if self.transition_fn is not None:
            return np.asarray(self.transition_fn(x, rng), dtype=np.float64)
        if self.f is not None:
            f = np.asarray(self.f, dtype=np.float64)
            mean = x @ f.T
            noise = rng.normal(0.0, self.q_scale, size=mean.shape)
            return mean + noise
        # random walk default
        return x + rng.normal(0.0, self.q_scale, size=x.shape)

    def observe(self, states: np.ndarray) -> np.ndarray:
        x = np.asarray(states, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if self.observe_fn is not None:
            return np.asarray(self.observe_fn(x), dtype=np.float64)
        return x[:, :1].copy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "f": None if self.f is None else np.asarray(self.f).tolist(),
            "q_scale": float(self.q_scale),
            "dt": float(self.dt),
            "application": self.application,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransitionModel:
        return cls(
            f=None if data.get("f") is None else np.asarray(data["f"], dtype=np.float64),
            q_scale=float(data.get("q_scale", 0.01)),
            dt=float(data.get("dt", 1.0)),
            application=str(data.get("application", "custom")),
            metadata=dict(data.get("metadata") or {}),
        )


def build_transition(
    settings: ParticleSettings,
    *,
    application: Application | None = None,
    n_states: int | None = None,
) -> TransitionModel:
    app = application or settings.application
    n = int(n_states if n_states is not None else settings.n_states)
    dt = float(settings.system.dt)
    q = float(settings.system.process_noise_scale)

    if app == "nonlinear_trend":
        # local level with nonlinear drift: x += tanh(x)*dt + noise
        def trans(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            drift = np.tanh(x) * dt
            return x + drift + rng.normal(0.0, q, size=x.shape)

        def obs(x: np.ndarray) -> np.ndarray:
            return x[:, :1]

        return TransitionModel(
            f=np.eye(max(n, 1)),
            q_scale=q,
            dt=dt,
            application=app,
            transition_fn=trans,
            observe_fn=obs,
            metadata={"n_states": max(n, 1)},
        )

    if app == "volatility":
        # AR(1) log-vol; observation = exp(x/2) scale proxy via exp(x)
        phi = 0.95

        def trans(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            return phi * x + rng.normal(0.0, q, size=x.shape)

        def obs(x: np.ndarray) -> np.ndarray:
            return np.exp(x[:, :1])

        return TransitionModel(
            f=np.array([[phi]]),
            q_scale=q,
            dt=dt,
            application=app,
            transition_fn=trans,
            observe_fn=obs,
            metadata={"n_states": 1, "phi": phi},
        )

    if app == "liquidity":
        # bounded liquidity intensity via logistic of OU-like state
        def trans(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            return 0.9 * x + rng.normal(0.0, q, size=x.shape)

        def obs(x: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-x[:, :1]))

        return TransitionModel(
            f=np.array([[0.9]]),
            q_scale=q,
            dt=dt,
            application=app,
            transition_fn=trans,
            observe_fn=obs,
            metadata={"n_states": 1},
        )

    if app == "dynamic_corr":
        # Fisher-z correlation factor
        def trans(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            return 0.98 * x + rng.normal(0.0, q, size=x.shape)

        def obs(x: np.ndarray) -> np.ndarray:
            return np.tanh(x[:, :1])

        return TransitionModel(
            f=np.array([[0.98]]),
            q_scale=q,
            dt=dt,
            application=app,
            transition_fn=trans,
            observe_fn=obs,
            metadata={"n_states": 1},
        )

    if app == "market_stress":
        # sigmoid stress intensity
        def trans(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            return 0.85 * x + rng.normal(0.0, q * 1.5, size=x.shape)

        def obs(x: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-x[:, :1]))

        return TransitionModel(
            f=np.array([[0.85]]),
            q_scale=q * 1.5,
            dt=dt,
            application=app,
            transition_fn=trans,
            observe_fn=obs,
            metadata={"n_states": 1},
        )

    if app == "risk_factors":
        # multifactor random walk
        d = max(n, 2)
        f = np.eye(d)

        def trans(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            xx = x if x.shape[1] == d else np.pad(x, ((0, 0), (0, max(0, d - x.shape[1]))))[:, :d]
            return xx @ f.T + rng.normal(0.0, q, size=xx.shape)

        def obs(x: np.ndarray) -> np.ndarray:
            return x[:, :1]

        return TransitionModel(
            f=f,
            q_scale=q,
            dt=dt,
            application=app,
            transition_fn=trans,
            observe_fn=obs,
            metadata={"n_states": d},
        )

    # custom linear random walk
    d = max(n, 1)
    f = np.eye(d)
    return TransitionModel(
        f=f,
        q_scale=q,
        dt=dt,
        application="custom",
        metadata={"n_states": d},
    )


def propagate_cloud(
    cloud: ParticleCloud,
    model: TransitionModel,
    *,
    rng: np.random.Generator,
    t: int = 0,
) -> ParticleCloud:
    new_states = model.propagate(cloud.states, rng=rng, t=t)
    return ParticleCloud(
        states=new_states,
        log_weights=cloud.log_weights.copy(),
        likelihoods=cloud.likelihoods.copy(),
        timestamps=None if cloud.timestamps is None else cloud.timestamps.copy(),
        metadata=dict(cloud.metadata),
    )

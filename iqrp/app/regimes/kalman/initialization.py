"""System matrix builders and financial application templates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

from iqrp.app.regimes.kalman.config import KalmanSettings
from iqrp.app.regimes.kalman.covariance import ensure_spd

Application = Literal[
    "custom", "trend", "denoise", "dynamic_beta", "volatility", "spread", "pairs"
]


@dataclass
class LinearGaussianSSM:
    """Linear-Gaussian state-space system matrices."""

    f: np.ndarray
    h: np.ndarray
    q: np.ndarray
    r: np.ndarray
    x0: np.ndarray
    p0: np.ndarray
    b: np.ndarray | None = None
    application: str = "custom"
    # optional nonlinear hooks for EKF/UKF
    f_fn: Callable[[np.ndarray], np.ndarray] | None = None
    h_fn: Callable[[np.ndarray], np.ndarray] | None = None
    f_jac: Callable[[np.ndarray], np.ndarray] | None = None
    h_jac: Callable[[np.ndarray], np.ndarray] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_states(self) -> int:
        return int(self.f.shape[0])

    @property
    def n_obs(self) -> int:
        return int(self.h.shape[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "f": self.f.tolist(),
            "h": self.h.tolist(),
            "q": self.q.tolist(),
            "r": self.r.tolist(),
            "x0": self.x0.tolist(),
            "p0": self.p0.tolist(),
            "b": None if self.b is None else self.b.tolist(),
            "application": self.application,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LinearGaussianSSM:
        return cls(
            f=np.asarray(data["f"], dtype=np.float64),
            h=np.asarray(data["h"], dtype=np.float64),
            q=ensure_spd(data["q"]),
            r=ensure_spd(data["r"]),
            x0=np.asarray(data["x0"], dtype=np.float64).reshape(-1),
            p0=ensure_spd(data["p0"]),
            b=None if data.get("b") is None else np.asarray(data["b"], dtype=np.float64),
            application=str(data.get("application", "custom")),
            metadata=dict(data.get("metadata") or {}),
        )


def build_system(
    settings: KalmanSettings,
    *,
    n_states: int | None = None,
    n_obs: int | None = None,
    application: Application | None = None,
) -> LinearGaussianSSM:
    app = application or settings.application
    n = int(n_states if n_states is not None else settings.n_states)
    m = int(n_obs if n_obs is not None else settings.n_obs)
    dt = float(settings.system.dt)
    q_scale = float(settings.system.process_noise_scale)
    r_scale = float(settings.system.observation_noise_scale)
    x_scale = float(settings.system.initial_state_scale)
    p_scale = float(settings.system.initial_covariance_scale)

    if app == "trend":
        # local linear trend: state = [level, slope]
        f = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        h = np.array([[1.0, 0.0]], dtype=np.float64)
        q = q_scale * np.array(
            [[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]], dtype=np.float64
        )
        r = np.array([[r_scale]], dtype=np.float64)
        x0 = np.zeros(2)
        p0 = p_scale * np.eye(2)
        return LinearGaussianSSM(f, h, q, r, x0, p0, application="trend")

    if app == "denoise":
        f = np.array([[1.0]], dtype=np.float64)
        h = np.array([[1.0]], dtype=np.float64)
        q = np.array([[q_scale]], dtype=np.float64)
        r = np.array([[r_scale]], dtype=np.float64)
        return LinearGaussianSSM(f, h, q, r, np.zeros(1), p_scale * np.eye(1), application="denoise")

    if app == "dynamic_beta":
        # state = [alpha, beta]; observation = alpha + beta * market (+ noise)
        # market enters via time-varying H — stored as identity placeholder; caller sets H_t
        f = np.eye(2)
        h = np.array([[1.0, 0.0]], dtype=np.float64)  # overwritten per-step with [1, mkt]
        q = q_scale * np.eye(2)
        r = np.array([[r_scale]], dtype=np.float64)
        return LinearGaussianSSM(
            f, h, q, r, np.zeros(2), p_scale * np.eye(2), application="dynamic_beta"
        )

    if app == "volatility":
        # AR(1) on log-variance; observation is squared return proxy
        phi = 0.95
        f = np.array([[phi]], dtype=np.float64)
        h = np.array([[1.0]], dtype=np.float64)
        q = np.array([[q_scale]], dtype=np.float64)
        r = np.array([[r_scale]], dtype=np.float64)

        def f_fn(x: np.ndarray) -> np.ndarray:
            return np.array([phi * float(x[0])])

        def h_fn(x: np.ndarray) -> np.ndarray:
            return np.array([float(np.exp(x[0]))])

        def f_jac(x: np.ndarray) -> np.ndarray:
            return np.array([[phi]])

        def h_jac(x: np.ndarray) -> np.ndarray:
            return np.array([[float(np.exp(x[0]))]])

        return LinearGaussianSSM(
            f,
            h,
            q,
            r,
            np.array([x_scale]),
            p_scale * np.eye(1),
            application="volatility",
            f_fn=f_fn,
            h_fn=h_fn,
            f_jac=f_jac,
            h_jac=h_jac,
        )

    if app in ("spread", "pairs"):
        # OU-like spread: x_{t+1} = (1-kappa*dt) x + noise; observe spread
        kappa = 0.1
        f = np.array([[max(1.0 - kappa * dt, 0.0)]], dtype=np.float64)
        h = np.array([[1.0]], dtype=np.float64)
        q = np.array([[q_scale]], dtype=np.float64)
        r = np.array([[r_scale]], dtype=np.float64)
        return LinearGaussianSSM(
            f, h, q, r, np.zeros(1), p_scale * np.eye(1), application=app
        )

    # custom random-walk / identity defaults
    n = max(n, 1)
    m = max(m, 1)
    f = np.eye(n)
    h = np.zeros((m, n))
    h[:, : min(m, n)] = np.eye(min(m, n))
    q = q_scale * np.eye(n)
    r = r_scale * np.eye(m)
    return LinearGaussianSSM(
        f, h, q, r, x_scale * np.zeros(n), p_scale * np.eye(n), application="custom"
    )


def numerical_jacobian(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    x0 = np.asarray(x, dtype=np.float64).reshape(-1)
    y0 = np.asarray(fn(x0), dtype=np.float64).reshape(-1)
    jac = np.empty((y0.size, x0.size), dtype=np.float64)
    for i in range(x0.size):
        xp = x0.copy()
        xp[i] += eps
        yp = np.asarray(fn(xp), dtype=np.float64).reshape(-1)
        jac[:, i] = (yp - y0) / eps
    return jac

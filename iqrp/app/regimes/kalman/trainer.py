"""Training / noise estimation for Kalman filters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.regimes.kalman.adaptive import adapt_noise_from_trace
from iqrp.app.regimes.kalman.config import KalmanSettings
from iqrp.app.regimes.kalman.covariance import ensure_spd
from iqrp.app.regimes.kalman.ekf import filter_ekf
from iqrp.app.regimes.kalman.initialization import LinearGaussianSSM, build_system
from iqrp.app.regimes.kalman.linear import FilterTrace, filter_linear
from iqrp.app.regimes.kalman.ukf import filter_ukf
from iqrp.app.regimes.kalman.adaptive import filter_adaptive


@dataclass
class TrainResult:
    system: LinearGaussianSSM
    trace: FilterTrace
    history: list[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def simulate_lds(
    system: LinearGaussianSSM,
    n_steps: int,
    *,
    rng: np.random.Generator | None = None,
    controls: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate latent states and observations from a linear-Gaussian SSM."""
    gen = rng or np.random.default_rng()
    n, m = system.n_states, system.n_obs
    x = system.x0.copy()
    if np.allclose(system.p0, 0):
        pass
    else:
        x = gen.multivariate_normal(system.x0, ensure_spd(system.p0))
    states = np.empty((n_steps, n), dtype=np.float64)
    obs = np.empty((n_steps, m), dtype=np.float64)
    for t in range(n_steps):
        u = None if controls is None else np.asarray(controls[t], dtype=np.float64)
        mean = system.f @ x
        if system.b is not None and u is not None:
            mean = mean + system.b @ u.reshape(-1)
        x = gen.multivariate_normal(mean, ensure_spd(system.q))
        if system.h_fn is not None:
            y_mean = np.asarray(system.h_fn(x), dtype=np.float64).reshape(-1)
        else:
            y_mean = system.h @ x
        y = gen.multivariate_normal(y_mean, ensure_spd(system.r))
        states[t] = x
        obs[t] = y
    return states, obs


def run_filter(
    observations: np.ndarray,
    system: LinearGaussianSSM,
    settings: KalmanSettings,
    *,
    controls: np.ndarray | None = None,
    h_seq: np.ndarray | None = None,
) -> FilterTrace:
    ft = settings.filter_type
    if ft == "ekf":
        return filter_ekf(observations, system)
    if ft == "ukf":
        return filter_ukf(
            observations,
            system,
            alpha=settings.ukf.alpha,
            beta=settings.ukf.beta,
            kappa=settings.ukf.kappa,
        )
    if ft == "adaptive":
        return filter_adaptive(
            observations,
            system,
            window=settings.adaptive.window,
            process_adapt_rate=settings.adaptive.process_adapt_rate,
            observation_adapt_rate=settings.adaptive.observation_adapt_rate,
            innovation_threshold=settings.adaptive.innovation_threshold,
            controls=controls,
            h_seq=h_seq,
        )
    return filter_linear(observations, system, controls=controls, h_seq=h_seq)


class KalmanTrainer:
    def __init__(self, settings: KalmanSettings | None = None) -> None:
        self.settings = settings or KalmanSettings.default()

    def build_system(
        self,
        *,
        n_states: int | None = None,
        n_obs: int | None = None,
        application: str | None = None,
    ) -> LinearGaussianSSM:
        from typing import cast

        from iqrp.app.regimes.kalman.initialization import Application

        app = cast(Application | None, application)
        return build_system(self.settings, n_states=n_states, n_obs=n_obs, application=app)

    def fit(
        self,
        observations: np.ndarray,
        *,
        system: LinearGaussianSSM | None = None,
        controls: np.ndarray | None = None,
        h_seq: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> TrainResult:
        y = np.asarray(observations, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        sys = system or self.build_system(n_obs=y.shape[1])
        # seed x0 from first observation if still zero
        if np.allclose(sys.x0, 0.0) and y.shape[0] > 0:
            x0 = np.zeros(sys.n_states, dtype=np.float64)
            x0[0] = float(y[0, 0])
            sys = LinearGaussianSSM(
                f=sys.f,
                h=sys.h,
                q=sys.q,
                r=sys.r,
                x0=x0,
                p0=sys.p0,
                b=sys.b,
                application=sys.application,
                f_fn=sys.f_fn,
                h_fn=sys.h_fn,
                f_jac=sys.f_jac,
                h_jac=sys.h_jac,
                metadata=dict(sys.metadata),
            )

        history: list[float] = []
        max_iter = max(1, int(self.settings.training.em_iterations))
        tol = float(self.settings.training.tol)
        estimate_noise = bool(self.settings.training.estimate_noise)
        prev_ll = -np.inf
        converged = False
        trace = run_filter(y, sys, self.settings, controls=controls, h_seq=h_seq)
        history.append(trace.log_likelihood)

        for it in range(1, max_iter):
            if estimate_noise and self.settings.filter_type in ("linear", "adaptive"):
                q_hat, r_hat = adapt_noise_from_trace(trace, sys)
                # blend toward empirical noise
                alpha = 0.3
                sys = LinearGaussianSSM(
                    f=sys.f,
                    h=sys.h,
                    q=ensure_spd((1 - alpha) * sys.q + alpha * q_hat),
                    r=ensure_spd((1 - alpha) * sys.r + alpha * r_hat),
                    x0=sys.x0,
                    p0=sys.p0,
                    b=sys.b,
                    application=sys.application,
                    f_fn=sys.f_fn,
                    h_fn=sys.h_fn,
                    f_jac=sys.f_jac,
                    h_jac=sys.h_jac,
                    metadata=dict(sys.metadata),
                )
            trace = run_filter(y, sys, self.settings, controls=controls, h_seq=h_seq)
            history.append(trace.log_likelihood)
            if abs(trace.log_likelihood - prev_ll) < tol:
                converged = True
                break
            prev_ll = trace.log_likelihood

        # pull adaptive Q/R if present
        if "q_final" in trace.metadata:
            sys = LinearGaussianSSM(
                f=sys.f,
                h=sys.h,
                q=ensure_spd(trace.metadata["q_final"]),
                r=ensure_spd(trace.metadata.get("r_final", sys.r)),
                x0=sys.x0,
                p0=sys.p0,
                b=sys.b,
                application=sys.application,
                f_fn=sys.f_fn,
                h_fn=sys.h_fn,
                f_jac=sys.f_jac,
                h_jac=sys.h_jac,
                metadata=dict(sys.metadata),
            )

        return TrainResult(
            system=sys,
            trace=trace,
            history=history,
            n_iter=len(history),
            converged=converged or len(history) >= max_iter,
            metadata={"filter_type": self.settings.filter_type, "rng": rng},
        )

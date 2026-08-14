"""Linear Kalman filter batch / online runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.regimes.kalman.covariance import ensure_spd
from iqrp.app.regimes.kalman.initialization import LinearGaussianSSM
from iqrp.app.regimes.kalman.prediction import predict_state
from iqrp.app.regimes.kalman.update import update_state


@dataclass
class FilterTrace:
    means: np.ndarray
    covs: np.ndarray
    pred_means: np.ndarray
    pred_covs: np.ndarray
    innovations: np.ndarray
    innovation_covs: np.ndarray
    gains: np.ndarray
    log_likelihood: float
    metadata: dict[str, Any] = field(default_factory=dict)


def filter_linear(
    observations: np.ndarray,
    system: LinearGaussianSSM,
    *,
    controls: np.ndarray | None = None,
    h_seq: np.ndarray | None = None,
    f_seq: np.ndarray | None = None,
    q_seq: np.ndarray | None = None,
    r_seq: np.ndarray | None = None,
) -> FilterTrace:
    """Run linear KF over an observation sequence ``(T, m)``."""
    z = np.asarray(observations, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    t_steps, m = z.shape
    n = system.n_states
    means = np.empty((t_steps, n), dtype=np.float64)
    covs = np.empty((t_steps, n, n), dtype=np.float64)
    pred_means = np.empty((t_steps, n), dtype=np.float64)
    pred_covs = np.empty((t_steps, n, n), dtype=np.float64)
    innovs = np.empty((t_steps, m), dtype=np.float64)
    s_seq = np.empty((t_steps, m, m), dtype=np.float64)
    gains = np.empty((t_steps, n, m), dtype=np.float64)

    x = system.x0.copy()
    p = ensure_spd(system.p0)
    ll = 0.0
    for t in range(t_steps):
        f = system.f if f_seq is None else np.asarray(f_seq[t], dtype=np.float64)
        q = system.q if q_seq is None else ensure_spd(q_seq[t])
        h = system.h if h_seq is None else np.asarray(h_seq[t], dtype=np.float64)
        r = system.r if r_seq is None else ensure_spd(r_seq[t])
        u = None if controls is None else np.asarray(controls[t], dtype=np.float64)
        x_pred, p_pred = predict_state(x, p, f, q, b=system.b, u=u)
        x, p, innov, s, k = update_state(x_pred, p_pred, z[t], h, r)
        means[t], covs[t] = x, p
        pred_means[t], pred_covs[t] = x_pred, p_pred
        innovs[t], s_seq[t], gains[t] = innov, s, k
        ll += _gaussian_ll(innov, s)
    return FilterTrace(
        means=means,
        covs=covs,
        pred_means=pred_means,
        pred_covs=pred_covs,
        innovations=innovs,
        innovation_covs=s_seq,
        gains=gains,
        log_likelihood=ll,
    )


def _gaussian_ll(innov: np.ndarray, s: np.ndarray) -> float:
    v = np.asarray(innov, dtype=np.float64).reshape(-1)
    s_arr = ensure_spd(s)
    m = v.size
    try:
        sign, logdet = np.linalg.slogdet(s_arr)
        if sign <= 0:
            raise np.linalg.LinAlgError
        quad = float(v @ np.linalg.solve(s_arr, v))
    except np.linalg.LinAlgError:
        logdet = float(np.sum(np.log(np.clip(np.diag(s_arr), 1e-12, None))))
        quad = float(v @ (np.linalg.pinv(s_arr) @ v))
    return -0.5 * (m * np.log(2 * np.pi) + logdet + quad)

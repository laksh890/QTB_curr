"""Adaptive Kalman filter with covariance matching."""

from __future__ import annotations

from collections import deque

import numpy as np

from iqrp.app.regimes.kalman.covariance import ensure_spd, mahalanobis
from iqrp.app.regimes.kalman.initialization import LinearGaussianSSM
from iqrp.app.regimes.kalman.linear import FilterTrace
from iqrp.app.regimes.kalman.prediction import predict_state
from iqrp.app.regimes.kalman.update import update_state


def filter_adaptive(
    observations: np.ndarray,
    system: LinearGaussianSSM,
    *,
    window: int = 20,
    process_adapt_rate: float = 0.05,
    observation_adapt_rate: float = 0.05,
    innovation_threshold: float = 3.0,
    controls: np.ndarray | None = None,
    h_seq: np.ndarray | None = None,
) -> FilterTrace:
    """Linear KF with online Q/R adaptation via innovation covariance matching."""
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
    q = ensure_spd(system.q)
    r = ensure_spd(system.r)
    innov_hist: deque[np.ndarray] = deque(maxlen=max(2, int(window)))
    ll = 0.0
    from iqrp.app.regimes.kalman.linear import _gaussian_ll

    for t in range(t_steps):
        h = system.h if h_seq is None else np.asarray(h_seq[t], dtype=np.float64)
        u = None if controls is None else np.asarray(controls[t], dtype=np.float64)
        x_pred, p_pred = predict_state(x, p, system.f, q, b=system.b, u=u)
        x, p, innov, s, k = update_state(x_pred, p_pred, z[t], h, r)
        innov_hist.append(innov.copy())

        # covariance matching
        if len(innov_hist) >= max(3, window // 2):
            emp = np.cov(np.asarray(innov_hist).T) if m > 1 else np.array([[float(np.var(innov_hist))]])
            emp = ensure_spd(emp)
            # R ← (1-α) R + α emp
            r = ensure_spd((1.0 - observation_adapt_rate) * r + observation_adapt_rate * emp)
            # inflate Q when innovations are large
            d = mahalanobis(innov, s)
            if d > innovation_threshold:
                q = ensure_spd(q * (1.0 + process_adapt_rate * (d / innovation_threshold)))
            else:
                q = ensure_spd((1.0 - process_adapt_rate * 0.1) * q + process_adapt_rate * 0.1 * system.q)

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
        metadata={
            "filter": "adaptive",
            "q_final": q,
            "r_final": r,
        },
    )


def adapt_noise_from_trace(trace: FilterTrace, system: LinearGaussianSSM) -> tuple[np.ndarray, np.ndarray]:
    """Batch covariance-matching estimate of Q, R from a filter trace."""
    innov = trace.innovations
    if innov.ndim == 1:
        innov = innov.reshape(-1, 1)
    r_hat = ensure_spd(np.cov(innov.T) if innov.shape[1] > 1 else np.array([[float(np.var(innov))]]))
    # rough Q from state increments
    dx = np.diff(trace.means, axis=0)
    if dx.size == 0:
        q_hat = system.q.copy()
    else:
        q_hat = ensure_spd(np.cov(dx.T) if dx.shape[1] > 1 else np.array([[float(np.var(dx))]]))
    return q_hat, r_hat

"""Extended Kalman Filter."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from iqrp.app.regimes.kalman.covariance import ensure_spd
from iqrp.app.regimes.kalman.initialization import LinearGaussianSSM, numerical_jacobian
from iqrp.app.regimes.kalman.linear import FilterTrace, _gaussian_ll
from iqrp.app.regimes.kalman.prediction import predict_nonlinear
from iqrp.app.regimes.kalman.update import update_nonlinear


def filter_ekf(
    observations: np.ndarray,
    system: LinearGaussianSSM,
    *,
    f_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    h_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    f_jac: Callable[[np.ndarray], np.ndarray] | None = None,
    h_jac: Callable[[np.ndarray], np.ndarray] | None = None,
) -> FilterTrace:
    """EKF with analytic or numerical Jacobians."""
    z = np.asarray(observations, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    f_fn = f_fn or system.f_fn or (lambda x: system.f @ x)
    h_fn = h_fn or system.h_fn or (lambda x: system.h @ x)
    f_jac = f_jac or system.f_jac or (lambda x: numerical_jacobian(f_fn, x))
    h_jac = h_jac or system.h_jac or (lambda x: numerical_jacobian(h_fn, x))

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
        x_pred, p_pred = predict_nonlinear(x, p, f_fn, f_jac, system.q)
        x, p, innov, s, k = update_nonlinear(x_pred, p_pred, z[t], h_fn, h_jac, system.r)
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
        metadata={"filter": "ekf"},
    )

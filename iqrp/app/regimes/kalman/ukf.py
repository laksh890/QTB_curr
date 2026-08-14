"""Unscented Kalman Filter."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from iqrp.app.regimes.kalman.covariance import ensure_spd
from iqrp.app.regimes.kalman.initialization import LinearGaussianSSM
from iqrp.app.regimes.kalman.linear import FilterTrace, _gaussian_ll


def sigma_points(
    x: np.ndarray,
    p: np.ndarray,
    *,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(sigma_points (2n+1, n), wm, wc)``."""
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x_arr.size
    lam = alpha**2 * (n + kappa) - n
    c = n + lam
    p_spd = ensure_spd(p)
    try:
        chol = np.linalg.cholesky(c * p_spd)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(ensure_spd(p_spd, jitter=1e-6) * c)
    pts = np.empty((2 * n + 1, n), dtype=np.float64)
    pts[0] = x_arr
    for i in range(n):
        pts[i + 1] = x_arr + chol[:, i]
        pts[n + i + 1] = x_arr - chol[:, i]
    wm = np.full(2 * n + 1, 0.5 / c)
    wc = wm.copy()
    wm[0] = lam / c
    wc[0] = lam / c + (1.0 - alpha**2 + beta)
    return pts, wm, wc


def unscented_transform(
    pts: np.ndarray,
    wm: np.ndarray,
    wc: np.ndarray,
    noise: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.sum(wm[:, None] * pts, axis=0)
    diff = pts - mean
    cov = ensure_spd(
        (wc[:, None, None] * (diff[:, :, None] @ diff[:, None, :])).sum(axis=0) + noise
    )
    return mean, cov


def filter_ukf(
    observations: np.ndarray,
    system: LinearGaussianSSM,
    *,
    alpha: float = 1e-3,
    beta: float = 2.0,
    kappa: float = 0.0,
    f_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    h_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> FilterTrace:
    z = np.asarray(observations, dtype=np.float64)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    f_fn = f_fn or system.f_fn or (lambda x: system.f @ x)
    h_fn = h_fn or system.h_fn or (lambda x: system.h @ x)

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
        # predict
        pts, wm, wc = sigma_points(x, p, alpha=alpha, beta=beta, kappa=kappa)
        pts_f = np.vstack([np.asarray(f_fn(pt), dtype=np.float64).reshape(-1) for pt in pts])
        x_pred, p_pred = unscented_transform(pts_f, wm, wc, system.q)
        # update
        pts2, wm2, wc2 = sigma_points(x_pred, p_pred, alpha=alpha, beta=beta, kappa=kappa)
        pts_h = np.vstack([np.asarray(h_fn(pt), dtype=np.float64).reshape(-1) for pt in pts2])
        y_pred, s = unscented_transform(pts_h, wm2, wc2, system.r)
        # cross-covariance
        dx = pts2 - x_pred
        dy = pts_h - y_pred
        p_xy = np.zeros((n, m), dtype=np.float64)
        for i in range(pts2.shape[0]):
            p_xy += wc2[i] * np.outer(dx[i], dy[i])
        try:
            k = p_xy @ np.linalg.inv(s)
        except np.linalg.LinAlgError:
            k = p_xy @ np.linalg.pinv(s)
        innov = z[t] - y_pred
        x = x_pred + k @ innov
        p = ensure_spd(p_pred - k @ s @ k.T)
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
        metadata={"filter": "ukf"},
    )

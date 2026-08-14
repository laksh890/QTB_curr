"""Kalman update (correction) step."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from iqrp.app.regimes.kalman.covariance import ensure_spd, joseph_update, mahalanobis


def update_state(
    x_pred: np.ndarray,
    p_pred: np.ndarray,
    z: np.ndarray,
    h: np.ndarray,
    r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear update.

    Returns ``(x, P, innovation, S, K)``.
    """
    x = np.asarray(x_pred, dtype=np.float64).reshape(-1)
    p = ensure_spd(p_pred)
    h_arr = np.asarray(h, dtype=np.float64)
    r_arr = ensure_spd(r)
    z_arr = np.asarray(z, dtype=np.float64).reshape(-1)
    y_hat = h_arr @ x
    innov = z_arr - y_hat
    s = ensure_spd(h_arr @ p @ h_arr.T + r_arr)
    try:
        k = p @ h_arr.T @ np.linalg.inv(s)
    except np.linalg.LinAlgError:
        k = p @ h_arr.T @ np.linalg.pinv(s)
    x_new = x + k @ innov
    p_new = joseph_update(p, k, h_arr, r_arr)
    return x_new, p_new, innov, s, k


def update_nonlinear(
    x_pred: np.ndarray,
    p_pred: np.ndarray,
    z: np.ndarray,
    h_fn: Callable[[np.ndarray], np.ndarray],
    h_jac: Callable[[np.ndarray], np.ndarray],
    r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x_pred, dtype=np.float64).reshape(-1)
    p = ensure_spd(p_pred)
    h = np.asarray(h_jac(x), dtype=np.float64)
    r_arr = ensure_spd(r)
    z_arr = np.asarray(z, dtype=np.float64).reshape(-1)
    y_hat = np.asarray(h_fn(x), dtype=np.float64).reshape(-1)
    innov = z_arr - y_hat
    s = ensure_spd(h @ p @ h.T + r_arr)
    try:
        k = p @ h.T @ np.linalg.inv(s)
    except np.linalg.LinAlgError:
        k = p @ h.T @ np.linalg.pinv(s)
    x_new = x + k @ innov
    p_new = joseph_update(p, k, h, r_arr)
    return x_new, p_new, innov, s, k


def innovation_statistics(innovations: np.ndarray, s_seq: np.ndarray) -> dict[str, float]:
    innov = np.asarray(innovations, dtype=np.float64)
    if innov.ndim == 1:
        innov = innov.reshape(-1, 1)
    s_arr = np.asarray(s_seq, dtype=np.float64)
    dists = []
    for t in range(innov.shape[0]):
        s_t = s_arr[t] if s_arr.ndim == 3 else s_arr
        dists.append(mahalanobis(innov[t], s_t))
    d = np.asarray(dists, dtype=np.float64)
    return {
        "mean_innovation": float(np.mean(innov)),
        "std_innovation": float(np.std(innov)),
        "mean_mahalanobis": float(np.mean(d)) if d.size else 0.0,
        "max_mahalanobis": float(np.max(d)) if d.size else 0.0,
        "n": float(innov.shape[0]),
    }

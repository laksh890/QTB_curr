"""Kalman predict step and multi-step forecasting."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from iqrp.app.regimes.kalman.covariance import ensure_spd


def predict_state(
    x: np.ndarray,
    p: np.ndarray,
    f: np.ndarray,
    q: np.ndarray,
    *,
    b: np.ndarray | None = None,
    u: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear predict: ``x = F x + B u``, ``P = F P F' + Q``."""
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    f_arr = np.asarray(f, dtype=np.float64)
    q_arr = ensure_spd(q)
    x_pred = f_arr @ x_arr
    if b is not None and u is not None:
        x_pred = x_pred + np.asarray(b, dtype=np.float64) @ np.asarray(u, dtype=np.float64).reshape(
            -1
        )
    p_pred = ensure_spd(f_arr @ p @ f_arr.T + q_arr)
    return x_pred, p_pred


def predict_nonlinear(
    x: np.ndarray,
    p: np.ndarray,
    f_fn: Callable[[np.ndarray], np.ndarray],
    f_jac: Callable[[np.ndarray], np.ndarray],
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    x_pred = np.asarray(f_fn(x_arr), dtype=np.float64).reshape(-1)
    f = np.asarray(f_jac(x_arr), dtype=np.float64)
    p_pred = ensure_spd(f @ p @ f.T + ensure_spd(q))
    return x_pred, p_pred


def n_step_predict(
    x: np.ndarray,
    p: np.ndarray,
    f: np.ndarray,
    q: np.ndarray,
    *,
    horizon: int,
    b: np.ndarray | None = None,
    controls: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return stacked means ``(H, n)`` and covariances ``(H, n, n)``."""
    h = max(1, int(horizon))
    n = np.asarray(x, dtype=np.float64).reshape(-1).size
    means = np.empty((h, n), dtype=np.float64)
    covs = np.empty((h, n, n), dtype=np.float64)
    xi, pi = np.asarray(x, dtype=np.float64).reshape(-1), ensure_spd(p)
    for t in range(h):
        u = None if controls is None else np.asarray(controls[t], dtype=np.float64)
        xi, pi = predict_state(xi, pi, f, q, b=b, u=u)
        means[t] = xi
        covs[t] = pi
    return means, covs


def prediction_intervals(
    mean: np.ndarray,
    cov: np.ndarray,
    *,
    level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy import stats  # type: ignore[import-untyped]

    z = float(stats.norm.ppf(0.5 + level / 2.0))
    m = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.sqrt(np.clip(np.diag(ensure_spd(cov)), 1e-300, None))
    return m - z * std, m + z * std


def forecast_observation(
    x: np.ndarray,
    p: np.ndarray,
    h: np.ndarray,
    r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict observation mean and covariance."""
    h_arr = np.asarray(h, dtype=np.float64)
    y_hat = h_arr @ np.asarray(x, dtype=np.float64).reshape(-1)
    s = ensure_spd(h_arr @ p @ h_arr.T + ensure_spd(r))
    return y_hat, s

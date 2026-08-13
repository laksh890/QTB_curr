"""Rauch–Tung–Striebel Kalman smoother."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iqrp.app.regimes.kalman.covariance import ensure_spd
from iqrp.app.regimes.kalman.initialization import LinearGaussianSSM
from iqrp.app.regimes.kalman.linear import FilterTrace


@dataclass
class SmoothTrace:
    means: np.ndarray
    covs: np.ndarray
    gains: np.ndarray


def rts_smooth(
    trace: FilterTrace,
    system: LinearGaussianSSM,
    *,
    f_seq: np.ndarray | None = None,
) -> SmoothTrace:
    """Fixed-interval RTS smoother using filter predictions/updates."""
    means_f = trace.means
    covs_f = trace.covs
    pred_m = trace.pred_means
    pred_p = trace.pred_covs
    t_steps, n = means_f.shape
    means = means_f.copy()
    covs = covs_f.copy()
    gains = np.zeros((t_steps, n, n), dtype=np.float64)

    for t in range(t_steps - 2, -1, -1):
        f = system.f if f_seq is None else np.asarray(f_seq[t + 1], dtype=np.float64)
        p_pred = ensure_spd(pred_p[t + 1])
        try:
            g = covs_f[t] @ f.T @ np.linalg.inv(p_pred)
        except np.linalg.LinAlgError:
            g = covs_f[t] @ f.T @ np.linalg.pinv(p_pred)
        gains[t] = g
        means[t] = means_f[t] + g @ (means[t + 1] - pred_m[t + 1])
        covs[t] = ensure_spd(covs_f[t] + g @ (covs[t + 1] - p_pred) @ g.T)
    return SmoothTrace(means=means, covs=covs, gains=gains)

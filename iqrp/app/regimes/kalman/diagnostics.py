"""Diagnostics for Kalman filter runs."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.regimes.kalman.covariance import ensure_spd
from iqrp.app.regimes.kalman.initialization import LinearGaussianSSM
from iqrp.app.regimes.kalman.linear import FilterTrace
from iqrp.app.regimes.kalman.smoothing import SmoothTrace
from iqrp.app.regimes.kalman.update import innovation_statistics


class KalmanDiagnostics:
    def report(
        self,
        system: LinearGaussianSSM,
        trace: FilterTrace,
        *,
        smooth: SmoothTrace | None = None,
        history: list[float] | None = None,
    ) -> dict[str, Any]:
        innov_stats = innovation_statistics(trace.innovations, trace.innovation_covs)
        gains = np.asarray(trace.gains, dtype=np.float64)
        cov_traces = np.array([float(np.trace(c)) for c in trace.covs], dtype=np.float64)
        gain_norms = np.array(
            [float(np.linalg.norm(gains[t])) for t in range(gains.shape[0])], dtype=np.float64
        )
        # whiteness: lag-1 autocorrelation of innovations (primary channel)
        innov = np.asarray(trace.innovations, dtype=np.float64)
        if innov.ndim == 1:
            innov = innov.reshape(-1, 1)
        ac1 = _lag1_corr(innov[:, 0])
        # filter stability: eigenvalues of F and average gain
        eig_f = np.linalg.eigvals(system.f)
        spectral_radius = float(np.max(np.abs(eig_f)))
        stable = spectral_radius < 1.0 + 1e-6 or spectral_radius <= 1.0 + 1e-3
        # Joseph-form residual PSD check
        min_eig_p = float(min(np.min(np.linalg.eigvalsh(ensure_spd(c))) for c in trace.covs[: min(50, len(trace.covs))]))
        noise = {
            "q_diag": np.diag(ensure_spd(system.q)).tolist(),
            "r_diag": np.diag(ensure_spd(system.r)).tolist(),
        }
        if "q_final" in trace.metadata:
            noise["q_final_diag"] = np.diag(ensure_spd(trace.metadata["q_final"])).tolist()
        if "r_final" in trace.metadata:
            noise["r_final_diag"] = np.diag(ensure_spd(trace.metadata["r_final"])).tolist()

        out: dict[str, Any] = {
            "history": list(history or []),
            "log_likelihood": float(trace.log_likelihood),
            "innovation": innov_stats,
            "innovation_autocorr_lag1": ac1,
            "kalman_gain": {
                "mean_norm": float(np.mean(gain_norms)) if gain_norms.size else 0.0,
                "max_norm": float(np.max(gain_norms)) if gain_norms.size else 0.0,
                "final": gains[-1].tolist() if gains.size else [],
            },
            "state_covariance": {
                "mean_trace": float(np.mean(cov_traces)) if cov_traces.size else 0.0,
                "final_trace": float(cov_traces[-1]) if cov_traces.size else 0.0,
                "min_eigenvalue": min_eig_p,
            },
            "prediction_error": {
                "mse": float(np.mean(innov**2)),
                "rmse": float(np.sqrt(np.mean(innov**2))),
            },
            "noise_estimates": noise,
            "filter_stability": {
                "spectral_radius_f": spectral_radius,
                "stable": bool(stable),
                "positive_definite_cov": min_eig_p > 0,
            },
            "n_states": system.n_states,
            "n_obs": system.n_obs,
            "application": system.application,
            "filter": trace.metadata.get("filter", "linear"),
        }
        if smooth is not None:
            out["smoothed"] = {
                "mean_trace": float(np.mean([np.trace(c) for c in smooth.covs])),
                "final_mean": smooth.means[-1].tolist(),
            }
        return out


def _lag1_corr(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if v.size < 3:
        return 0.0
    a, b = v[:-1], v[1:]
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt(np.sum(a**2) * np.sum(b**2)))
    if den < 1e-300:
        return 0.0
    return float(np.sum(a * b) / den)

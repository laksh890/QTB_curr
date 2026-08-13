"""Evaluation metrics for Kalman filters."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.regimes.kalman.covariance import ensure_spd, mahalanobis
from iqrp.app.regimes.kalman.linear import FilterTrace
from iqrp.app.regimes.kalman.smoothing import SmoothTrace
from iqrp.app.regimes.kalman.update import innovation_statistics


class KalmanEvaluator:
    def evaluate(
        self,
        *,
        observations: np.ndarray,
        trace: FilterTrace,
        smooth: SmoothTrace | None = None,
        true_states: np.ndarray | None = None,
        n_params: int = 0,
    ) -> dict[str, Any]:
        y = np.asarray(observations, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        innov_stats = innovation_statistics(trace.innovations, trace.innovation_covs)
        mse_innov = float(np.mean(trace.innovations**2))
        rmse_innov = float(np.sqrt(mse_innov))
        # one-step observation prediction error using innovation
        metrics: dict[str, float] = {
            "log_likelihood": float(trace.log_likelihood),
            "mse_innovation": mse_innov,
            "rmse_innovation": rmse_innov,
            "mean_abs_innovation": float(np.mean(np.abs(trace.innovations))),
            "mean_mahalanobis": innov_stats["mean_mahalanobis"],
            "max_mahalanobis": innov_stats["max_mahalanobis"],
            "n_obs": float(y.shape[0]),
            "n_params": float(n_params),
        }
        if n_params > 0 and y.shape[0] > 0:
            from iqrp.app.math.probability.likelihood import aic as aic_score, bic as bic_score

            metrics["aic"] = aic_score(-trace.log_likelihood, n_params)
            metrics["bic"] = bic_score(-trace.log_likelihood, n_params, y.shape[0])

        details: dict[str, Any] = {"innovation_statistics": innov_stats}
        if true_states is not None:
            truth = np.asarray(true_states, dtype=np.float64)
            if truth.ndim == 1:
                truth = truth.reshape(-1, 1)
            est = trace.means[:, : truth.shape[1]]
            err = est - truth
            metrics["state_mse"] = float(np.mean(err**2))
            metrics["state_rmse"] = float(np.sqrt(metrics["state_mse"]))
            metrics["state_mae"] = float(np.mean(np.abs(err)))
            # correlation of primary state
            if truth.shape[0] > 2:
                a, b = est[:, 0], truth[:, 0]
                if float(np.std(a)) < 1e-15 or float(np.std(b)) < 1e-15:
                    metrics["state_corr"] = 0.0
                else:
                    c = np.corrcoef(a, b)[0, 1]
                    metrics["state_corr"] = float(c) if np.isfinite(c) else 0.0
            if smooth is not None:
                s_est = smooth.means[:, : truth.shape[1]]
                s_err = s_est - truth
                metrics["smooth_mse"] = float(np.mean(s_err**2))
                metrics["smooth_rmse"] = float(np.sqrt(metrics["smooth_mse"]))
            # coverage: fraction of true states within 95% ellipse of filtered cov
            covered = 0
            for t in range(min(truth.shape[0], trace.covs.shape[0])):
                d = mahalanobis(truth[t] - est[t], ensure_spd(trace.covs[t][: truth.shape[1], : truth.shape[1]]))
                # chi2 95% approx for dim d: use 2*d as loose threshold for d dims via sqrt
                thresh = np.sqrt(float(stats_chi2_crit(truth.shape[1])))
                if d <= thresh:
                    covered += 1
            metrics["cov_coverage_95"] = covered / max(truth.shape[0], 1)
        return {"metrics": metrics, "details": details}


def stats_chi2_crit(df: int, level: float = 0.95) -> float:
    try:
        from scipy import stats  # type: ignore[import-untyped]

        return float(stats.chi2.ppf(level, df=max(df, 1)))
    except Exception:
        # crude fallback
        return float(max(df, 1) * 4.0)

"""Evaluation metrics for particle filters."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.regimes.particle.particle import FilterTrace
from iqrp.app.regimes.particle.prediction import particle_diversity
from iqrp.app.regimes.particle.smoothing import SmoothTrace
from iqrp.app.regimes.particle.weighting import effective_sample_size


class ParticleEvaluator:
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
        mean_ess = float(np.mean(trace.ess)) if trace.ess.size else 0.0
        resample_rate = float(np.mean(trace.resampled)) if trace.resampled.size else 0.0
        diversities = [particle_diversity(c) for c in trace.clouds] if trace.clouds else [0.0]
        metrics: dict[str, float] = {
            "log_likelihood": float(trace.log_likelihood),
            "mean_ess": mean_ess,
            "min_ess": float(np.min(trace.ess)) if trace.ess.size else 0.0,
            "resample_rate": resample_rate,
            "mean_diversity": float(np.mean(diversities)),
            "n_obs": float(y.shape[0]),
            "n_params": float(n_params),
        }
        # one-step prediction error vs primary observation
        if trace.means.size:
            pred = trace.means[:, 0]
            err = pred - y[: pred.size, 0]
            metrics["mse_prediction"] = float(np.mean(err**2))
            metrics["rmse_prediction"] = float(np.sqrt(metrics["mse_prediction"]))
        if n_params > 0 and y.shape[0] > 0:
            from iqrp.app.math.probability.likelihood import aic as aic_score, bic as bic_score

            metrics["aic"] = aic_score(-trace.log_likelihood, n_params)
            metrics["bic"] = bic_score(-trace.log_likelihood, n_params, y.shape[0])
        details: dict[str, Any] = {
            "ess": trace.ess,
            "resampled": trace.resampled,
        }
        if true_states is not None:
            truth = np.asarray(true_states, dtype=np.float64)
            if truth.ndim == 1:
                truth = truth.reshape(-1, 1)
            est = trace.means[:, : truth.shape[1]]
            err = est - truth
            metrics["state_mse"] = float(np.mean(err**2))
            metrics["state_rmse"] = float(np.sqrt(metrics["state_mse"]))
            metrics["state_mae"] = float(np.mean(np.abs(err)))
            if truth.shape[0] > 2:
                a, b = est[:, 0], truth[:, 0]
                if float(np.std(a)) < 1e-15 or float(np.std(b)) < 1e-15:
                    metrics["state_corr"] = 0.0
                else:
                    c = np.corrcoef(a, b)[0, 1]
                    metrics["state_corr"] = float(c) if np.isfinite(c) else 0.0
            if smooth is not None:
                s_est = smooth.means[:, : truth.shape[1]]
                metrics["smooth_mse"] = float(np.mean((s_est - truth) ** 2))
            # crude posterior calibration: fraction of truth in 95% particle quantiles
            covered = 0
            for t, cloud in enumerate(trace.clouds[: truth.shape[0]]):
                w = cloud.weights
                x = cloud.states[:, 0]
                order = np.argsort(x)
                xs, ws = x[order], w[order]
                cdf = np.cumsum(ws)
                cdf = cdf / max(float(cdf[-1]), 1e-300)
                lo = float(np.interp(0.025, cdf, xs))
                hi = float(np.interp(0.975, cdf, xs))
                if lo <= truth[t, 0] <= hi:
                    covered += 1
            metrics["posterior_coverage_95"] = covered / max(truth.shape[0], 1)
        return {"metrics": metrics, "details": details}

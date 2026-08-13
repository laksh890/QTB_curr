"""Diagnostics for particle filter runs."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.regimes.particle.particle import FilterTrace
from iqrp.app.regimes.particle.prediction import particle_diversity
from iqrp.app.regimes.particle.propagation import TransitionModel
from iqrp.app.regimes.particle.smoothing import SmoothTrace
from iqrp.app.regimes.particle.weighting import weight_diagnostics, weight_entropy


class ParticleDiagnostics:
    def report(
        self,
        model: TransitionModel,
        trace: FilterTrace,
        *,
        smooth: SmoothTrace | None = None,
        history: list[float] | None = None,
    ) -> dict[str, Any]:
        ess = np.asarray(trace.ess, dtype=np.float64)
        diversities = [particle_diversity(c) for c in trace.clouds] if trace.clouds else []
        entropies = [weight_entropy(c.weights) for c in trace.clouds] if trace.clouds else []
        final_w = weight_diagnostics(trace.clouds[-1]) if trace.clouds else {}
        cov_traces = (
            np.array([float(np.trace(c)) for c in trace.covs], dtype=np.float64)
            if trace.covs.size
            else np.array([])
        )
        out: dict[str, Any] = {
            "history": list(history or []),
            "log_likelihood": float(trace.log_likelihood),
            "ess": {
                "mean": float(np.mean(ess)) if ess.size else 0.0,
                "min": float(np.min(ess)) if ess.size else 0.0,
                "final": float(ess[-1]) if ess.size else 0.0,
                "timeline": ess,
            },
            "weight_entropy": {
                "mean": float(np.mean(entropies)) if entropies else 0.0,
                "final": float(entropies[-1]) if entropies else 0.0,
            },
            "particle_degeneracy": {
                "mean_degeneracy_ratio": float(
                    np.mean([1.0 - e / max(trace.clouds[i].n_particles, 1) for i, e in enumerate(ess)])
                )
                if ess.size and trace.clouds
                else 0.0,
                "final": final_w,
            },
            "resampling_history": {
                "rate": float(np.mean(trace.resampled)) if trace.resampled.size else 0.0,
                "timeline": trace.resampled,
                "n_events": int(np.sum(trace.resampled)) if trace.resampled.size else 0,
            },
            "particle_diversity": {
                "mean": float(np.mean(diversities)) if diversities else 0.0,
                "min": float(np.min(diversities)) if diversities else 0.0,
                "final": float(diversities[-1]) if diversities else 0.0,
            },
            "posterior_uncertainty": {
                "mean_cov_trace": float(np.mean(cov_traces)) if cov_traces.size else 0.0,
                "final_cov_trace": float(cov_traces[-1]) if cov_traces.size else 0.0,
            },
            "prediction_error": {
                "mean_abs_state": float(np.mean(np.abs(trace.means))) if trace.means.size else 0.0,
            },
            "application": model.application,
            "filter": trace.metadata.get("filter", "bootstrap"),
            "n_states": model.n_states,
        }
        if smooth is not None:
            out["smoothed"] = {
                "mean_final": smooth.means[-1].tolist() if smooth.means.size else [],
                "n_trajectories": int(smooth.trajectories.shape[0]) if smooth.trajectories.size else 0,
            }
        return out

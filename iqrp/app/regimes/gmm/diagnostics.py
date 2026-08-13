"""Diagnostics for Gaussian mixture regime models."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math.statistics.entropy import entropy
from iqrp.app.regimes.gmm.mixture import GaussianMixtureParams
from iqrp.app.regimes.gmm.prediction import (
    detect_outliers,
    regime_occupancy,
    regime_persistence,
    regime_similarity,
    transition_frequency,
)


class GMMDiagnostics:
    def report(
        self,
        params: GaussianMixtureParams,
        *,
        x: np.ndarray,
        responsibilities: np.ndarray,
        history: list[float] | None = None,
        density_quantile: float = 0.01,
        rare_occupancy: float = 0.05,
    ) -> dict[str, Any]:
        hard = np.argmax(responsibilities, axis=1)
        occ = regime_occupancy(responsibilities)
        rare = [i for i, o in enumerate(occ) if o < rare_occupancy]
        outliers = detect_outliers(params, x, density_quantile=density_quantile)
        return {
            "history": list(history or []),
            "weights": params.weights,
            "means": params.means,
            "covars": params.covars,
            "covariance_type": params.covariance_type,
            "occupancy": occ,
            "persistence": regime_persistence(hard, params.n_components),
            "transition_frequency": transition_frequency(hard, params.n_components),
            "similarity": regime_similarity(params.means),
            "mean_entropy": float(np.mean([entropy(row) for row in responsibilities])),
            "rare_clusters": rare,
            "outliers": {
                "n_outliers": outliers["n_outliers"],
                "threshold": outliers["threshold"],
            },
            "n_components": params.n_components,
            "n_features": params.n_features,
            "n_params": params.n_params(),
        }

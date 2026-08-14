"""Visualization helpers for forecast intelligence (data-oriented, no GUI)."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.forecasting.intelligence.config import VisualizationConfig
from iqrp.app.forecasting.intelligence.ranking import RankedModel


def leaderboard_chart(
    ranked: list[RankedModel], *, config: VisualizationConfig | None = None
) -> dict[str, Any]:
    cfg = config or VisualizationConfig()
    top = ranked[: max(cfg.top_n, 1)]
    return {
        "type": "bar",
        "labels": [r.name for r in top],
        "values": [r.score for r in top],
        "title": "Model Leaderboard",
    }


def forecast_chart(
    timestamps: list[Any] | np.ndarray,
    actual: np.ndarray | None,
    predicted: np.ndarray,
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict[str, Any]:
    ts = [str(t) for t in list(timestamps)]
    payload: dict[str, Any] = {
        "type": "line",
        "timestamps": ts,
        "predicted": np.asarray(predicted, dtype=np.float64).reshape(-1).tolist(),
        "title": "Forecast vs Actual",
    }
    if actual is not None:
        payload["actual"] = np.asarray(actual, dtype=np.float64).reshape(-1).tolist()
    if lower is not None and upper is not None:
        payload["lower"] = np.asarray(lower, dtype=np.float64).reshape(-1).tolist()
        payload["upper"] = np.asarray(upper, dtype=np.float64).reshape(-1).tolist()
    return payload


def drift_chart(feature_psi: dict[str, float]) -> dict[str, Any]:
    keys = sorted(feature_psi.keys())
    return {
        "type": "bar",
        "labels": keys,
        "values": [feature_psi[k] for k in keys],
        "title": "Feature Drift (PSI)",
    }


def residual_hist(residuals: np.ndarray, *, bins: int = 20) -> dict[str, Any]:
    r = np.asarray(residuals, dtype=np.float64).reshape(-1)
    hist, edges = np.histogram(r, bins=bins)
    return {
        "type": "histogram",
        "counts": hist.tolist(),
        "edges": edges.tolist(),
        "title": "Residual Distribution",
    }

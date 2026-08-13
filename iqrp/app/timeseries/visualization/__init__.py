"""Data-oriented visualization payloads (no GUI dependency)."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, ChangePointResult, DecompositionResult


def decomposition_chart(result: DecompositionResult) -> dict[str, Any]:
    return {
        "type": "decomposition",
        "method": result.method,
        "observed": result.observed.tolist(),
        "trend": result.trend.tolist(),
        "seasonal": result.seasonal.tolist(),
        "residual": result.residual.tolist(),
    }


def spectrum_chart(frequencies: np.ndarray, power: np.ndarray) -> dict[str, Any]:
    return {
        "type": "spectrum",
        "frequencies": np.asarray(frequencies, dtype=np.float64).tolist(),
        "power": np.asarray(power, dtype=np.float64).tolist(),
        "title": "Power Spectrum",
    }


def change_point_chart(x: np.ndarray, result: ChangePointResult) -> dict[str, Any]:
    return {
        "type": "change_points",
        "series": np.asarray(x, dtype=np.float64).tolist(),
        "indices": list(result.indices),
        "method": result.method,
    }


def acf_chart(result: AnalysisResult) -> dict[str, Any]:
    vals = result.value
    if isinstance(vals, np.ndarray):
        vals = vals.tolist()
    return {
        "type": "acf",
        "values": vals,
        "lags": (result.metadata or {}).get("lags"),
        "method": result.method,
    }


def anomaly_chart(x: np.ndarray, indices: list[int]) -> dict[str, Any]:
    return {
        "type": "anomalies",
        "series": np.asarray(x, dtype=np.float64).tolist(),
        "indices": list(indices),
    }

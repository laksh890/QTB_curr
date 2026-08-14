"""Uncertainty quantification helpers for forecasts."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.forecasting.base.prediction import DistributionForecast, QuantileForecast
from iqrp.app.math.statistics.entropy import entropy


def predictive_entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return np.asarray([float(entropy(row)) for row in p], dtype=np.float64)


def quantile_from_samples(
    samples: np.ndarray,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    *,
    horizon: int | None = None,
) -> list[QuantileForecast]:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    h = arr.shape[1]
    out: list[QuantileForecast] = []
    for t in range(h):
        qmap = {float(q): float(np.quantile(arr[:, t], q)) for q in quantiles}
        out.append(QuantileForecast(quantiles=qmap, horizon=t + 1))
    if horizon is not None:
        return out[: int(horizon)]
    return out


def distribution_from_samples(samples: np.ndarray, *, horizon: int = 1) -> DistributionForecast:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        mean = float(np.mean(arr))
        var = float(np.var(arr))
        return DistributionForecast(mean=mean, variance=var, samples=arr, horizon=horizon)
    return DistributionForecast(
        mean=np.mean(arr, axis=0),
        variance=np.var(arr, axis=0),
        samples=arr,
        horizon=horizon,
    )


def forecast_uncertainty_report(
    values: np.ndarray,
    *,
    intervals_width: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    report: dict[str, Any] = {
        "n_steps": int(v.size),
        "value_std": float(np.std(v)) if v.size else 0.0,
        "value_range": float(np.ptp(v)) if v.size else 0.0,
    }
    if intervals_width is not None:
        w = np.asarray(intervals_width, dtype=np.float64).reshape(-1)
        report["mean_interval_width"] = float(np.mean(w)) if w.size else 0.0
        report["max_interval_width"] = float(np.max(w)) if w.size else 0.0
    if probabilities is not None:
        ent = predictive_entropy(probabilities)
        report["mean_entropy"] = float(np.mean(ent)) if ent.size else 0.0
    return report

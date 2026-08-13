"""Residual analysis for forecast diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class ResidualReport:
    residuals: np.ndarray
    mean: float
    std: float
    skewness: float
    kurtosis: float
    acf: list[float]
    bias: float
    mae: float
    rmse: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "residuals": self.residuals.tolist(),
            "mean": self.mean,
            "std": self.std,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "acf": list(self.acf),
            "bias": self.bias,
            "mae": self.mae,
            "rmse": self.rmse,
            "metadata": dict(self.metadata),
        }


def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = min(yt.size, yp.size)
    return yt[:n] - yp[:n]


def _moments(r: np.ndarray) -> tuple[float, float, float, float]:
    if r.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mu = float(np.mean(r))
    sd = float(np.std(r))
    if sd < 1e-12:
        return mu, sd, 0.0, 0.0
    z = (r - mu) / sd
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4) - 3.0)
    return mu, sd, skew, kurt


def autocorrelation(series: np.ndarray, *, max_lag: int = 10) -> list[float]:
    x = np.asarray(series, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return []
    x = x - np.mean(x)
    var = float(np.dot(x, x))
    if var <= 1e-300:
        return [0.0] * min(max_lag, x.size - 1)
    out: list[float] = []
    for lag in range(1, min(max_lag, x.size - 1) + 1):
        out.append(float(np.dot(x[:-lag], x[lag:]) / var))
    return out


def residual_analysis(
    y_true: np.ndarray, y_pred: np.ndarray, *, max_lag: int = 10
) -> ResidualReport:
    r = compute_residuals(y_true, y_pred)
    mu, sd, skew, kurt = _moments(r)
    return ResidualReport(
        residuals=r,
        mean=mu,
        std=sd,
        skewness=skew,
        kurtosis=kurt,
        acf=autocorrelation(r, max_lag=max_lag),
        bias=mu,
        mae=float(np.mean(np.abs(r))) if r.size else 0.0,
        rmse=float(np.sqrt(np.mean(r**2))) if r.size else 0.0,
    )


def forecast_error_by_horizon(
    y_true: np.ndarray, y_pred: np.ndarray, *, horizons: int
) -> dict[int, float]:
    """``y_pred`` shape ``(T, H)`` or flat with horizon blocks."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    h = max(int(horizons), 1)
    out: dict[int, float] = {}
    if yp.ndim == 2:
        for j in range(min(h, yp.shape[1])):
            n = min(yt.reshape(-1).size, yp.shape[0])
            err = yt.reshape(-1)[:n] - yp[:n, j]
            out[j + 1] = float(np.mean(np.abs(err)))
        return out
    # flat: evaluate overall
    r = compute_residuals(yt, yp)
    out[1] = float(np.mean(np.abs(r))) if r.size else 0.0
    return out

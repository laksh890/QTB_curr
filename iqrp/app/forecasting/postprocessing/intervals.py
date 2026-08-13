"""Prediction and confidence interval construction."""

from __future__ import annotations

from typing import Literal

import numpy as np

from iqrp.app.forecasting.base.prediction import PredictionInterval

IntervalMethod = Literal["residual", "quantile", "gaussian"]


def residual_intervals(
    predictions: np.ndarray,
    *,
    residual_std: float | np.ndarray | None = None,
    level: float = 0.95,
    kind: str = "prediction",
) -> list[PredictionInterval]:
    preds = np.asarray(predictions, dtype=np.float64).reshape(-1)
    if residual_std is None:
        # heuristic: 5% of |prediction| + small floor
        sigma = np.maximum(0.05 * np.abs(preds), 1e-3)
    else:
        sigma = np.broadcast_to(np.asarray(residual_std, dtype=np.float64), preds.shape).copy()
    z = _z_score(level)
    out: list[PredictionInterval] = []
    for i, mu in enumerate(preds):
        s = float(sigma[i])
        out.append(
            PredictionInterval(
                lower=float(mu - z * s),
                upper=float(mu + z * s),
                level=level,
                kind=kind,
                midpoint=float(mu),
            )
        )
    return out


def gaussian_intervals(
    mean: np.ndarray,
    variance: np.ndarray,
    *,
    level: float = 0.95,
    kind: str = "prediction",
) -> list[PredictionInterval]:
    mu = np.asarray(mean, dtype=np.float64).reshape(-1)
    var = np.asarray(variance, dtype=np.float64).reshape(-1)
    sigma = np.sqrt(np.clip(var, 0.0, None))
    return residual_intervals(mu, residual_std=sigma, level=level, kind=kind)


def quantile_intervals(
    quantile_paths: dict[float, np.ndarray],
    *,
    level: float = 0.95,
    kind: str = "prediction",
) -> list[PredictionInterval]:
    """Build intervals from quantile forecasts bracketing ``level``."""
    alpha = (1.0 - level) / 2.0
    lo_q = min(quantile_paths, key=lambda q: abs(q - alpha))
    hi_q = min(quantile_paths, key=lambda q: abs(q - (1.0 - alpha)))
    mid_q = min(quantile_paths, key=lambda q: abs(q - 0.5))
    lo = np.asarray(quantile_paths[lo_q], dtype=np.float64).reshape(-1)
    hi = np.asarray(quantile_paths[hi_q], dtype=np.float64).reshape(-1)
    mid = np.asarray(quantile_paths[mid_q], dtype=np.float64).reshape(-1)
    n = min(lo.size, hi.size, mid.size)
    return [
        PredictionInterval(
            lower=float(lo[i]),
            upper=float(hi[i]),
            level=level,
            kind=kind,
            midpoint=float(mid[i]),
        )
        for i in range(n)
    ]


def confidence_intervals_from_samples(
    samples: np.ndarray,
    *,
    level: float = 0.95,
) -> list[PredictionInterval]:
    """``samples`` shape ``(n_samples, horizon)``."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    alpha = (1.0 - level) / 2.0
    lo = np.quantile(arr, alpha, axis=0)
    hi = np.quantile(arr, 1.0 - alpha, axis=0)
    mid = np.mean(arr, axis=0)
    return [
        PredictionInterval(
            lower=float(lo[i]),
            upper=float(hi[i]),
            level=level,
            kind="confidence",
            midpoint=float(mid[i]),
        )
        for i in range(lo.size)
    ]


def _z_score(level: float) -> float:
    # approximate inverse erf for common levels
    table = {
        0.80: 1.2815515655446004,
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
    if level in table:
        return table[level]
    # rough fallback
    return 1.959963984540054

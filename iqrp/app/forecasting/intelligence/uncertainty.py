"""Uncertainty quantification for forecast intelligence."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.forecasting.base.prediction import PredictionInterval


def prediction_intervals(
    point: np.ndarray,
    *,
    residual_std: float,
    level: float = 0.95,
) -> list[PredictionInterval]:
    # approximate normal quantile without requiring SciPy at import time
    try:
        from scipy.stats import norm

        z = float(norm.ppf(0.5 + level / 2.0))
    except Exception:  # noqa: BLE001
        # common levels
        z = {0.9: 1.64485, 0.95: 1.95996, 0.99: 2.57583}.get(round(level, 2), 1.96)
    path = np.asarray(point, dtype=np.float64).reshape(-1)
    return [
        PredictionInterval(lower=float(v - z * residual_std), upper=float(v + z * residual_std), level=level)
        for v in path
    ]


def ensemble_uncertainty(preds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not preds:
        return {"mean": np.asarray([]), "std": np.asarray([]), "agreement": np.asarray([])}
    stack = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in preds.values()], axis=0)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    # agreement: fraction within 1 std of mean
    agreement = np.mean(np.abs(stack - mean) <= (std + 1e-8), axis=0)
    return {"mean": mean, "std": std, "agreement": agreement, "epistemic": std, "aleatoric": std * 0.5}


def forecast_distribution(
    point: np.ndarray,
    residual_std: float,
    *,
    n_samples: int = 100,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    gen = rng or np.random.default_rng(0)
    path = np.asarray(point, dtype=np.float64).reshape(-1)
    return gen.normal(path, max(residual_std, 1e-8), size=(n_samples, path.size))


def model_agreement(preds: dict[str, np.ndarray], *, tol: float = 0.1) -> float:
    if len(preds) < 2:
        return 1.0
    stack = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in preds.values()], axis=0)
    mean = stack.mean(axis=0)
    return float(np.mean(np.abs(stack - mean) <= tol))

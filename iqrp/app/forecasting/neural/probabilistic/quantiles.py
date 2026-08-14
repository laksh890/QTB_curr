"""Quantile helpers."""

from __future__ import annotations

import numpy as np

from iqrp.app.forecasting.neural.probabilistic.distributions import (
    gaussian_quantiles,
    prediction_intervals_from_quantiles,
    student_t_quantiles,
)


def extract_point_forecast(
    pred: np.ndarray, *, task: str, alphas: tuple[float, ...] = (0.1, 0.5, 0.9)
) -> np.ndarray:
    p = np.asarray(pred, dtype=np.float64)
    if task == "quantile" and p.ndim >= 2:
        # median quantile
        mid = int(np.argmin(np.abs(np.asarray(alphas) - 0.5)))
        return p[..., mid]
    if (task == "distribution" or p.ndim == 3) and p.shape[-1] >= 2:
        return p[..., 0]
    return p.reshape(p.shape[0], -1) if p.ndim > 1 else p


def quantiles_from_prediction(
    pred: np.ndarray,
    *,
    task: str,
    alphas: tuple[float, ...] = (0.1, 0.5, 0.9),
    distribution: str = "gaussian",
) -> np.ndarray:
    p = np.asarray(pred, dtype=np.float64)
    if task == "quantile" and p.ndim >= 2 and p.shape[-1] == len(alphas):
        return p
    if p.ndim >= 2 and p.shape[-1] >= 2:
        mu, log_s = p[..., 0], p[..., 1]
        sigma = np.exp(log_s)
        if distribution == "student_t":
            return student_t_quantiles(mu, sigma, alphas=alphas)
        return gaussian_quantiles(mu, sigma, alphas=alphas)
    # residual-scale fallback
    mu = extract_point_forecast(p, task=task, alphas=alphas)
    sigma = np.maximum(0.1 * np.abs(mu), 1e-3)
    return gaussian_quantiles(mu, sigma, alphas=alphas)


def interval_from_prediction(
    pred: np.ndarray,
    *,
    task: str,
    alphas: tuple[float, ...] = (0.1, 0.5, 0.9),
    distribution: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray]:
    q = quantiles_from_prediction(pred, task=task, alphas=alphas, distribution=distribution)
    return prediction_intervals_from_quantiles(q, alphas)

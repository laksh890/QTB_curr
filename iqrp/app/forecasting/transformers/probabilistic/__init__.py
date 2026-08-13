"""Probabilistic output helpers for transformers."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.forecasting.neural.probabilistic.distributions import (
    gaussian_quantiles,
    sample_gaussian,
    student_t_quantiles,
)
from iqrp.app.forecasting.neural.base.torch_utils import has_torch


def mixture_density_params(pred: np.ndarray, n_mixtures: int = 3) -> dict[str, np.ndarray]:
    """Parse flat MDN output (B, H, K*3) -> pi, mu, log_sigma."""
    p = np.asarray(pred, dtype=np.float64)
    if p.ndim == 2:
        p = p[:, None, :]
    k = n_mixtures
    flat = p.reshape(p.shape[0], p.shape[1], k, 3)
    logits, mu, log_s = flat[..., 0], flat[..., 1], flat[..., 2]
    # softmax pi
    logits = logits - logits.max(axis=-1, keepdims=True)
    pi = np.exp(logits)
    pi = pi / np.maximum(pi.sum(axis=-1, keepdims=True), 1e-8)
    return {"pi": pi, "mu": mu, "sigma": np.exp(log_s)}


def mixture_mean(pred: np.ndarray, n_mixtures: int = 3) -> np.ndarray:
    d = mixture_density_params(pred, n_mixtures)
    return np.sum(d["pi"] * d["mu"], axis=-1)


def gaussian_head_quantiles(pred: np.ndarray, alphas: tuple[float, ...] = (0.1, 0.5, 0.9)) -> np.ndarray:
    p = np.asarray(pred, dtype=np.float64)
    if p.ndim >= 2 and p.shape[-1] >= 2:
        return gaussian_quantiles(p[..., 0], np.exp(p[..., 1]), alphas=alphas)
    mu = p.reshape(p.shape[0], -1)
    return gaussian_quantiles(mu, np.maximum(0.1 * np.abs(mu), 1e-3), alphas=alphas)


def student_t_head_quantiles(pred: np.ndarray, alphas: tuple[float, ...] = (0.1, 0.5, 0.9)) -> np.ndarray:
    p = np.asarray(pred, dtype=np.float64)
    if p.ndim >= 2 and p.shape[-1] >= 2:
        return student_t_quantiles(p[..., 0], np.exp(p[..., 1]), alphas=alphas)
    mu = p.reshape(p.shape[0], -1)
    return student_t_quantiles(mu, np.maximum(0.1 * np.abs(mu), 1e-3), alphas=alphas)


__all__ = [
    "mixture_density_params",
    "mixture_mean",
    "gaussian_head_quantiles",
    "student_t_head_quantiles",
    "sample_gaussian",
]

"""Probabilistic distributions, quantiles, and uncertainty utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.forecasting.neural.base.torch_utils import from_tensor, has_torch, to_tensor


def gaussian_quantiles(mu: np.ndarray, sigma: np.ndarray, alphas: tuple[float, ...] = (0.1, 0.5, 0.9)) -> np.ndarray:
    from scipy.stats import norm

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    qs = []
    for a in alphas:
        qs.append(mu + sigma * float(norm.ppf(a)))
    return np.stack(qs, axis=-1)


def student_t_quantiles(
    mu: np.ndarray, scale: np.ndarray, *, df: float = 5.0, alphas: tuple[float, ...] = (0.1, 0.5, 0.9)
) -> np.ndarray:
    from scipy.stats import t as student_t

    mu = np.asarray(mu, dtype=np.float64)
    scale = np.maximum(np.asarray(scale, dtype=np.float64), 1e-8)
    qs = []
    for a in alphas:
        qs.append(mu + scale * float(student_t.ppf(a, df)))
    return np.stack(qs, axis=-1)


def prediction_intervals_from_quantiles(quantiles: np.ndarray, alphas: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    q = np.asarray(quantiles, dtype=np.float64)
    # assume first/last are lower/upper if sorted alphas
    order = np.argsort(alphas)
    lo = q[..., order[0]]
    hi = q[..., order[-1]]
    return lo, hi


def aleatoric_from_gaussian(pred: np.ndarray) -> np.ndarray:
    """pred (..., 2) -> sigma."""
    p = np.asarray(pred, dtype=np.float64)
    if p.shape[-1] < 2:
        return np.zeros(p.shape[:-1], dtype=np.float64)
    return np.exp(p[..., 1])


def epistemic_mc_dropout(module: Any, X: np.ndarray, *, n_samples: int = 20, device: Any = None) -> tuple[np.ndarray, np.ndarray]:
    """Enable dropout at inference for epistemic uncertainty."""
    if not has_torch():
        y = np.asarray(module(X) if callable(module) else X.mean(axis=1), dtype=np.float64)
        return y, np.zeros_like(y)
    import torch

    was_training = module.training
    module.train()  # keep dropout on
    preds = []
    with torch.no_grad():
        xb = to_tensor(X, device)
        for _ in range(max(int(n_samples), 1)):
            out = module(xb)
            if isinstance(out, (tuple, list)):
                out = out[0]
            arr = from_tensor(out)
            if arr.ndim == 3 and arr.shape[-1] >= 1:
                arr = arr[..., 0]
            preds.append(arr)
    module.train(was_training)
    stack = np.stack(preds, axis=0)
    return stack.mean(axis=0), stack.std(axis=0)


def sample_gaussian(mu: np.ndarray, sigma: np.ndarray, n: int = 50, rng: np.random.Generator | None = None) -> np.ndarray:
    gen = rng or np.random.default_rng(0)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-8)
    return gen.normal(mu, sigma, size=(n,) + mu.shape)

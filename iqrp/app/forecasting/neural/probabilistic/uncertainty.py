"""Uncertainty aggregation helpers."""

from __future__ import annotations

import numpy as np

from iqrp.app.forecasting.neural.probabilistic.distributions import (
    aleatoric_from_gaussian,
    epistemic_mc_dropout,
)


def total_uncertainty(
    module,
    X: np.ndarray,
    pred: np.ndarray,
    *,
    mc_dropout: bool = False,
    n_samples: int = 20,
    device=None,
) -> dict[str, np.ndarray]:
    aleatoric = aleatoric_from_gaussian(pred) if pred.ndim >= 2 and pred.shape[-1] >= 2 else np.zeros(pred.shape[:2] if pred.ndim > 1 else pred.shape)
    if mc_dropout:
        mean, epistemic = epistemic_mc_dropout(module, X, n_samples=n_samples, device=device)
    else:
        mean = pred[..., 0] if pred.ndim >= 2 and pred.shape[-1] >= 2 else pred
        epistemic = np.zeros_like(aleatoric)
    total = np.sqrt(np.maximum(aleatoric, 0) ** 2 + np.maximum(epistemic, 0) ** 2)
    return {"mean": mean, "aleatoric": aleatoric, "epistemic": epistemic, "total": total}

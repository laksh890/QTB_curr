"""Blending utilities for holdout-weighted ensembles."""

from __future__ import annotations

import numpy as np


def blend_predictions(
    preds: dict[str, np.ndarray],
    *,
    scores: dict[str, float] | None = None,
    min_weight: float = 0.05,
) -> np.ndarray:
    names = list(preds)
    if not names:
        return np.asarray([])
    if scores is None:
        w = np.ones(len(names)) / len(names)
    else:
        inv = np.asarray(
            [1.0 / max(float(scores.get(n, 1.0)), 1e-6) for n in names], dtype=np.float64
        )
        inv = np.maximum(inv, min_weight)
        w = inv / inv.sum()
    X = np.column_stack([np.asarray(preds[n], dtype=np.float64).reshape(-1) for n in names])
    return X @ w


def holdout_blend_weights(
    preds: dict[str, np.ndarray],
    y_holdout: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(y_holdout, dtype=np.float64).reshape(-1)
    weights: dict[str, float] = {}
    for name, p in preds.items():
        err = float(np.mean((np.asarray(p).reshape(-1)[: y.size] - y) ** 2))
        weights[name] = 1.0 / max(err, 1e-8)
    s = sum(weights.values()) or 1.0
    return {k: v / s for k, v in weights.items()}

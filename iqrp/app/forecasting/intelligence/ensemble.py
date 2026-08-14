"""Ensemble construction for forecast intelligence."""

from __future__ import annotations

import numpy as np

from iqrp.app.forecasting.intelligence.blending import blend_predictions
from iqrp.app.forecasting.intelligence.config import EnsembleConfig
from iqrp.app.forecasting.intelligence.gating import moe_combine
from iqrp.app.forecasting.intelligence.stacking import stack_predictions


def weighted_average(
    preds: dict[str, np.ndarray], weights: dict[str, float] | None = None
) -> np.ndarray:
    names = list(preds)
    if not names:
        return np.asarray([])
    if weights is None:
        w = {n: 1.0 / len(names) for n in names}
    else:
        s = sum(max(weights.get(n, 0.0), 0.0) for n in names) or 1.0
        w = {n: max(weights.get(n, 0.0), 0.0) / s for n in names}
    out = None
    for n in names:
        p = np.asarray(preds[n], dtype=np.float64)
        out = w[n] * p if out is None else out + w[n] * p
    return out if out is not None else np.asarray([])


def median_ensemble(preds: dict[str, np.ndarray]) -> np.ndarray:
    if not preds:
        return np.asarray([])
    stack = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in preds.values()], axis=0)
    return np.median(stack, axis=0)


def bayesian_model_averaging(
    preds: dict[str, np.ndarray],
    scores: dict[str, float],
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    # convert errors to weights via softmax(-score/T)
    names = list(preds)
    logits = np.asarray(
        [-float(scores.get(n, 1e6)) / max(temperature, 1e-6) for n in names], dtype=np.float64
    )
    logits -= logits.max()
    w = np.exp(logits)
    w = w / w.sum()
    return weighted_average(preds, {n: float(w[i]) for i, n in enumerate(names)})


def voting_ensemble(preds: dict[str, np.ndarray], *, threshold: float = 0.0) -> np.ndarray:
    """Majority vote on sign / binary direction."""
    if not preds:
        return np.asarray([])
    stack = np.stack(
        [
            (np.asarray(v, dtype=np.float64).reshape(-1) > threshold).astype(float)
            for v in preds.values()
        ],
        axis=0,
    )
    return (stack.mean(axis=0) >= 0.5).astype(np.float64)


def dynamic_ensemble_selection(
    preds: dict[str, np.ndarray],
    recent_errors: dict[str, float],
    *,
    top_k: int = 2,
) -> np.ndarray:
    ordered = sorted(recent_errors.items(), key=lambda kv: kv[1])
    keep = [n for n, _ in ordered[: max(int(top_k), 1)] if n in preds]
    if not keep:
        keep = list(preds)[:1]
    sub = {n: preds[n] for n in keep}
    inv = {n: 1.0 / max(recent_errors.get(n, 1.0), 1e-6) for n in keep}
    return weighted_average(sub, inv)


def build_ensemble(
    preds: dict[str, np.ndarray],
    *,
    config: EnsembleConfig,
    scores: dict[str, float] | None = None,
    meta_features: np.ndarray | None = None,
    gate_weights: np.ndarray | None = None,
) -> np.ndarray:
    method = config.method
    if method == "none" or not preds:
        return next(iter(preds.values())) if preds else np.asarray([])
    if method == "median":
        return median_ensemble(preds)
    if method == "bma":
        return bayesian_model_averaging(preds, scores or dict.fromkeys(preds, 1.0))
    if method == "voting":
        return voting_ensemble(preds)
    if method == "stacking":
        return stack_predictions(preds, meta_features=meta_features)
    if method == "blending":
        return blend_predictions(preds, scores=scores)
    if method == "moe":
        return moe_combine(preds, gate_weights=gate_weights)
    if method == "dynamic":
        return dynamic_ensemble_selection(
            preds, scores or dict.fromkeys(preds, 1.0), top_k=config.top_k
        )
    # weighted default — invert rmse-like scores
    if scores:
        inv = {n: 1.0 / max(float(scores.get(n, 1.0)), 1e-6) for n in preds}
        return weighted_average(preds, inv)
    return weighted_average(preds)

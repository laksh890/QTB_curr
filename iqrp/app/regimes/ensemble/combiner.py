"""Combine member regime probabilities into a unified posterior."""

from __future__ import annotations

from typing import Literal

import numpy as np

from iqrp.app.math.utils.numerical_stability import stable_softmax

CombineMethod = Literal[
    "majority",
    "weighted",
    "soft_voting",
    "bma",
    "stacking",
    "confidence",
    "dynamic",
    "meta",
]


def majority_vote(member_proba: list[np.ndarray], *, n_states: int) -> np.ndarray:
    """Hard vote then one-hot soft posterior."""
    t = member_proba[0].shape[0]
    votes = np.zeros((t, n_states), dtype=np.float64)
    for p in member_proba:
        hard = np.argmax(p, axis=1)
        for i, h in enumerate(hard):
            if 0 <= int(h) < n_states:
                votes[i, int(h)] += 1.0
    row = np.clip(votes.sum(axis=1, keepdims=True), 1e-300, None)
    return votes / row


def weighted_vote(
    member_proba: list[np.ndarray],
    weights: np.ndarray,
    *,
    n_states: int,
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / max(float(w.sum()), 1e-300)
    t = member_proba[0].shape[0]
    votes = np.zeros((t, n_states), dtype=np.float64)
    for wi, p in zip(w, member_proba, strict=False):
        hard = np.argmax(p, axis=1)
        for i, h in enumerate(hard):
            if 0 <= int(h) < n_states:
                votes[i, int(h)] += float(wi)
    row = np.clip(votes.sum(axis=1, keepdims=True), 1e-300, None)
    return votes / row


def soft_voting(member_proba: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = w / max(float(w.sum()), 1e-300)
    stack = np.stack(member_proba, axis=0)  # (M, T, K)
    out = np.tensordot(w, stack, axes=(0, 0))
    row = np.clip(out.sum(axis=1, keepdims=True), 1e-300, None)
    return out / row


def bayesian_model_averaging(
    member_proba: list[np.ndarray],
    log_evidence: np.ndarray,
) -> np.ndarray:
    """BMA with softmax model posterior from log-evidence."""
    w = stable_softmax(np.asarray(log_evidence, dtype=np.float64).reshape(-1))
    return soft_voting(member_proba, w)


def confidence_weighted(
    member_proba: list[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Weight each member by base weight × max-proba confidence at each step."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    t, k = member_proba[0].shape
    out = np.zeros((t, k), dtype=np.float64)
    for i in range(t):
        conf = np.array([float(p[i].max()) for p in member_proba], dtype=np.float64)
        cw = w * conf
        cw = cw / max(float(cw.sum()), 1e-300)
        out[i] = sum(cw[j] * member_proba[j][i] for j in range(len(member_proba)))
    row = np.clip(out.sum(axis=1, keepdims=True), 1e-300, None)
    return out / row


def stacking_combine(
    member_proba: list[np.ndarray],
    meta_weights: np.ndarray,
) -> np.ndarray:
    """Linear stacking: meta_weights shape ``(M,)`` or ``(M, K)``."""
    mw = np.asarray(meta_weights, dtype=np.float64)
    if mw.ndim == 1:
        return soft_voting(member_proba, mw)
    # per-class stacking
    t, k = member_proba[0].shape
    out = np.zeros((t, k), dtype=np.float64)
    for c in range(k):
        cols = np.column_stack([p[:, c] for p in member_proba])
        wc = mw[:, c] if mw.shape[1] == k else mw[:, 0]
        wc = wc / max(float(wc.sum()), 1e-300)
        out[:, c] = cols @ wc
    row = np.clip(out.sum(axis=1, keepdims=True), 1e-300, None)
    return out / row


def dynamic_combine(
    member_proba: list[np.ndarray],
    weights: np.ndarray,
    *,
    lookback: int = 20,
) -> np.ndarray:
    """Blend soft voting with recent confidence-weighted average."""
    soft = soft_voting(member_proba, weights)
    conf = confidence_weighted(member_proba, weights)
    # more confidence blend in recent window via exponential mix
    alpha = 0.5
    return (1.0 - alpha) * soft + alpha * conf


def meta_select(
    member_proba: list[np.ndarray],
    scores: np.ndarray,
) -> np.ndarray:
    """Pick single best model by score (meta-ensemble selection)."""
    best = int(np.argmax(np.asarray(scores, dtype=np.float64).reshape(-1)))
    best = int(np.clip(best, 0, len(member_proba) - 1))
    return member_proba[best].copy()


def combine(
    member_proba: list[np.ndarray],
    weights: np.ndarray,
    *,
    method: CombineMethod = "soft_voting",
    n_states: int | None = None,
    log_evidence: np.ndarray | None = None,
    meta_weights: np.ndarray | None = None,
    scores: np.ndarray | None = None,
) -> np.ndarray:
    if not member_proba:
        raise ValueError("No member probabilities to combine")
    k = int(n_states if n_states is not None else member_proba[0].shape[1])
    if method == "majority":
        return majority_vote(member_proba, n_states=k)
    if method == "weighted":
        return weighted_vote(member_proba, weights, n_states=k)
    if method == "bma":
        ev = log_evidence if log_evidence is not None else np.log(np.clip(weights, 1e-300, None))
        return bayesian_model_averaging(member_proba, ev)
    if method == "stacking":
        return stacking_combine(member_proba, meta_weights if meta_weights is not None else weights)
    if method == "confidence":
        return confidence_weighted(member_proba, weights)
    if method == "dynamic":
        return dynamic_combine(member_proba, weights)
    if method == "meta":
        sc = scores if scores is not None else weights
        return meta_select(member_proba, sc)
    return soft_voting(member_proba, weights)

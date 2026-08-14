"""Mixture-of-experts gating for dynamic ensembles."""

from __future__ import annotations

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=-1, keepdims=True), 1e-12)


def moe_combine(
    preds: dict[str, np.ndarray],
    *,
    gate_weights: np.ndarray | None = None,
    gate_logits: np.ndarray | None = None,
) -> np.ndarray:
    names = list(preds)
    if not names:
        return np.asarray([])
    X = np.column_stack([np.asarray(preds[n], dtype=np.float64).reshape(-1) for n in names])
    n, k = X.shape
    if gate_weights is not None:
        w = np.asarray(gate_weights, dtype=np.float64)
        if w.ndim == 1:
            w = np.clip(w, 0, None)
            w = w / (w.sum() or 1.0)
            return X @ w[:k]
        # (n, k)
        w = w.reshape(n, -1)[:, :k]
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        return (X * w).sum(axis=1)
    if gate_logits is not None:
        logits = np.asarray(gate_logits, dtype=np.float64)
        if logits.ndim == 1:
            w = softmax(logits)[:k]
            return X @ w
        w = softmax(logits.reshape(n, -1)[:, :k])
        return (X * w).sum(axis=1)
    return X.mean(axis=1)


def regime_gate_logits(regime_ids: np.ndarray, n_experts: int) -> np.ndarray:
    """One-hot-ish logits from discrete regimes."""
    r = np.asarray(regime_ids).reshape(-1).astype(int)
    n = r.size
    logits = np.zeros((n, max(n_experts, 1)), dtype=np.float64)
    for i, rid in enumerate(r):
        logits[i, int(rid) % n_experts] = 5.0
    return logits

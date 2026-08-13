"""Stacking meta-learner for forecast ensembles."""

from __future__ import annotations

from typing import Any

import numpy as np


def stack_predictions(
    preds: dict[str, np.ndarray],
    *,
    meta_features: np.ndarray | None = None,
    meta_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Linear stacking: learn or apply weights over base predictions.

    If ``meta_weights`` provided, use them; else equal weights.
    ``meta_features`` reserved for future conditioned stacking.
    """
    names = list(preds)
    if not names:
        return np.asarray([])
    X = np.column_stack([np.asarray(preds[n], dtype=np.float64).reshape(-1) for n in names])
    if meta_weights is not None:
        w = np.asarray(meta_weights, dtype=np.float64).reshape(-1)
        if w.size != X.shape[1]:
            w = np.ones(X.shape[1]) / X.shape[1]
        w = np.clip(w, 0, None)
        w = w / (w.sum() or 1.0)
        return X @ w
    if meta_features is not None:
        # ridge-like closed form if meta_features is actually y_true for training
        y = np.asarray(meta_features, dtype=np.float64).reshape(-1)
        n = min(y.size, X.shape[0])
        if n >= X.shape[1]:
            XtX = X[:n].T @ X[:n] + 1e-3 * np.eye(X.shape[1])
            Xty = X[:n].T @ y[:n]
            try:
                w = np.linalg.solve(XtX, Xty)
                w = np.clip(w, 0, None)
                w = w / (w.sum() or 1.0)
                return X @ w
            except Exception:  # noqa: BLE001  # pragma: no cover
                pass
    return X.mean(axis=1)


def fit_stacker(base_preds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Fit non-negative least-squares-ish stacking weights."""
    X = np.asarray(base_preds, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = min(X.shape[0], y.size)
    XtX = X[:n].T @ X[:n] + 1e-3 * np.eye(X.shape[1])
    Xty = X[:n].T @ y[:n]
    w = np.linalg.solve(XtX, Xty)
    w = np.clip(w, 0, None)
    return w / (w.sum() or 1.0)

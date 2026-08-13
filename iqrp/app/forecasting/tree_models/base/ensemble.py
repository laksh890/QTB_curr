"""Bagging, averaging, stacking and blending for tree models."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from iqrp.app.forecasting.tree_models.base.backends import (
    BackendName,
    create_estimator,
    estimator_predict,
)


def bagging_predict(
    backend: BackendName,
    X: np.ndarray,
    y: np.ndarray,
    X_new: np.ndarray,
    *,
    task: str = "regression",
    params: dict[str, Any] | None = None,
    n_bags: int = 5,
    random_state: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    preds = []
    n = X.shape[0]
    for i in range(max(n_bags, 1)):
        idx = rng.integers(0, n, size=n)
        est = create_estimator(backend, task=task, params=params or {})
        est.fit(X[idx], y[idx])
        preds.append(estimator_predict(est, X_new))
    return np.mean(np.stack(preds), axis=0)


def weighted_average(preds: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    arr = np.stack([np.asarray(p, dtype=np.float64).reshape(-1) for p in preds], axis=0)
    if weights is None:
        return arr.mean(axis=0)
    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() or 1.0)
    return (arr * w[:, None]).sum(axis=0)


def stacking_predict(
    base_preds_train: np.ndarray,
    y_train: np.ndarray,
    base_preds_test: np.ndarray,
) -> np.ndarray:
    """Linear stacking meta-learner."""
    A = np.asarray(base_preds_train, dtype=np.float64)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    A_ = np.column_stack([np.ones(A.shape[0]), A])
    coef, *_ = np.linalg.lstsq(A_, y, rcond=None)
    T = np.asarray(base_preds_test, dtype=np.float64)
    if T.ndim == 1:
        T = T.reshape(-1, 1)
    T_ = np.column_stack([np.ones(T.shape[0]), T])
    return T_ @ coef


def blending_predict(
    holdout_preds: np.ndarray,
    y_holdout: np.ndarray,
    test_preds: np.ndarray,
) -> np.ndarray:
    return stacking_predict(holdout_preds, y_holdout, test_preds)


def ensemble_fit_predict(
    backends: list[BackendName],
    X: np.ndarray,
    y: np.ndarray,
    X_new: np.ndarray,
    *,
    method: Literal["bagging", "average", "stacking", "blending"] = "average",
    task: str = "regression",
    params: dict[str, Any] | None = None,
    n_bags: int = 5,
) -> np.ndarray:
    if method == "bagging":
        return bagging_predict(backends[0], X, y, X_new, task=task, params=params, n_bags=n_bags)
    preds_train = []
    preds_test = []
    n = X.shape[0]
    split = int(n * 0.8)
    for b in backends:
        est = create_estimator(b, task=task, params=params or {})
        est.fit(X[:split], y[:split])
        preds_train.append(estimator_predict(est, X[split:]))
        # refit full for test
        est2 = create_estimator(b, task=task, params=params or {})
        est2.fit(X, y)
        preds_test.append(estimator_predict(est2, X_new))
    train_mat = np.column_stack(preds_train)
    test_mat = np.column_stack(preds_test)
    if method == "average":
        return weighted_average(preds_test)
    if method in {"stacking", "blending"}:
        return stacking_predict(train_mat, y[split:], test_mat)
    return weighted_average(preds_test)

"""Lightweight feature selection utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np

SelectionMethod = Literal["none", "variance", "correlation", "mutual_info"]


def select_by_variance(
    x: np.ndarray, *, threshold: float = 0.0, max_features: int | None = None
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    var = np.var(arr, axis=0)
    mask = var > threshold
    idx = np.where(mask)[0]
    if idx.size == 0:
        idx = np.array([int(np.argmax(var))], dtype=np.int64)
    if max_features is not None and idx.size > max_features:
        order = np.argsort(var[idx])[::-1][:max_features]
        idx = idx[order]
    return np.sort(idx)


def select_by_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float = 0.95,
    max_features: int | None = None,
) -> np.ndarray:
    """Drop highly collinear features; keep those most correlated with ``y``."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(arr.shape[0], target.size)
    arr, target = arr[:n], target[:n]
    f = arr.shape[1]
    # correlation with target
    scores = np.zeros(f, dtype=np.float64)
    for j in range(f):
        a = arr[:, j]
        if np.std(a) < 1e-12 or np.std(target) < 1e-12:
            scores[j] = 0.0
        else:
            scores[j] = abs(float(np.corrcoef(a, target)[0, 1]))
    keep = list(range(f))
    # greedy remove collinear
    removed: set[int] = set()
    for i in range(f):
        if i in removed:
            continue
        for j in range(i + 1, f):
            if j in removed:
                continue
            if np.std(arr[:, i]) < 1e-12 or np.std(arr[:, j]) < 1e-12:
                continue
            c = abs(float(np.corrcoef(arr[:, i], arr[:, j])[0, 1]))
            if c >= threshold:
                # drop lower target correlation
                drop = j if scores[i] >= scores[j] else i
                removed.add(drop)
    idx = np.asarray([i for i in keep if i not in removed], dtype=np.int64)
    if idx.size == 0:
        idx = np.array([int(np.argmax(scores))], dtype=np.int64)
    if max_features is not None and idx.size > max_features:
        order = np.argsort(scores[idx])[::-1][:max_features]
        idx = idx[order]
    return np.sort(idx)


def _mutual_info_score(x: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> float:
    """Histogram-based mutual information (continuous approximation)."""
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    b = np.asarray(y, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    if n < 2:
        return 0.0
    a, b = a[:n], b[:n]

    # digitize
    def _bins(v: np.ndarray) -> np.ndarray:
        qs = np.linspace(0, 100, n_bins + 1)
        edges = np.unique(np.percentile(v, qs))
        if edges.size < 2:
            return np.zeros(v.size, dtype=np.int64)
        return np.clip(np.digitize(v, edges[1:-1]), 0, edges.size - 2)

    ca, cb = _bins(a), _bins(b)
    k = int(max(ca.max(), cb.max()) + 1)
    joint = np.zeros((k, k), dtype=np.float64)
    for i in range(n):
        joint[ca[i], cb[i]] += 1.0
    joint /= joint.sum()
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(joint > 0, joint / (pa @ pb), 1.0)
        mi = np.nansum(joint * np.log(np.clip(ratio, 1e-300, None)))
    return float(max(mi, 0.0))


def select_by_mutual_info(
    x: np.ndarray, y: np.ndarray, *, max_features: int | None = None
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    scores = np.asarray(
        [_mutual_info_score(arr[:, j], y) for j in range(arr.shape[1])], dtype=np.float64
    )
    order = np.argsort(scores)[::-1]
    if max_features is not None:
        order = order[:max_features]
    return np.sort(order)


def select_features(
    x: np.ndarray,
    y: np.ndarray | None = None,
    *,
    method: SelectionMethod = "none",
    max_features: int | None = None,
    variance_threshold: float = 0.0,
    correlation_threshold: float = 0.95,
) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    f = arr.shape[1]
    if method == "none" or f == 0:
        idx = np.arange(f)
        if max_features is not None:
            idx = idx[:max_features]
        return idx
    if method == "variance":
        return select_by_variance(arr, threshold=variance_threshold, max_features=max_features)
    if y is None:
        return select_by_variance(arr, threshold=variance_threshold, max_features=max_features)
    if method == "correlation":
        return select_by_correlation(
            arr, y, threshold=correlation_threshold, max_features=max_features
        )
    return select_by_mutual_info(arr, y, max_features=max_features)

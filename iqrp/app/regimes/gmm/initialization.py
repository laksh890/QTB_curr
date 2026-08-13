"""Parameter initialization for Gaussian mixtures."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from iqrp.app.regimes.gmm.covariance import CovarianceType, estimate_covariances

InitMethod = Literal["random", "kmeans", "kmeans++", "hierarchical", "user"]


def initialize_parameters(
    x: np.ndarray,
    n_components: int,
    *,
    method: InitMethod = "kmeans",
    covariance_type: CovarianceType = "full",
    reg_covar: float = 1e-6,
    user_params: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(weights, means, covars)``."""
    rng = rng or np.random.default_rng()
    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n, _d = y.shape
    k = int(n_components)

    if method == "user" and user_params:
        w = np.asarray(user_params["weights"], dtype=np.float64)
        m = np.asarray(user_params["means"], dtype=np.float64)
        c = np.asarray(user_params["covars"], dtype=np.float64)
        w = w / max(float(w.sum()), 1e-300)
        return w, m, c

    if method == "kmeans++":
        means = _kmeans_plus_plus(y, k, rng)
        labels = _nearest_labels(y, means)
    elif method == "hierarchical":
        means, labels = _hierarchical_centers(y, k)
    elif method == "random":
        idx = rng.choice(n, size=k, replace=(n < k))
        means = y[idx].copy()
        labels = _nearest_labels(y, means)
    else:  # kmeans
        means, labels = _kmeans(y, k, rng=rng, n_iter=25)

    resp = np.zeros((n, k), dtype=np.float64)
    for i, lab in enumerate(labels):
        if 0 <= int(lab) < k:
            resp[i, int(lab)] = 1.0
    # soften empty
    if np.any(resp.sum(axis=0) < 1e-12):
        resp = np.full((n, k), 1.0 / k)
    nk = np.clip(resp.sum(axis=0), 1e-12, None)
    weights = nk / n
    means = (resp.T @ y) / nk[:, None]
    covars = estimate_covariances(
        y, resp, means, covariance_type=covariance_type, reg_covar=reg_covar
    )
    return weights, means, covars


def _kmeans(
    y: np.ndarray,
    k: int,
    *,
    rng: np.random.Generator,
    n_iter: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    n = y.shape[0]
    centers = (
        y[rng.choice(n, size=k, replace=False)].copy()
        if n >= k
        else y[rng.choice(n, size=k, replace=True)].copy()
    )
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        labels = _nearest_labels(y, centers)
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = y[mask].mean(axis=0)
    return centers, labels


def _kmeans_plus_plus(y: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = y.shape[0]
    centers = np.empty((k, y.shape[1]), dtype=np.float64)
    centers[0] = y[int(rng.integers(0, n))]
    for j in range(1, k):
        d2 = np.min(((y[:, None, :] - centers[None, :j, :]) ** 2).sum(axis=2), axis=1)
        d2 = np.clip(d2, 1e-300, None)
        probs = d2 / d2.sum()
        centers[j] = y[int(rng.choice(n, p=probs))]
    return centers


def _hierarchical_centers(y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Agglomerative clustering via repeated nearest-centroid merge (average linkage proxy)."""
    n = y.shape[0]
    if n <= k:
        means = y.copy()
        if n < k:
            pad = np.repeat(y[-1:], k - n, axis=0)
            means = np.vstack([means, pad])
        labels = np.arange(min(n, k), dtype=np.int64)
        if n < k:
            labels = np.concatenate([labels, np.full(k - n, k - 1, dtype=np.int64)])
        # map points
        labs = _nearest_labels(y, means[:k])
        return means[:k], labs

    # start with random subsample of centers then kmeans refine
    np.random.default_rng(0)
    # Ward-like: recursive bipartition by PCA axis
    clusters = [np.arange(n)]
    while len(clusters) < k:
        # split largest cluster
        sizes = [c.size for c in clusters]
        idx = int(np.argmax(sizes))
        members = clusters.pop(idx)
        if members.size < 2:
            clusters.append(members)
            break
        pts = y[members]
        direction = pts - pts.mean(axis=0)
        # first PC via power iteration
        v = direction.T @ direction @ np.ones(pts.shape[1])
        nrm = np.linalg.norm(v)
        v = v / nrm if nrm > 1e-12 else np.eye(1, pts.shape[1]).ravel()
        scores = pts @ v
        med = np.median(scores)
        left = members[scores <= med]
        right = members[scores > med]
        if left.size == 0 or right.size == 0:
            mid = members.size // 2
            left, right = members[:mid], members[mid:]
        clusters.append(left)
        clusters.append(right)
    means = np.vstack([y[c].mean(axis=0) for c in clusters[:k]])
    labels = _nearest_labels(y, means)
    return means, labels


def _nearest_labels(y: np.ndarray, centers: np.ndarray) -> np.ndarray:
    d = ((y[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d, axis=1).astype(np.int64)

"""Hierarchical Risk Parity (HRP) and Hierarchical Equal Risk Contribution (HERC)."""

from __future__ import annotations

from typing import Any

import numpy as np


def _corr_from_cov(cov: np.ndarray) -> np.ndarray:
    c = 0.5 * (cov + cov.T)
    vol = np.sqrt(np.maximum(np.diag(c), 1e-18))
    denom = np.outer(vol, vol)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, c / denom, 0.0)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def _distance_from_corr(corr: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))


def _scipy_linkage(dist: np.ndarray, method: str = "single") -> np.ndarray | None:
    try:
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform

        # Condensed distance
        condensed = squareform(dist, checks=False)
        return np.asarray(linkage(condensed, method=method), dtype=np.float64)
    except Exception:  # noqa: BLE001
        return None


def _numpy_agglomerative(dist: np.ndarray, method: str = "single") -> np.ndarray:
    """Pure-numpy agglomerative clustering returning a scipy-like linkage matrix."""
    n = dist.shape[0]
    # Active clusters: map cluster id -> member indices
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    active = set(range(n))
    next_id = n
    # Pairwise cluster distances (initialize with leaf distances)
    cd = dist.copy()
    # Expandable matrix indexed by cluster id — use dict of pairs
    pair_dist: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            pair_dist[(i, j)] = float(dist[i, j])

    Z = np.zeros((n - 1, 4), dtype=np.float64)
    for step in range(n - 1):
        # Find closest pair
        best = None
        best_d = np.inf
        alist = sorted(active)
        for ii, a in enumerate(alist):
            for b in alist[ii + 1 :]:
                key = (a, b) if a < b else (b, a)
                d = pair_dist.get(key, np.inf)
                if d < best_d:
                    best_d = d
                    best = (a, b)
        assert best is not None
        a, b = best
        size_a = len(clusters[a])
        size_b = len(clusters[b])
        Z[step, 0] = float(a)
        Z[step, 1] = float(b)
        Z[step, 2] = float(best_d)
        Z[step, 3] = float(size_a + size_b)
        # Merge
        new_members = clusters[a] + clusters[b]
        clusters[next_id] = new_members
        active.remove(a)
        active.remove(b)
        # Distances from new cluster to remaining
        for c in list(active):
            key_ac = (a, c) if a < c else (c, a)
            key_bc = (b, c) if b < c else (c, b)
            da = pair_dist.get(key_ac, np.inf)
            db = pair_dist.get(key_bc, np.inf)
            m = str(method).lower()
            if m == "complete":
                dnc = max(da, db)
            elif m in ("average", "upaverage", "weighted"):
                dnc = (size_a * da + size_b * db) / (size_a + size_b)
            else:  # single
                dnc = min(da, db)
            key_nc = (next_id, c) if next_id < c else (c, next_id)
            pair_dist[key_nc] = float(dnc)
        active.add(next_id)
        next_id += 1
    return Z


def _quasi_diag_order(linkage_matrix: np.ndarray, n: int) -> list[int]:
    """Seriation / quasi-diagonalization order from linkage."""
    # Build tree recursively from root
    root = 2 * n - 2  # last cluster id for n leaves
    children: dict[int, tuple[int, int]] = {}
    for i in range(linkage_matrix.shape[0]):
        cid = n + i
        children[cid] = (int(linkage_matrix[i, 0]), int(linkage_matrix[i, 1]))

    order: list[int] = []

    def _walk(node: int) -> None:
        if node < n:
            order.append(node)
            return
        left, right = children[node]
        _walk(left)
        _walk(right)

    _walk(root)
    return order


def _cluster_var(cov: np.ndarray, items: list[int]) -> float:
    if not items:
        return 0.0
    sub = cov[np.ix_(items, items)]
    # Inverse-vol cluster weights
    diag = np.maximum(np.diag(sub), 1e-18)
    w = 1.0 / np.sqrt(diag)
    w = w / float(np.sum(w))
    return float(w @ sub @ w)


def _recursive_bisection(
    cov: np.ndarray,
    ordered: list[int],
    *,
    equal_risk: bool = False,
) -> np.ndarray:
    """HRP (variance split) or HERC (equal risk split) recursive bisection."""
    n = cov.shape[0]
    w = np.ones(n, dtype=np.float64)

    def _bisect(items: list[int]) -> None:
        if len(items) <= 1:
            return
        split = len(items) // 2
        left = items[:split]
        right = items[split:]
        var_l = _cluster_var(cov, left)
        var_r = _cluster_var(cov, right)
        if equal_risk:
            # HERC: allocate equal risk to each child cluster
            alpha = 0.5
        else:
            # HRP: inverse-variance allocation between clusters
            denom = var_l + var_r
            if denom <= 1e-18:
                alpha = 0.5
            else:
                alpha = 1.0 - var_l / denom  # weight to left
        for i in left:
            w[i] *= alpha
        for i in right:
            w[i] *= 1.0 - alpha
        _bisect(left)
        _bisect(right)

    _bisect(list(ordered))
    s = float(np.sum(w))
    if s > 0:
        w = w / s
    return w


def hrp_weights(
    cov: Any,
    *,
    names: list[str] | None = None,
    corr: Any | None = None,
    linkage: str = "single",
) -> dict[str, Any]:
    """Classic Hierarchical Risk Parity (Lopez de Prado)."""
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be square")
    n = c.shape[0]
    keys = names if names and len(names) == n else [f"s{i}" for i in range(n)]
    if n == 0:
        return {"name": "hrp", "weights": {}, "weight_vector": [], "order": []}
    if n == 1:
        return {
            "name": "hrp",
            "weights": {keys[0]: 1.0},
            "weight_vector": [1.0],
            "order": keys,
            "linkage_method": linkage,
        }

    if corr is not None:
        corr_m = np.asarray(corr, dtype=np.float64)
        if corr_m.shape != (n, n):
            corr_m = _corr_from_cov(c)
    else:
        corr_m = _corr_from_cov(c)

    dist = _distance_from_corr(corr_m)
    np.fill_diagonal(dist, 0.0)
    Z = _scipy_linkage(dist, method=linkage)
    used = "scipy" if Z is not None else "numpy"
    if Z is None:
        Z = _numpy_agglomerative(dist, method=linkage)
    order_idx = _quasi_diag_order(Z, n)
    w = _recursive_bisection(c, order_idx, equal_risk=False)
    return {
        "name": "hrp",
        "weights": {keys[i]: float(w[i]) for i in range(n)},
        "weight_vector": w.tolist(),
        "order": [keys[i] for i in order_idx],
        "order_indices": order_idx,
        "linkage_method": linkage,
        "linkage_backend": used,
        "distance_matrix": dist.tolist(),
    }


def herc_weights(
    cov: Any,
    *,
    names: list[str] | None = None,
    corr: Any | None = None,
    linkage: str = "single",
) -> dict[str, Any]:
    """Hierarchical Equal Risk Contribution."""
    c = np.asarray(cov, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("cov must be square")
    n = c.shape[0]
    keys = names if names and len(names) == n else [f"s{i}" for i in range(n)]
    if n == 0:
        return {"name": "herc", "weights": {}, "weight_vector": [], "order": []}
    if n == 1:
        return {
            "name": "herc",
            "weights": {keys[0]: 1.0},
            "weight_vector": [1.0],
            "order": keys,
            "linkage_method": linkage,
        }

    if corr is not None:
        corr_m = np.asarray(corr, dtype=np.float64)
        if corr_m.shape != (n, n):
            corr_m = _corr_from_cov(c)
    else:
        corr_m = _corr_from_cov(c)

    dist = _distance_from_corr(corr_m)
    np.fill_diagonal(dist, 0.0)
    Z = _scipy_linkage(dist, method=linkage)
    used = "scipy" if Z is not None else "numpy"
    if Z is None:
        Z = _numpy_agglomerative(dist, method=linkage)
    order_idx = _quasi_diag_order(Z, n)
    w = _recursive_bisection(c, order_idx, equal_risk=True)
    return {
        "name": "herc",
        "weights": {keys[i]: float(w[i]) for i in range(n)},
        "weight_vector": w.tolist(),
        "order": [keys[i] for i in order_idx],
        "order_indices": order_idx,
        "linkage_method": linkage,
        "linkage_backend": used,
        "distance_matrix": dist.tolist(),
    }

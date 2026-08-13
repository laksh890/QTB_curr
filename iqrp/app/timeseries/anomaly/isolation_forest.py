"""Isolation Forest anomaly detection with sklearn or pure-numpy fallback."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def isolation_forest_anomalies(
    x: np.ndarray | list[float],
    *,
    contamination: float = 0.05,
    n_trees: int = 100,
    max_depth: int | None = None,
    window: int = 1,
    seed: int = 42,
) -> AnalysisResult:
    """Isolation Forest on univariate (or lagged) features (FULL_SAMPLE).

    Tries ``sklearn.ensemble.IsolationForest`` when available; otherwise uses a
    pure-NumPy random isolation tree ensemble.
    """
    y = as_float_array(x)
    n = y.size
    cont = float(np.clip(contamination, 0.001, 0.5))
    w = max(int(window), 1)
    if n < max(10, w + 2):
        return AnalysisResult(
            method="isolation_forest_anomalies",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="no isolation-based anomalies",
            alternative_hypothesis="anomalous points isolated by short paths",
            parameters={"contamination": cont, "n_trees": n_trees, "window": w, "seed": seed},
        )
    # feature matrix: rolling windows as rows (causal features of length w)
    if w == 1:
        X = y.reshape(-1, 1)
        row_index = np.arange(n)
    else:
        n_rows = n - w + 1
        X = np.lib.stride_tricks.as_strided(
            y,
            shape=(n_rows, w),
            strides=(y.strides[0], y.strides[0]),
            writeable=False,
        ).copy()
        row_index = np.arange(w - 1, n)
    finite_rows = np.isfinite(X).all(axis=1)
    if finite_rows.sum() < 10:
        return AnalysisResult(
            method="isolation_forest_anomalies",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="no isolation-based anomalies",
            alternative_hypothesis="anomalous points isolated by short paths",
            parameters={"contamination": cont, "n_trees": n_trees, "window": w, "seed": seed},
        )
    Xf = X[finite_rows]
    idx_map = row_index[finite_rows]

    backend = "numpy"
    try:
        from sklearn.ensemble import IsolationForest  # type: ignore

        clf = IsolationForest(
            n_estimators=int(n_trees),
            contamination=cont,
            max_samples="auto",
            random_state=int(seed),
        )
        pred = clf.fit_predict(Xf)
        scores_raw = -clf.score_samples(Xf)  # higher = more anomalous
        is_anom = pred == -1
        backend = "sklearn"
    except Exception:
        scores_raw, is_anom = _numpy_isolation_forest(
            Xf,
            n_trees=int(n_trees),
            contamination=cont,
            max_depth=max_depth,
            seed=int(seed),
        )

    scores = np.full(n, np.nan, dtype=np.float64)
    mask = np.zeros(n, dtype=bool)
    scores[idx_map] = scores_raw
    mask[idx_map] = is_anom
    indices = np.flatnonzero(mask).tolist()
    return AnalysisResult(
        method="isolation_forest_anomalies",
        value={"indices": indices, "scores": scores, "is_anomaly": mask},
        statistic=float(np.nanmax(scores)) if np.isfinite(scores).any() else np.nan,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="no isolation-based anomalies",
        alternative_hypothesis="anomalous points isolated by short paths",
        significant=len(indices) > 0,
        parameters={
            "contamination": cont,
            "n_trees": int(n_trees),
            "window": w,
            "seed": int(seed),
            "max_depth": max_depth,
        },
        metadata={"n": n, "n_anomalies": len(indices), "backend": backend},
    )


def _numpy_isolation_forest(
    X: np.ndarray,
    *,
    n_trees: int,
    contamination: float,
    max_depth: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    depth_limit = int(max_depth) if max_depth is not None else int(np.ceil(np.log2(max(n, 2))))
    path_lengths = np.zeros(n, dtype=np.float64)
    for _ in range(max(n_trees, 1)):
        path_lengths += _isolation_tree_depths(X, rng, depth_limit)
    path_lengths /= max(n_trees, 1)
    # anomaly score: shorter paths → higher score
    c_n = _avg_path_length(n)
    scores = np.power(2.0, -path_lengths / max(c_n, 1e-12))
    k = max(1, int(np.ceil(contamination * n)))
    cutoff = np.partition(scores, -k)[-k]
    is_anom = scores >= cutoff
    return scores, is_anom


def _avg_path_length(n: int) -> float:
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return 2.0 * (np.log(n - 1.0) + 0.5772156649) - 2.0 * (n - 1.0) / n


def _isolation_tree_depths(X: np.ndarray, rng: np.random.Generator, max_depth: int) -> np.ndarray:
    n = X.shape[0]
    depths = np.zeros(n, dtype=np.float64)
    # iterative stack: (indices, depth)
    stack: list[tuple[np.ndarray, int]] = [(np.arange(n), 0)]
    while stack:
        idx, depth = stack.pop()
        if idx.size <= 1 or depth >= max_depth:
            depths[idx] += depth + _avg_path_length(int(idx.size))
            continue
        feat = int(rng.integers(0, X.shape[1]))
        col = X[idx, feat]
        lo, hi = float(np.min(col)), float(np.max(col))
        if hi - lo < 1e-15:
            depths[idx] += depth + _avg_path_length(int(idx.size))
            continue
        split = float(rng.uniform(lo, hi))
        left = idx[col < split]
        right = idx[col >= split]
        if left.size == 0 or right.size == 0:
            depths[idx] += depth + _avg_path_length(int(idx.size))
            continue
        stack.append((left, depth + 1))
        stack.append((right, depth + 1))
    return depths

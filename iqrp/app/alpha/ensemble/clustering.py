"""Correlation-based clustering of alpha signals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from iqrp.app.alpha.ensemble.correlation import signal_correlation_matrix


def _corr_matrix_from_input(
    corr: Mapping[str, Any] | np.ndarray,
    labels: Sequence[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    if isinstance(corr, Mapping) and "matrix" in corr:
        mat = np.asarray(corr["matrix"], dtype=np.float64)
        labs = list(corr.get("labels", labels or []))
        if len(labs) != mat.shape[0]:
            labs = [f"s{i}" for i in range(mat.shape[0])]
        return labs, mat
    mat = np.asarray(corr, dtype=np.float64)
    labs = list(labels) if labels is not None else [f"s{i}" for i in range(mat.shape[0])]
    return labs, mat


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """Convert correlation to distance: ``sqrt(0.5 * (1 - rho))``."""
    c = np.asarray(corr, dtype=np.float64)
    c = np.clip(np.nan_to_num(c, nan=0.0), -1.0, 1.0)
    d = np.sqrt(np.maximum(0.5 * (1.0 - c), 0.0))
    np.fill_diagonal(d, 0.0)
    return d


def hierarchical_correlation_clusters(
    corr: Mapping[str, Any] | np.ndarray,
    *,
    labels: Sequence[str] | None = None,
    threshold: float = 0.5,
    max_clusters: int | None = None,
) -> dict[str, Any]:
    """Greedy agglomerative clustering on correlation distance.

    Merges nearest pairs while distance < ``threshold`` (corr distance).
    """
    labs, mat = _corr_matrix_from_input(corr, labels)
    n = len(labs)
    if n == 0:
        return {
            "name": "hierarchical_correlation_clusters",
            "clusters": [],
            "labels": [],
            "assignment": {},
        }

    dist = correlation_distance(mat)
    # start with each signal its own cluster
    clusters: list[list[int]] = [[i] for i in range(n)]

    def cluster_distance(a: list[int], b: list[int]) -> float:
        vals = [dist[i, j] for i in a for j in b]
        return float(np.mean(vals)) if vals else float("inf")

    while len(clusters) > 1:
        if max_clusters is not None and len(clusters) <= max_clusters:
            break
        best = None
        best_d = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = cluster_distance(clusters[i], clusters[j])
                if d < best_d:
                    best_d = d
                    best = (i, j)
        if best is None or best_d >= float(threshold):
            break
        i, j = best
        merged = clusters[i] + clusters[j]
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]

    named = [[labs[i] for i in c] for c in clusters]
    assignment = {labs[i]: ci for ci, c in enumerate(clusters) for i in c}
    return {
        "name": "hierarchical_correlation_clusters",
        "threshold": float(threshold),
        "n_clusters": len(named),
        "clusters": named,
        "assignment": assignment,
        "labels": labs,
    }


def cluster_signals_from_series(
    series: Mapping[str, Any],
    *,
    kind: str = "prediction",
    threshold: float = 0.5,
) -> dict[str, Any]:
    corr = signal_correlation_matrix(series, kind=kind)  # type: ignore[arg-type]
    return hierarchical_correlation_clusters(corr, threshold=threshold)


def representative_per_cluster(
    clusters: Mapping[str, Any] | Sequence[Sequence[str]],
    metrics_by_signal: Mapping[str, Mapping[str, Any]],
    *,
    key: str = "ic",
) -> list[str]:
    """Pick the highest-|metric| member from each cluster."""
    if isinstance(clusters, Mapping):
        groups = clusters.get("clusters", [])
    else:
        groups = list(clusters)
    reps: list[str] = []
    for group in groups:
        if not group:
            continue
        best = max(
            group,
            key=lambda n: abs(float(metrics_by_signal.get(n, {}).get(key, 0.0) or 0.0)),
        )
        reps.append(str(best))
    return reps

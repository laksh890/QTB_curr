"""Eigenvalue / eigenvector routines."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math._array import as_matrix


def eig(a: Any) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eig(as_matrix(a))
    return np.asarray(values), np.asarray(vectors)


def eigh(a: Any) -> tuple[np.ndarray, np.ndarray]:
    """Hermitian / symmetric eigen-decomposition (sorted ascending)."""
    values, vectors = np.linalg.eigh(as_matrix(a))
    return np.asarray(values), np.asarray(vectors)


def spectral_radius(a: Any) -> float:
    values, _ = eig(a)
    return float(np.max(np.abs(values)))


def condition_number(a: Any) -> float:
    return float(np.linalg.cond(as_matrix(a)))


def principal_components(a: Any, *, n_components: int = 2) -> dict[str, np.ndarray]:
    x = as_matrix(a)
    x = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(x, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    idx = np.argsort(values)[::-1]
    values = values[idx]
    vectors = vectors[:, idx]
    k = min(n_components, vectors.shape[1])
    comps = vectors[:, :k]
    scores = x @ comps
    return {
        "eigenvalues": values[:k],
        "components": comps,
        "scores": scores,
        "explained_variance_ratio": values[:k] / max(float(values.sum()), 1e-15),
    }

"""Sparse matrix helpers (SciPy sparse)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse  # type: ignore[import-untyped]

from iqrp.app.math._array import as_matrix


def to_csr(a: Any) -> Any:
    if sparse.issparse(a):
        return a.tocsr()
    return sparse.csr_matrix(as_matrix(a))


def to_csc(a: Any) -> Any:
    if sparse.issparse(a):
        return a.tocsc()
    return sparse.csc_matrix(as_matrix(a))


def sparse_multiply(a: Any, b: Any) -> Any:
    return to_csr(a) @ to_csr(b)


def sparse_add(a: Any, b: Any) -> Any:
    return to_csr(a) + to_csr(b)


def sparsity(a: Any) -> float:
    m = to_csr(a)
    total = m.shape[0] * m.shape[1]
    if total == 0:
        return 0.0
    return float(1.0 - m.nnz / total)


def dense(a: Any) -> np.ndarray:
    if sparse.issparse(a):
        return np.asarray(a.toarray(), dtype=np.float64)
    return as_matrix(a)


def identity(n: int, *, format: str = "csr") -> Any:
    return sparse.eye(n, format=format)

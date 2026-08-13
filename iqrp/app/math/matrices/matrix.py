"""Core dense matrix operations."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math._array import as_matrix


def multiply(a: Any, b: Any) -> np.ndarray:
    return np.asarray(as_matrix(a) @ as_matrix(b), dtype=np.float64)


def inverse(a: Any) -> np.ndarray:
    return np.linalg.inv(as_matrix(a))


def pseudo_inverse(a: Any, *, rcond: float | None = None) -> np.ndarray:
    return np.linalg.pinv(as_matrix(a), rcond=rcond)


def transpose(a: Any) -> np.ndarray:
    return as_matrix(a).T


def det(a: Any) -> float:
    return float(np.linalg.det(as_matrix(a)))


def trace(a: Any) -> float:
    return float(np.trace(as_matrix(a)))


def kronecker(a: Any, b: Any) -> np.ndarray:
    return np.kron(as_matrix(a), as_matrix(b))


def hadamard(a: Any, b: Any) -> np.ndarray:
    return np.asarray(as_matrix(a) * as_matrix(b), dtype=np.float64)


def is_symmetric(a: Any, *, tol: float = 1e-10) -> bool:
    m = as_matrix(a)
    return bool(np.allclose(m, m.T, atol=tol))


def is_positive_definite(a: Any) -> bool:
    m = as_matrix(a)
    try:
        np.linalg.cholesky(m)
        return True
    except np.linalg.LinAlgError:
        return False


def normalize_rows(a: Any) -> np.ndarray:
    m = as_matrix(a).astype(np.float64)
    s = m.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return m / s


def frobenius_norm(a: Any) -> float:
    return float(np.linalg.norm(as_matrix(a), ord="fro"))

"""Matrix factorizations."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg  # type: ignore[import-untyped]

from iqrp.app.math._array import as_matrix


def lu(a: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p, lower, upper = linalg.lu(as_matrix(a))
    return np.asarray(p), np.asarray(lower), np.asarray(upper)


def qr(a: Any, *, mode: str = "reduced") -> tuple[np.ndarray, np.ndarray]:
    result = np.linalg.qr(as_matrix(a), mode=mode)  # type: ignore[call-overload]
    if isinstance(result, tuple):
        q, r = result
    else:
        q, r = result.Q, result.R
    return np.asarray(q), np.asarray(r)


def cholesky(a: Any) -> np.ndarray:
    return np.linalg.cholesky(as_matrix(a))


def svd(a: Any, *, full_matrices: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, s, vt = np.linalg.svd(as_matrix(a), full_matrices=full_matrices)
    return np.asarray(u), np.asarray(s), np.asarray(vt)

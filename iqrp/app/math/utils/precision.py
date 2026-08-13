"""Floating-point precision utilities."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from iqrp.app.math._array import as_array

Precision = Literal["float32", "float64", "float128"]


_DTYPE_MAP: dict[str, type[np.floating[Any]]] = {
    "float32": np.float32,
    "float64": np.float64,
}


def resolve_dtype(precision: Precision = "float64") -> type[np.floating[Any]]:
    if precision == "float128":
        # Platform-dependent; fall back to float64 if unavailable
        dt: type[np.floating[Any]] = getattr(np, "float128", np.float64)
        return dt
    return _DTYPE_MAP[precision]


def cast(a: Any, precision: Precision = "float64") -> np.ndarray:
    return as_array(a, dtype=resolve_dtype(precision))


def machine_eps(precision: Precision = "float64") -> float:
    return float(np.finfo(resolve_dtype(precision)).eps)


def relative_error(estimate: Any, truth: Any, *, eps: float = 1e-15) -> np.ndarray:
    e = as_array(estimate)
    t = as_array(truth)
    return np.asarray(np.abs(e - t) / np.maximum(np.abs(t), eps), dtype=np.float64)


def is_close(
    a: Any,
    b: Any,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> bool:
    return bool(np.allclose(as_array(a), as_array(b), rtol=rtol, atol=atol))


def nextafter(x: float, toward: float) -> float:
    return float(np.nextafter(x, toward))

"""Array conversion helpers for NumPy / Polars interoperability."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl


def as_array(data: Any, *, dtype: type[np.floating[Any]] | None = np.float64) -> np.ndarray:
    """Convert array-like / Polars Series / DataFrame column block to ndarray."""
    if isinstance(data, np.ndarray):
        return data.astype(dtype, copy=False) if dtype is not None else data
    if isinstance(data, pl.Series):
        return data.to_numpy().astype(dtype, copy=False) if dtype is not None else data.to_numpy()
    if isinstance(data, pl.DataFrame):
        return data.to_numpy().astype(dtype, copy=False) if dtype is not None else data.to_numpy()
    return np.asarray(data, dtype=dtype)


def as_vector(data: Any) -> np.ndarray:
    arr = as_array(data).ravel()
    return arr


def as_matrix(data: Any) -> np.ndarray:
    arr = as_array(data)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1D/2D array, got ndim={arr.ndim}")
    return arr

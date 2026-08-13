"""Conditional probability utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math._array import as_array, as_matrix
from iqrp.app.math.utils.numerical_stability import safe_divide


def conditional_probability(joint: Any, marginal: Any) -> np.ndarray:
    """P(A|B) = P(A,B) / P(B) with safe division."""
    return safe_divide(as_array(joint), as_array(marginal), fill=0.0)


def conditional_from_table(joint_table: Any, *, axis: int = 1) -> np.ndarray:
    """Row/column normalize a joint probability table to conditionals."""
    table = as_matrix(joint_table).astype(np.float64)
    if axis == 1:
        denom = table.sum(axis=1, keepdims=True)
        return safe_divide(table, denom, fill=0.0)
    denom = table.sum(axis=0, keepdims=True)
    return safe_divide(table, denom, fill=0.0)


def law_of_total_probability(conditionals: Any, prior: Any) -> np.ndarray:
    """P(A) = sum_i P(A|B_i) P(B_i)."""
    c = as_array(conditionals)
    p = as_array(prior).ravel()
    if c.ndim == 1:
        return np.asarray(np.dot(c, p), dtype=np.float64)
    return np.asarray(c @ p, dtype=np.float64)


def chain_rule(conditionals: list[Any]) -> np.ndarray:
    """Product of successive conditional factors (elementwise)."""
    out = np.ones_like(as_array(conditionals[0]), dtype=np.float64)
    for c in conditionals:
        out = out * as_array(c)
    return out

"""Information-theory convenience API (re-exports + extras)."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math._array import as_vector
from iqrp.app.math.statistics.entropy import (
    conditional_entropy,
    cross_entropy,
    entropy,
    information_gain,
    js_divergence,
    kl_divergence,
    mutual_information,
)


def empirical_entropy(samples: Any, *, bins: int = 20) -> float:
    x = as_vector(samples)
    hist, _ = np.histogram(x, bins=bins, density=False)
    return entropy(hist)


def empirical_mutual_information(x: Any, y: Any, *, bins: int = 10) -> float:
    a = as_vector(x)
    b = as_vector(y)
    joint, _, _ = np.histogram2d(a, b, bins=bins)
    return mutual_information(joint)


def normalized_mutual_information(joint: Any) -> float:
    from iqrp.app.math._array import as_array

    table = as_array(joint).astype(np.float64)
    table = table / table.sum()
    hx = entropy(table.sum(axis=1))
    hy = entropy(table.sum(axis=0))
    mi = mutual_information(table)
    denom = 0.5 * (hx + hy)
    return float(mi / denom) if denom > 0 else 0.0


__all__ = [
    "conditional_entropy",
    "cross_entropy",
    "empirical_entropy",
    "empirical_mutual_information",
    "entropy",
    "information_gain",
    "js_divergence",
    "kl_divergence",
    "mutual_information",
    "normalized_mutual_information",
]

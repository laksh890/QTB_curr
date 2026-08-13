"""Permutation entropy (Bandt & Pompe)."""

from __future__ import annotations

import itertools
import math

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def permutation_entropy(
    x: np.ndarray | list[float],
    *,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> AnalysisResult:
    """Bandt–Pompe permutation entropy.

    Statistical descriptor only — not a guaranteed predictive signal.
    """
    y = as_float_array(x)
    finite = y[np.isfinite(y)]
    n = finite.size
    m = max(int(order), 2)
    tau = max(int(delay), 1)
    n_patterns = n - (m - 1) * tau
    if n_patterns < m * 2:
        return AnalysisResult(
            method="permutation_entropy",
            value="insufficient_data",
            statistic=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="uniform ordinal pattern distribution (max PE)",
            alternative_hypothesis="structured ordinal dynamics (lower PE)",
            parameters={"order": m, "delay": tau, "normalize": normalize},
        )
    embed = np.column_stack([finite[i * tau : i * tau + n_patterns] for i in range(m)])
    patterns = np.argsort(embed, axis=1, kind="mergesort")
    factorial = int(math.factorial(m))
    perm_index = {p: i for i, p in enumerate(itertools.permutations(range(m)))}
    counts = np.zeros(factorial, dtype=np.float64)
    for row in patterns:
        counts[perm_index[tuple(int(v) for v in row)]] += 1.0
    p = counts / counts.sum()
    p = p[p > 0]
    H = float(-np.sum(p * np.log(p)))
    H_max = float(np.log(factorial))
    value = H / H_max if normalize and H_max > 0 else H
    return AnalysisResult(
        method="permutation_entropy",
        value=float(value),
        statistic=float(value),
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="uniform ordinal pattern distribution (max PE)",
        alternative_hypothesis="structured ordinal dynamics (lower PE)",
        significant=bool(normalize and value < 0.9),
        parameters={"order": m, "delay": tau, "normalize": normalize},
        metadata={"n": n, "n_patterns": n_patterns, "raw_entropy": H, "n_perms": factorial},
    )

"""STAMP-lite brute-force matrix profile."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def compute_matrix_profile(
    x: np.ndarray | list[float],
    *,
    window: int = 32,
    exclude: int | None = None,
) -> AnalysisResult:
    """Brute-force z-normalized Euclidean matrix profile (FULL_SAMPLE).

    Correctness-first STAMP-lite: O((n-m+1)^2 · m) for research/validation.
    """
    y = as_float_array(x)
    finite = np.isfinite(y)
    if not finite.all():
        y = y.copy()
        y[~finite] = float(np.nanmean(y[finite])) if finite.any() else 0.0
    n = y.size
    m = max(int(window), 2)
    excl = int(exclude) if exclude is not None else m // 2
    n_sub = n - m + 1
    if n_sub < 3:
        return AnalysisResult(
            method="compute_matrix_profile",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis=None,
            alternative_hypothesis=None,
            parameters={"window": m, "exclude": excl},
        )
    # precompute z-normalized subsequences
    subs = np.lib.stride_tricks.as_strided(
        y, shape=(n_sub, m), strides=(y.strides[0], y.strides[0]), writeable=False
    ).copy()
    mu = subs.mean(axis=1, keepdims=True)
    sd = subs.std(axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    zn = (subs - mu) / sd

    mp = np.full(n_sub, np.inf, dtype=np.float64)
    mpi = np.full(n_sub, -1, dtype=np.int64)
    for i in range(n_sub):
        # distance to all j outside exclusion zone
        diffs = zn - zn[i]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        lo = max(0, i - excl)
        hi = min(n_sub, i + excl + 1)
        dists[lo:hi] = np.inf
        j = int(np.argmin(dists))
        if np.isfinite(dists[j]):
            mp[i] = dists[j]
            mpi[i] = j

    return AnalysisResult(
        method="compute_matrix_profile",
        value={"matrix_profile": mp, "profile_index": mpi},
        statistic=float(np.nanmax(mp[np.isfinite(mp)])) if np.isfinite(mp).any() else np.nan,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        parameters={"window": m, "exclude": excl},
        metadata={"n": n, "n_subsequences": n_sub},
    )

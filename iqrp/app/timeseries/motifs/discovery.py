"""Motif discovery from matrix profile."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array
from iqrp.app.timeseries.motifs.matrix_profile import compute_matrix_profile


def find_motifs(
    x: np.ndarray | list[float],
    *,
    window: int = 32,
    top_k: int = 3,
    max_distance: float | None = None,
) -> AnalysisResult:
    """Find top-k motif pairs (nearest-neighbor pairs with smallest MP) (FULL_SAMPLE)."""
    y = as_float_array(x)
    w = max(int(window), 2)
    k = max(int(top_k), 1)
    mp_res = compute_matrix_profile(y, window=w)
    if isinstance(mp_res.value, str):
        return AnalysisResult(
            method="find_motifs",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="no repeated subsequences (motifs)",
            alternative_hypothesis="one or more motif pairs exist",
            parameters={"window": w, "top_k": k, "max_distance": max_distance},
        )
    mp = np.asarray(mp_res.value["matrix_profile"], dtype=np.float64)
    mpi = np.asarray(mp_res.value["profile_index"], dtype=np.int64)
    order = np.argsort(np.where(np.isfinite(mp), mp, np.inf))
    motifs: list[dict] = []
    used: set[int] = set()
    for i in order:
        i = int(i)
        j = int(mpi[i])
        if j < 0 or not np.isfinite(mp[i]):
            continue
        if max_distance is not None and mp[i] > max_distance:
            break
        if i in used or j in used:
            continue
        if any(abs(i - u) < w or abs(j - u) < w for u in used):
            continue
        motifs.append(
            {
                "index_a": i,
                "index_b": j,
                "distance": float(mp[i]),
                "subsequence_a": y[i : i + w].tolist(),
                "subsequence_b": y[j : j + w].tolist(),
            }
        )
        used.add(i)
        used.add(j)
        if len(motifs) >= k:
            break
    return AnalysisResult(
        method="find_motifs",
        value=motifs,
        statistic=motifs[0]["distance"] if motifs else np.nan,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="no repeated subsequences (motifs)",
        alternative_hypothesis="one or more motif pairs exist",
        significant=len(motifs) > 0,
        parameters={"window": w, "top_k": k, "max_distance": max_distance},
        metadata={"n_motifs": len(motifs), "n": y.size},
    )

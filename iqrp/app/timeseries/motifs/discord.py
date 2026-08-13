"""Discord (anomaly subsequence) discovery."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array
from iqrp.app.timeseries.motifs.matrix_profile import compute_matrix_profile


def find_discords(
    x: np.ndarray | list[float],
    *,
    window: int = 32,
    top_k: int = 3,
) -> AnalysisResult:
    """Find top-k discords (largest matrix-profile values) (FULL_SAMPLE)."""
    y = as_float_array(x)
    w = max(int(window), 2)
    k = max(int(top_k), 1)
    mp_res = compute_matrix_profile(y, window=w)
    if isinstance(mp_res.value, str):
        return AnalysisResult(
            method="find_discords",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="no discord subsequences",
            alternative_hypothesis="one or more discord subsequences exist",
            parameters={"window": w, "top_k": k},
        )
    mp = np.asarray(mp_res.value["matrix_profile"], dtype=np.float64)
    order = np.argsort(np.where(np.isfinite(mp), mp, -np.inf))[::-1]
    discords: list[dict] = []
    used: list[int] = []
    for i in order:
        i = int(i)
        if not np.isfinite(mp[i]):
            continue
        if any(abs(i - u) < w for u in used):
            continue
        discords.append(
            {
                "index": i,
                "distance": float(mp[i]),
                "subsequence": y[i : i + w].tolist(),
            }
        )
        used.append(i)
        if len(discords) >= k:
            break
    return AnalysisResult(
        method="find_discords",
        value=discords,
        statistic=discords[0]["distance"] if discords else np.nan,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="no discord subsequences",
        alternative_hypothesis="one or more discord subsequences exist",
        significant=len(discords) > 0,
        parameters={"window": w, "top_k": k},
        metadata={"n_discords": len(discords), "n": y.size},
    )

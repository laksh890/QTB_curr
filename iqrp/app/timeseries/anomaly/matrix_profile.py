"""Matrix-profile based anomaly (discord) detection."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array
from iqrp.app.timeseries.motifs.matrix_profile import compute_matrix_profile


def matrix_profile_anomalies(
    x: np.ndarray | list[float],
    *,
    window: int = 32,
    top_k: int = 5,
    z_threshold: float = 2.0,
) -> AnalysisResult:
    """Flag discord subsequences via matrix profile peaks (FULL_SAMPLE)."""
    y = as_float_array(x)
    w = max(int(window), 4)
    k = max(int(top_k), 1)
    if y.size < 2 * w + 2:
        return AnalysisResult(
            method="matrix_profile_anomalies",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="no unusual subsequences (discords)",
            alternative_hypothesis="one or more discord subsequences present",
            parameters={"window": w, "top_k": k, "z_threshold": z_threshold},
        )
    mp_res = compute_matrix_profile(y, window=w)
    if isinstance(mp_res.value, str):
        return AnalysisResult(
            method="matrix_profile_anomalies",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="no unusual subsequences (discords)",
            alternative_hypothesis="one or more discord subsequences present",
            parameters={"window": w, "top_k": k, "z_threshold": z_threshold},
        )
    mp = np.asarray(mp_res.value["matrix_profile"], dtype=np.float64)
    finite = mp[np.isfinite(mp)]
    if finite.size < 3:
        return AnalysisResult(
            method="matrix_profile_anomalies",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="no unusual subsequences (discords)",
            alternative_hypothesis="one or more discord subsequences present",
            parameters={"window": w, "top_k": k, "z_threshold": z_threshold},
        )
    mu, sd = float(np.mean(finite)), float(np.std(finite))
    sd = sd if sd > 1e-12 else 1.0
    z = (mp - mu) / sd
    # top-k peaks with exclusion zone
    candidates = np.argsort(np.where(np.isfinite(mp), mp, -np.inf))[::-1]
    indices: list[int] = []
    for idx in candidates:
        if not np.isfinite(mp[idx]):
            continue
        if z[idx] < z_threshold and len(indices) == 0:
            # still take top discords even if below threshold once we have none
            pass
        if any(abs(int(idx) - j) < w for j in indices):
            continue
        if z[idx] >= z_threshold or len(indices) < k:
            indices.append(int(idx))
        if len(indices) >= k:
            break
    # keep only those above threshold for significance flag
    sig_indices = [i for i in indices if z[i] >= z_threshold]
    scores = np.full(y.size, np.nan, dtype=np.float64)
    scores[: mp.size] = mp
    mask = np.zeros(y.size, dtype=bool)
    mask[sig_indices] = True
    return AnalysisResult(
        method="matrix_profile_anomalies",
        value={
            "indices": sig_indices,
            "scores": scores,
            "is_anomaly": mask,
            "top_discords": indices,
        },
        statistic=float(np.nanmax(mp)),
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="no unusual subsequences (discords)",
        alternative_hypothesis="one or more discord subsequences present",
        significant=len(sig_indices) > 0,
        parameters={"window": w, "top_k": k, "z_threshold": z_threshold},
        metadata={"n": y.size, "profile_mean": mu, "profile_std": sd},
    )

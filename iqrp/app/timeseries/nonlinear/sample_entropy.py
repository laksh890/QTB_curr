"""Sample entropy (Richman & Moorman)."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def sample_entropy(
    x: np.ndarray | list[float],
    *,
    m: int = 2,
    r: float | None = None,
) -> AnalysisResult:
    """Sample entropy SampEn(m, r).

    Statistical descriptor only — not a guaranteed predictive signal.
    """
    y = as_float_array(x)
    finite = y[np.isfinite(y)]
    n = finite.size
    embed_m = max(int(m), 1)
    if n < embed_m + 3:
        return AnalysisResult(
            method="sample_entropy",
            value="insufficient_data",
            statistic=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="high SampEn (irregular / unpredictable)",
            alternative_hypothesis="low SampEn (regular / self-similar)",
            parameters={"m": embed_m, "r": r},
        )
    tol = float(r) if r is not None else 0.2 * float(np.std(finite, ddof=1))
    tol = max(tol, 1e-12)

    def _count(mm: int) -> float:
        nvec = n - mm + 1
        templates = np.lib.stride_tricks.as_strided(
            finite,
            shape=(nvec, mm),
            strides=(finite.strides[0], finite.strides[0]),
            writeable=False,
        ).copy()
        count = 0
        for i in range(nvec - 1):
            # Chebyshev distance to subsequent templates (exclude self)
            dists = np.max(np.abs(templates[i + 1 :] - templates[i]), axis=1)
            count += int(np.sum(dists <= tol))
        # number of comparisons
        n_comp = nvec * (nvec - 1) / 2.0
        return count / n_comp if n_comp > 0 else 0.0

    B = _count(embed_m)
    A = _count(embed_m + 1)
    if B < 1e-15 or A < 1e-15:
        sampen = np.inf
    else:
        sampen = float(-np.log(A / B))
    return AnalysisResult(
        method="sample_entropy",
        value=float(sampen) if np.isfinite(sampen) else sampen,
        statistic=float(sampen) if np.isfinite(sampen) else np.nan,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="high SampEn (irregular / unpredictable)",
        alternative_hypothesis="low SampEn (regular / self-similar)",
        significant=bool(np.isfinite(sampen) and sampen < 1.0),
        parameters={"m": embed_m, "r": tol},
        metadata={"n": n, "A": A, "B": B},
    )

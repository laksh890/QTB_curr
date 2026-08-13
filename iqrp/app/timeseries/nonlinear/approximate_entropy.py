"""Approximate entropy (Pincus)."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def approximate_entropy(
    x: np.ndarray | list[float],
    *,
    m: int = 2,
    r: float | None = None,
) -> AnalysisResult:
    """Approximate entropy ApEn(m, r).

    Statistical descriptor only — not a guaranteed predictive signal.
    """
    y = as_float_array(x)
    finite = y[np.isfinite(y)]
    n = finite.size
    embed_m = max(int(m), 1)
    if n < embed_m + 3:
        return AnalysisResult(
            method="approximate_entropy",
            value="insufficient_data",
            statistic=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="high ApEn (irregular / unpredictable)",
            alternative_hypothesis="low ApEn (regular / repetitive)",
            parameters={"m": embed_m, "r": r},
        )
    tol = float(r) if r is not None else 0.2 * float(np.std(finite, ddof=1))
    tol = max(tol, 1e-12)

    def _phi(mm: int) -> float:
        nvec = n - mm + 1
        templates = np.lib.stride_tricks.as_strided(
            finite,
            shape=(nvec, mm),
            strides=(finite.strides[0], finite.strides[0]),
            writeable=False,
        ).copy()
        # include self-matches (ApEn convention)
        C = np.empty(nvec, dtype=np.float64)
        for i in range(nvec):
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            C[i] = float(np.sum(dists <= tol)) / nvec
        return float(np.mean(np.log(np.clip(C, 1e-300, None))))

    apen = _phi(embed_m) - _phi(embed_m + 1)
    return AnalysisResult(
        method="approximate_entropy",
        value=float(apen),
        statistic=float(apen),
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="high ApEn (irregular / unpredictable)",
        alternative_hypothesis="low ApEn (regular / repetitive)",
        significant=apen < 0.5,
        parameters={"m": embed_m, "r": tol},
        metadata={"n": n},
    )

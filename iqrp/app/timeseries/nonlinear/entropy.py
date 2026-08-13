"""Shannon entropy of a discretized time series."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def shannon_entropy(
    x: np.ndarray | list[float],
    *,
    n_bins: int = 16,
    normalize: bool = True,
) -> AnalysisResult:
    """Histogram Shannon entropy of the marginal distribution.

    Statistical descriptor only — not a guaranteed predictive signal.
    """
    y = as_float_array(x)
    finite = y[np.isfinite(y)]
    n = finite.size
    bins = max(int(n_bins), 2)
    if n < bins:
        return AnalysisResult(
            method="shannon_entropy",
            value="insufficient_data",
            statistic=np.nan,
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="maximum-entropy (near-uniform) distribution",
            alternative_hypothesis="structured / peaked distribution (lower entropy)",
            parameters={"n_bins": bins, "normalize": normalize},
        )
    hist, _ = np.histogram(finite, bins=bins, density=False)
    p = hist.astype(np.float64)
    p = p[p > 0]
    p = p / p.sum()
    H = float(-np.sum(p * np.log(p)))
    H_max = float(np.log(bins))
    value = H / H_max if normalize and H_max > 0 else H
    return AnalysisResult(
        method="shannon_entropy",
        value=float(value),
        statistic=float(value),
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="maximum-entropy (near-uniform) distribution",
        alternative_hypothesis="structured / peaked distribution (lower entropy)",
        significant=bool(normalize and value < 0.8),
        parameters={"n_bins": bins, "normalize": normalize},
        metadata={"n": n, "raw_entropy": H, "max_entropy": H_max},
    )

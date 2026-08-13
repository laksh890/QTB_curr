"""Spectral density helpers and period conversion."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array
from iqrp.app.timeseries.spectral.welch import welch_psd


def spectral_density(
    x: np.ndarray | list[float],
    *,
    sample_rate: float = 1.0,
    nperseg: int = 64,
    method: str = "welch",
) -> AnalysisResult:
    """Estimate power spectral density (FULL_SAMPLE).

    ``method='welch'`` uses averaged periodograms; ``method='periodogram'``
    uses a single-window classical periodogram.
    """
    y = as_float_array(x)
    if y[np.isfinite(y)].size < 8:
        return AnalysisResult(
            method="spectral_density",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="white-noise flat spectrum",
            alternative_hypothesis="colored spectrum / periodic components",
            parameters={"sample_rate": sample_rate, "nperseg": nperseg, "method": method},
        )
    if method == "periodogram":
        from iqrp.app.timeseries.spectral.periodogram import periodogram

        res = periodogram(y, sample_rate=sample_rate, scaling="density")
    else:
        res = welch_psd(y, sample_rate=sample_rate, nperseg=nperseg)
    if isinstance(res.value, str):
        return AnalysisResult(
            method="spectral_density",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="white-noise flat spectrum",
            alternative_hypothesis="colored spectrum / periodic components",
            parameters={"sample_rate": sample_rate, "nperseg": nperseg, "method": method},
        )
    return AnalysisResult(
        method="spectral_density",
        value=res.value,
        statistic=res.statistic,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="white-noise flat spectrum",
        alternative_hypothesis="colored spectrum / periodic components",
        parameters={"sample_rate": sample_rate, "nperseg": nperseg, "method": method},
        metadata=dict(res.metadata),
    )


def period_from_frequency(
    frequency: float | np.ndarray | list[float],
    *,
    sample_rate: float = 1.0,
) -> AnalysisResult:
    """Convert frequency (cycles per sample-time unit) to period length."""
    f = np.asarray(frequency, dtype=np.float64).reshape(-1)
    fs = float(sample_rate) if sample_rate > 0 else 1.0
    if f.size == 0:
        return AnalysisResult(
            method="period_from_frequency",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis=None,
            alternative_hypothesis=None,
            parameters={"sample_rate": fs},
        )
    # period in sample units: T = 1/f when f is in cycles/sample;
    # if sample_rate given and f in Hz, T_seconds = 1/f, T_samples = fs/f
    with np.errstate(divide="ignore", invalid="ignore"):
        periods = np.where(np.abs(f) > 1e-15, 1.0 / f, np.inf)
        periods_samples = np.where(np.abs(f) > 1e-15, fs / f, np.inf)
    return AnalysisResult(
        method="period_from_frequency",
        value={"period": periods if periods.size > 1 else float(periods[0]),
               "period_samples": periods_samples if periods_samples.size > 1 else float(periods_samples[0])},
        temporal_mode=TemporalMode.FULL_SAMPLE,
        parameters={"sample_rate": fs},
        metadata={"frequencies": f.tolist()},
    )

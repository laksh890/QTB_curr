"""Periodogram spectral estimator."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def periodogram(
    x: np.ndarray | list[float],
    *,
    sample_rate: float = 1.0,
    detrend: bool = True,
    scaling: str = "density",
) -> AnalysisResult:
    """Classical (Schuster) periodogram (FULL_SAMPLE).

    Parameters
    ----------
    scaling:
        ``density`` returns power spectral density (units^2 / Hz);
        ``spectrum`` returns power spectrum (units^2).
    """
    y = as_float_array(x)
    finite = y[np.isfinite(y)]
    n = finite.size
    if n < 4:
        return AnalysisResult(
            method="periodogram",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="white-noise flat spectrum",
            alternative_hypothesis="colored spectrum / periodic components",
            parameters={"sample_rate": sample_rate, "detrend": detrend, "scaling": scaling},
        )
    z = finite.copy()
    if detrend:
        t = np.arange(n, dtype=np.float64)
        A = np.column_stack([np.ones(n), t])
        beta, *_ = np.linalg.lstsq(A, z, rcond=None)
        z = z - A @ beta
    fs = float(sample_rate) if sample_rate > 0 else 1.0
    window = np.hanning(n)
    zw = z * window
    win_power = float(np.sum(window**2))
    spec = np.fft.rfft(zw)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    if scaling == "spectrum":
        psd = (np.abs(spec) ** 2) / win_power
    else:
        psd = (np.abs(spec) ** 2) / (fs * win_power)
    if n % 2 == 0:
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0

    if psd.size > 2:
        total = float(np.sum(psd[1:]))
        g = float(np.max(psd[1:]) / total) if total > 0 else 0.0
    else:
        g = 0.0

    return AnalysisResult(
        method="periodogram",
        value={"frequencies": freqs, "psd": psd},
        statistic=g,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="white-noise flat spectrum",
        alternative_hypothesis="colored spectrum / periodic components",
        parameters={"sample_rate": fs, "detrend": detrend, "scaling": scaling},
        metadata={"n": n, "fisher_g": g, "window": "hann"},
    )

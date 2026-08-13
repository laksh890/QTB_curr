"""FFT-based spectral analysis."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def fft_spectrum(
    x: np.ndarray | list[float],
    *,
    sample_rate: float = 1.0,
    detrend: bool = True,
) -> AnalysisResult:
    """One-sided amplitude spectrum via rFFT (FULL_SAMPLE)."""
    y = as_float_array(x)
    finite = y[np.isfinite(y)]
    n = finite.size
    if n < 4:
        return AnalysisResult(
            method="fft_spectrum",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="flat spectrum (no dominant frequency)",
            alternative_hypothesis="one or more spectral peaks present",
            parameters={"sample_rate": sample_rate, "detrend": detrend},
        )
    z = finite.copy()
    if detrend:
        z = z - np.mean(z)
    fs = float(sample_rate) if sample_rate > 0 else 1.0
    spec = np.fft.rfft(z)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    amp = np.abs(spec) * 2.0 / n
    amp[0] = np.abs(spec[0]) / n
    power = (np.abs(spec) ** 2) / n
    return AnalysisResult(
        method="fft_spectrum",
        value={"frequencies": freqs, "amplitude": amp, "power": power},
        statistic=float(np.max(amp[1:])) if amp.size > 1 else float(amp[0]),
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="flat spectrum (no dominant frequency)",
        alternative_hypothesis="one or more spectral peaks present",
        parameters={"sample_rate": fs, "detrend": detrend, "n": n},
        metadata={"n": n, "freq_resolution": float(fs / n)},
    )


def dominant_frequencies(
    x: np.ndarray | list[float],
    *,
    sample_rate: float = 1.0,
    top_k: int = 3,
    detrend: bool = True,
    min_frequency: float = 0.0,
) -> AnalysisResult:
    """Extract top-k spectral peaks from the FFT amplitude spectrum (FULL_SAMPLE)."""
    res = fft_spectrum(x, sample_rate=sample_rate, detrend=detrend)
    if isinstance(res.value, str):
        return AnalysisResult(
            method="dominant_frequencies",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="flat spectrum (no dominant frequency)",
            alternative_hypothesis="one or more spectral peaks present",
            parameters={"sample_rate": sample_rate, "top_k": top_k},
        )
    freqs = np.asarray(res.value["frequencies"], dtype=np.float64)
    amp = np.asarray(res.value["amplitude"], dtype=np.float64)
    # exclude DC and below min_frequency
    mask = freqs > max(float(min_frequency), 0.0)
    if not np.any(mask):
        return AnalysisResult(
            method="dominant_frequencies",
            value=[],
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="flat spectrum (no dominant frequency)",
            alternative_hypothesis="one or more spectral peaks present",
            parameters={"sample_rate": sample_rate, "top_k": top_k},
        )
    f = freqs[mask]
    a = amp[mask]
    k = min(int(top_k), f.size)
    order = np.argsort(a)[::-1][:k]
    peaks = [
        {"frequency": float(f[i]), "amplitude": float(a[i]), "period": float(1.0 / f[i]) if f[i] > 1e-15 else np.inf}
        for i in order
    ]
    # significance vs mean amplitude
    mean_a = float(np.mean(a))
    significant = bool(peaks and peaks[0]["amplitude"] > 3.0 * mean_a) if mean_a > 0 else False
    return AnalysisResult(
        method="dominant_frequencies",
        value=peaks,
        statistic=peaks[0]["amplitude"] if peaks else 0.0,
        significant=significant,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="flat spectrum (no dominant frequency)",
        alternative_hypothesis="one or more spectral peaks present",
        parameters={"sample_rate": sample_rate, "top_k": top_k, "min_frequency": min_frequency},
        metadata={"mean_amplitude": mean_a},
    )

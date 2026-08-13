"""Welch's averaged periodogram PSD estimator."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def welch_psd(
    x: np.ndarray | list[float],
    *,
    sample_rate: float = 1.0,
    nperseg: int = 64,
    noverlap: int | None = None,
    detrend: bool = True,
) -> AnalysisResult:
    """Welch PSD with Hann windows and 50% overlap by default (FULL_SAMPLE)."""
    y = as_float_array(x)
    finite = y[np.isfinite(y)]
    n = finite.size
    seg = max(int(nperseg), 8)
    if n < seg:
        return AnalysisResult(
            method="welch_psd",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="white-noise flat spectrum",
            alternative_hypothesis="colored spectrum / periodic components",
            parameters={"sample_rate": sample_rate, "nperseg": seg, "noverlap": noverlap},
        )
    ov = int(noverlap) if noverlap is not None else seg // 2
    ov = int(np.clip(ov, 0, seg - 1))
    step = seg - ov
    fs = float(sample_rate) if sample_rate > 0 else 1.0
    window = np.hanning(seg)
    win_power = float(np.sum(window**2))
    freqs = np.fft.rfftfreq(seg, d=1.0 / fs)
    acc = np.zeros_like(freqs)
    n_segs = 0
    for start in range(0, n - seg + 1, step):
        chunk = finite[start : start + seg].copy()
        if detrend:
            t = np.arange(seg, dtype=np.float64)
            A = np.column_stack([np.ones(seg), t])
            beta, *_ = np.linalg.lstsq(A, chunk, rcond=None)
            chunk = chunk - A @ beta
        spec = np.fft.rfft(chunk * window)
        psd = (np.abs(spec) ** 2) / (fs * win_power)
        if seg % 2 == 0:
            psd[1:-1] *= 2.0
        else:
            psd[1:] *= 2.0
        acc += psd
        n_segs += 1
    if n_segs == 0:
        return AnalysisResult(
            method="welch_psd",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="white-noise flat spectrum",
            alternative_hypothesis="colored spectrum / periodic components",
            parameters={"sample_rate": fs, "nperseg": seg, "noverlap": ov},
        )
    psd_avg = acc / n_segs
    peak_freq = float(freqs[1 + int(np.argmax(psd_avg[1:]))]) if freqs.size > 1 else 0.0
    return AnalysisResult(
        method="welch_psd",
        value={"frequencies": freqs, "psd": psd_avg},
        statistic=float(np.max(psd_avg[1:])) if psd_avg.size > 1 else float(psd_avg[0]),
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="white-noise flat spectrum",
        alternative_hypothesis="colored spectrum / periodic components",
        parameters={"sample_rate": fs, "nperseg": seg, "noverlap": ov, "detrend": detrend},
        metadata={"n_segments": n_segs, "peak_frequency": peak_freq, "n": n},
    )

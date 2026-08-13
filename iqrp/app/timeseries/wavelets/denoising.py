"""Wavelet soft-threshold denoising via Haar DWT."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array
from iqrp.app.timeseries.wavelets.discrete import dwt_haar


def wavelet_denoise(
    x: np.ndarray | list[float],
    *,
    level: int | None = None,
    threshold: float | None = None,
    mode: str = "soft",
) -> AnalysisResult:
    """Haar wavelet denoising with universal or user threshold (FULL_SAMPLE)."""
    y = as_float_array(x)
    n_orig = y.size
    finite_mask = np.isfinite(y)
    if int(finite_mask.sum()) < 4:
        return AnalysisResult(
            method="wavelet_denoise",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="observations are pure noise (no recoverable signal)",
            alternative_hypothesis="signal component recoverable via sparse wavelet coeffs",
            parameters={"level": level, "threshold": threshold, "mode": mode},
        )
    dwt = dwt_haar(y[finite_mask], level=level)
    if isinstance(dwt.value, str):
        return AnalysisResult(
            method="wavelet_denoise",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="observations are pure noise (no recoverable signal)",
            alternative_hypothesis="signal component recoverable via sparse wavelet coeffs",
            parameters={"level": level, "threshold": threshold, "mode": mode},
        )
    details: list[np.ndarray] = [d.copy() for d in dwt.value["details"]]
    approx: np.ndarray = dwt.value["approximation"].copy()
    L = int(dwt.parameters["level"])

    # universal threshold from finest detail MAD
    finest = details[0]
    mad = float(np.median(np.abs(finest - np.median(finest)))) / 0.6745
    mad = mad if mad > 1e-15 else 1e-15
    n_used = int(dwt.parameters["n_used"])
    thr = float(threshold) if threshold is not None else mad * np.sqrt(2.0 * np.log(max(n_used, 2)))

    def _thresh(c: np.ndarray) -> np.ndarray:
        if mode == "hard":
            return np.where(np.abs(c) >= thr, c, 0.0)
        # soft
        return np.sign(c) * np.maximum(np.abs(c) - thr, 0.0)

    details = [_thresh(d) for d in details]
    recon = _idwt_haar(approx, details)

    out = np.full(n_orig, np.nan, dtype=np.float64)
    idx = np.flatnonzero(finite_mask)
    # recon length may be n_used
    m = min(recon.size, idx.size)
    out[idx[:m]] = recon[:m]
    # copy leftover finite points unchanged if padding trimmed one sample
    if m < idx.size:
        out[idx[m:]] = y[idx[m:]]

    noise_est = float(np.nanstd(y - out))
    return AnalysisResult(
        method="wavelet_denoise",
        value=out,
        statistic=thr,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="observations are pure noise (no recoverable signal)",
        alternative_hypothesis="signal component recoverable via sparse wavelet coeffs",
        parameters={"level": L, "threshold": thr, "mode": mode},
        metadata={"n": n_orig, "sigma_mad": mad, "residual_std": noise_est},
    )


def _idwt_haar(approx: np.ndarray, details: list[np.ndarray]) -> np.ndarray:
    """Inverse multi-level Haar DWT."""
    a = approx.copy()
    for d in reversed(details):
        # lengths should match
        m = min(a.size, d.size)
        a2 = a[:m]
        d2 = d[:m]
        even = (a2 + d2) / np.sqrt(2.0)
        odd = (a2 - d2) / np.sqrt(2.0)
        out = np.empty(2 * m, dtype=np.float64)
        out[0::2] = even
        out[1::2] = odd
        a = out
    return a

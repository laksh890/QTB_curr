"""Simplified continuous Morlet wavelet transform."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def cwt_morlet(
    x: np.ndarray | list[float],
    *,
    scales: np.ndarray | list[float] | None = None,
    omega0: float = 6.0,
    sample_rate: float = 1.0,
) -> AnalysisResult:
    """Simplified Morlet CWT via FFT convolution (FULL_SAMPLE).

    Uses the analytic Morlet wavelet
    ``ψ(t) = π^{-1/4} e^{i ω0 t} e^{-t²/2}`` (admissibility correction omitted
    for small ω0≈6 which is standard in practice).
    """
    y = as_float_array(x)
    finite = y[np.isfinite(y)]
    n = finite.size
    if n < 8:
        return AnalysisResult(
            method="cwt_morlet",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="no scale-localized oscillatory structure",
            alternative_hypothesis="time-frequency energy concentrations present",
            parameters={"omega0": omega0, "sample_rate": sample_rate},
        )
    z = finite - np.mean(finite)
    fs = float(sample_rate) if sample_rate > 0 else 1.0
    if scales is None:
        # dyadic scales covering ~2 to n/4 samples
        n_scales = min(32, max(4, int(np.log2(n)) * 4))
        sc = np.geomspace(2.0, max(n / 4.0, 2.0), n_scales)
    else:
        sc = as_float_array(scales)
        sc = sc[sc > 0]
    if sc.size == 0:
        return AnalysisResult(
            method="cwt_morlet",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="no scale-localized oscillatory structure",
            alternative_hypothesis="time-frequency energy concentrations present",
            parameters={"omega0": omega0, "sample_rate": fs},
        )

    # FFT of signal (zero-pad to next power of 2 for speed/circular-edge reduction)
    nfft = 1 << int(np.ceil(np.log2(n)))
    freqs = np.fft.fftfreq(nfft, d=1.0 / fs)
    sig_hat = np.fft.fft(z, n=nfft)
    coef = np.empty((sc.size, n), dtype=np.complex128)
    w0 = float(omega0)

    for i, s in enumerate(sc):
        # wavelet in frequency domain: Ψ(sω) scaled
        # Morlet FT ≈ π^{-1/4} H(ω) exp(-(sω - ω0)²/2) * sqrt(s)
        omega = 2.0 * np.pi * freqs
        sw = s * omega
        psi_hat = (np.pi**-0.25) * np.exp(-0.5 * (sw - w0) ** 2) * np.sqrt(s)
        psi_hat[freqs < 0] = 0.0  # analytic
        conv = np.fft.ifft(sig_hat * psi_hat)
        coef[i] = conv[:n]

    power = np.abs(coef) ** 2
    # ridge: scale of max power at each time
    ridge_idx = np.argmax(power, axis=0)
    ridge_scales = sc[ridge_idx]
    # dominant scale globally
    scale_energy = power.mean(axis=1)
    dom_i = int(np.argmax(scale_energy))

    return AnalysisResult(
        method="cwt_morlet",
        value={"coefficients": coef, "power": power, "scales": sc},
        statistic=float(scale_energy[dom_i]),
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="no scale-localized oscillatory structure",
        alternative_hypothesis="time-frequency energy concentrations present",
        parameters={"omega0": w0, "sample_rate": fs, "n_scales": int(sc.size)},
        metadata={
            "n": n,
            "dominant_scale": float(sc[dom_i]),
            "ridge_scales": ridge_scales,
            "scale_energy": scale_energy,
        },
    )

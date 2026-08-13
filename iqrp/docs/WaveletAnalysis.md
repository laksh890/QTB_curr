# Wavelet Analysis

Multi-scale time-frequency decomposition and denoising.
Coefficients and reconstructed series are **analytical measurements**, not signals.

## Location

`iqrp/app/timeseries/wavelets/`

- `discrete.py` — Haar DWT with per-level energy fractions
- `continuous.py` — simplified Morlet CWT (FFT convolution)
- `denoising.py` — soft/hard threshold Haar reconstruction

## Discrete Haar DWT

```python
from iqrp.app.timeseries.wavelets.discrete import dwt_haar

dwt = dwt_haar(x, level=None)  # auto max level from log2(n)
# value: approximation, details[], energy_fractions
```

## Continuous Morlet CWT

```python
from iqrp.app.timeseries.wavelets.continuous import cwt_morlet

cwt = cwt_morlet(x, scales=None, omega0=6.0, sample_rate=1.0)
# value: scales, coefficients (complex magnitude grid), frequencies
```

Dyadic scales default from ~2 to n/4 samples when `scales` is omitted.

## Denoising

```python
from iqrp.app.timeseries.wavelets.denoising import wavelet_denoise

den = wavelet_denoise(x, level=None, threshold=0.1, mode="soft")
# value: denoised series; universal threshold used if threshold is None
```

Engine:

```python
from iqrp.app.timeseries import TimeSeriesAnalyticsEngine

wav = TimeSeriesAnalyticsEngine().wavelet_analysis(x)
# keys: dwt, cwt, denoised
```

Hydra: `wavelet.wavelet` (haar), `wavelet.level`, `wavelet.threshold`.

## Temporal contract

All wavelet ops are `TemporalMode.FULL_SAMPLE` (bidirectional filters /
global thresholds). Denoised outputs must not be used as causal live features
without an explicitly causal redesign.

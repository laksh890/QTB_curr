# Spectral Analysis

Frequency-domain measurements of periodicity and power.
Peaks and PSD estimates are **evidence of cyclic structure**, not trading signals.

## Location

`iqrp/app/timeseries/spectral/`

- `fft.py` — one-sided rFFT amplitude / power; dominant peaks
- `periodogram.py` — classical (Hann-windowed) periodogram
- `welch.py` — Welch averaged PSD
- `spectral_density.py` — convenience wrapper + period helpers

## API

```python
from iqrp.app.timeseries.spectral.fft import fft_spectrum, dominant_frequencies
from iqrp.app.timeseries.spectral.periodogram import periodogram
from iqrp.app.timeseries.spectral.welch import welch_psd

fft = fft_spectrum(x, sample_rate=1.0, detrend=True)
# value: {"frequencies", "amplitude", "power"}

dom = dominant_frequencies(x, top_k=3, min_frequency=0.0)
# value: list of {frequency, amplitude, …} peaks (DC excluded)

peri = periodogram(x, scaling="density")   # or "spectrum"; Fisher g in statistic
welch = welch_psd(x, nperseg=64, noverlap=None)  # 50% overlap default
```

Engine battery:

```python
from iqrp.app.timeseries import TimeSeriesAnalyticsEngine

spec = TimeSeriesAnalyticsEngine().spectral_analysis(x)
# keys: fft, welch, periodogram, dominant
```

Hydra: `spectral.nperseg`, `spectral.detrend`.

## Dominant frequencies

`dominant_frequencies` ranks FFT amplitude peaks (excluding DC / below
`min_frequency`) and returns the top-k. Use as descriptive cycle candidates for
decomposition period choice — not as forecast horizons or trade triggers.

## Temporal contract

All estimators are `TemporalMode.FULL_SAMPLE`. For research diagnostics only;
do not treat spectral peaks as predictive alpha.

# Time-Series Analytics

Institutional Time-Series Analytics Platform for discovering structure in
financial series. **Not a forecasting engine** — outputs are measurements and
statistical evidence, never trading signals.

## Location

`iqrp/app/timeseries/`

| Area | Path |
|------|------|
| Engine façade | `orchestrator.py` (`TimeSeriesAnalyticsEngine`) |
| Settings | `config.py` / `configs/timeseries/default.yaml` |
| Result types | `base.py` (`AnalysisResult`, `DecompositionResult`, `ChangePointResult`, `TemporalMode`) |
| Registry | `registry.py` |
| Leakage-safe transforms | `transforms/`, `rolling.py`, `multiple_testing.py` |

## Engine API

```python
from iqrp.app.timeseries import TimeSeriesAnalyticsEngine, TimeSeriesSettings

eng = TimeSeriesAnalyticsEngine()
report = eng.analyze(x)          # stationarity, decompose, ACF, CPs, spectral, …
dec = eng.decompose(x)           # classical | stl | mstl
st = eng.stationarity(x)         # ADF / KPSS / PP / VR + multiple-testing adj.
cp = eng.change_points(x)        # cusum | binseg | pelt | bayesian | online
spec = eng.spectral_analysis(x)
wav = eng.wavelet_analysis(x)
dep = eng.dependence(x, y)
an = eng.anomalies(x)
mot = eng.motifs(x)
dtw = eng.dtw(x, y, soft=False)
```

Leakage-safe fit/transform:

```python
eng.fit(train).transform(test)
# or eng.fit_transform(train)
```

Persistence: `eng.save(path)` / `TimeSeriesAnalyticsEngine.load(path)`.

## Hydra config

Default: `iqrp/configs/timeseries/default.yaml`. Load via:

```python
settings = TimeSeriesSettings.from_hydra(
    overrides=["decomposition.method=classical", "change_points.method=cusum"]
)
eng = TimeSeriesAnalyticsEngine(settings)
```

Key blocks: `decomposition`, `stationarity`, `change_points`, `spectral`,
`wavelet`, `anomaly`, `motif`, `transform`, `multiple_testing`, `features`.

## Integration points

- **Transforms** — `TimeSeriesTransformer` with `rolling` / `expanding` /
  `training_only` contracts (see [LeakagePrevention](LeakagePrevention.md)).
- **Diagnostics / features** — `eng.diagnostics(x)`, `eng.features(x)`.
- **Visualization** — `eng.visualize(x)` returns chart payloads.
- **Synthetic tests** — `iqrp.app.timeseries.processes.simulate_process`.
- **Method registry** — `list_methods()` after `ensure_timeseries_loaded()`.

Every `AnalysisResult` carries `temporal_mode` and a disclaimer that values are
analytical evidence, not signals.

# Motif Discovery

Find repeated subsequences (motifs) and rare ones (discords) via matrix profile.
Motifs/discords are **pattern measurements**, not trading signals.

## Location

`iqrp/app/timeseries/motifs/`

- `matrix_profile.py` — z-normalized self-join matrix profile
- `discovery.py` — top-k motif pairs (nearest neighbors)
- `discord.py` — top-k discords (largest MP values)
- `similarity.py` — subsequence distance / nearest neighbors

## Matrix profile

```python
from iqrp.app.timeseries.motifs.matrix_profile import compute_matrix_profile

mp = compute_matrix_profile(x, window=32)
# value: {"matrix_profile", "profile_index"}
```

## Motifs and discords

```python
from iqrp.app.timeseries.motifs.discovery import find_motifs
from iqrp.app.timeseries.motifs.discord import find_discords

motifs = find_motifs(x, window=32, top_k=3, max_distance=None)
discords = find_discords(x, window=32, top_k=3)
```

Engine:

```python
from iqrp.app.timeseries import TimeSeriesAnalyticsEngine

eng = TimeSeriesAnalyticsEngine()
mot = eng.motifs(x)                       # settings.motif.window / top_k
disc = eng.detect(x, what="discords")
```

Hydra: `motif.window` (default 32), `motif.top_k` (default 3).

## Interpretation

- **Motifs** — subsequences with small nearest-neighbor distance (repeated shape).
- **Discords** — subsequences with large MP (most dissimilar to the rest);
  also used by `matrix_profile_anomalies`.

All motif ops are `TemporalMode.FULL_SAMPLE` (self-join uses the whole series).
Use for offline pattern discovery and labeling research only.

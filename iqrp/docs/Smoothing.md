# Smoothing

Offline and near-online smoothed latent-state inference.

## Location

`iqrp/app/state_space/smoothing/`

- `base_smoother.py`
- `fixed_interval.py` — classical forward–backward γ
- `fixed_lag.py` — sliding-window approximate smoother

## Fixed-interval

```python
from iqrp.app.state_space import FixedIntervalSmoother

smooth = FixedIntervalSmoother(settings).run(log_emissions, transition, filter_result=filt)
```

`SmootherResult` stores:

- `smoothed_states`
- `smoothed_probabilities` (γ)
- `backward_messages` (β)
- optional `log_likelihood` from the forward pass

## Fixed-lag

Configured by `smoothing.fixed_lag` (Hydra). For each time `t`, runs a local
fixed-interval smooth on a window of length `L` ending at / covering `t`.
Useful for streaming pipelines that cannot wait for the full series.

```python
from iqrp.app.state_space import FixedLagSmoother

smooth = FixedLagSmoother(settings).run(log_emissions, transition, lag=5)
```

## Model API

`StateSpaceModel.smooth(..., lag=None)` selects fixed-interval vs fixed-lag from
settings (or forces fixed-lag when `lag` is provided).

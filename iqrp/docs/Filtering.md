# Filtering

Forward and backward message passing for discrete latent-state models.

## Location

`iqrp/app/state_space/filtering/`

- `base_filter.py` — abstract filter contract
- `forward_filter.py` — scaled α recursion (+ chunked long-series mode)
- `backward_filter.py` — scaled β recursion

## Forward filter

Uses math-engine `logsumexp` for numerical stability:

```
log α_0(j) = log π_j + log b_0(j)
log α_t(j) = log b_t(j) + logsumexp_i[log α_{t-1}(i) + log P_ij]
```

Each step is scaled; `FilterResult.normalization_constants` stores `c_t` and
`log_likelihood = Σ_t log c_t`.

```python
from iqrp.app.state_space import ForwardFilter

result = ForwardFilter(settings).run(log_emissions, transition, initial=pi0)
```

`FilterResult` fields: `filtered_states`, `filtered_probabilities`,
`log_likelihood`, `normalization_constants`.

## Backward filter

```
log β_T(i) = 0
log β_t(i) = logsumexp_j[log P_ij + log b_{t+1}(j) + log β_{t+1}(j)]
```

Optional division by forward scales keeps messages commensurate with α.

## Memory efficiency

`filtering.chunk_size` (Hydra) stitches forward passes for long series without
materializing oversized temporaries beyond the chunk window.

## Probability utilities

`iqrp.app.state_space.base.probabilities`:

| Function | Role |
|----------|------|
| `forward_probabilities` | α, scales, LL |
| `backward_probabilities` | β |
| `state_occupancy_probabilities` | γ ∝ αβ |
| `transition_probabilities` | expected ξ |
| `joint_probabilities` | occupancy alias |
| `forecast_distribution` | π P^h |

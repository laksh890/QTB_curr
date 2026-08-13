# Transition Matrix

Count and probability matrices for the Markov Chain Engine.

## Location

`iqrp/app/regimes/markov/transition.py`, `estimator.py`

## Capabilities

| Feature | API |
|---------|-----|
| Transition counts | `TransitionMatrix.count_matrix()` |
| Probabilities | `probability_matrix(alpha=...)` |
| Row normalization | math-engine `normalize_rows` |
| Laplace smoothing | `laplace_alpha` |
| Dirichlet / Bayesian | `TransitionEstimator(method="bayesian")` |
| MLE / frequency | `method="mle"` / `"frequency"` |
| Weighted transitions | `method="weighted"` + `weights=` |
| Incremental updates | `update_pair`, `partial_fit`, forgetting factor |
| Sliding window | `apply_window(states, window_size)` |
| Sparse support | `sparse_probability_matrix()` (SciPy CSR) |

## Estimation methods

```python
from iqrp.app.regimes.markov import TransitionEstimator

est = TransitionEstimator(n_states=5, method="bayesian", dirichlet_alpha=1.0)
P = est.fit(states)
P2 = est.partial_fit(stream_chunk)  # exponential forgetting if configured
```

Online forgetting: `counts ← λ · counts` before accumulating new pairs when
`forgetting_factor < 1`.

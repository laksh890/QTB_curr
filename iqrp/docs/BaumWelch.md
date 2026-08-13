# Baum–Welch (EM)

Expectation–Maximization training for HMMs.

## Module

`iqrp/app/regimes/hmm/baum_welch.py`

**E-step:** forward–backward → γ, ξ  
**M-step:** Dirichlet-smoothed transitions; emission MLE (categorical or Gaussian)

```python
from iqrp.app.regimes.hmm import baum_welch

result = baum_welch(y, n_states=3, n_restarts=5, max_iter=100, tol=1e-4)
print(result.converged, result.log_likelihood, result.n_iter)
```

Supports warm start, early stopping, likelihood history, and threaded restarts.
`HMMTrainer.select_n_states` ranks 2..N models by AIC / BIC / LL.

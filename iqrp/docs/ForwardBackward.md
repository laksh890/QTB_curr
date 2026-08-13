# Forward–Backward Algorithm

Scaled log-space forward and backward message passing for HMMs.

## Modules

- `forward.py` — α recursion via state-space `forward_probabilities` (`logsumexp`)
- `backward.py` — β recursion
- `forward_backward.py` — γ posteriors, ξ expected transitions, occupancy

```python
from iqrp.app.regimes.hmm import forward_backward

fb = forward_backward(log_emissions, transition, initial=pi0)
print(fb.log_likelihood, fb.gamma.shape, fb.xi.shape)
```

Likelihood is `Σ_t log c_t` from forward scales. Posterior state probabilities
are row-normalized `γ_t ∝ α_t β_t`.

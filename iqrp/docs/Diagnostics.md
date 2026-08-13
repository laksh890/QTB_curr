# Diagnostics

Evaluation metrics and inference diagnostics for state-space models.

## Location

`iqrp/app/state_space/evaluation/`

- `metrics.py` — likelihood / information criteria / accuracy / stability
- `diagnostics.py` — occupancy, transitions, persistence, calibration, residuals

## Metrics

| Metric | Source |
|--------|--------|
| log likelihood | Filter marginal LL |
| AIC / BIC | Math-engine `aic` / `bic` |
| perplexity | `exp(-LL / T)` |
| state prediction accuracy | vs true labels |
| transition accuracy | pairwise transition match |
| cross entropy | Math-engine `cross_entropy` |
| state / persistence stability | run-length statistics |

```python
report = model.evaluate(frame, true_states=truth)
print(report["metrics"]["aic"], report["metrics"]["state_prediction_accuracy"])
```

## Diagnostics

```python
from iqrp.app.state_space import StateSpaceDiagnostics

diag = StateSpaceDiagnostics().analyze(
    states=pred,
    probabilities=proba,
    transition_matrix=P,
    observations=y,
    expected_observations=y_hat,
    log_likelihood_history=history,
)
```

Includes:

- state occupancy frequencies + entropy
- transition frequency / switch rate
- persistence run lengths vs model expected duration
- probability calibration (ECE-style bins)
- residual mean / variance / RMSE
- likelihood monotonicity / convergence checks

## Persistence

Artifacts (states, probabilities, forecasts, diagnostics) are written by
`StateStore` to Parquet + JSON with optional DuckDB views (`ss_states`,
`ss_transitions`, `ss_probabilities`).

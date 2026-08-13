# Hyperparameter Optimization

Configured via `TreeSettings.optimization.method`:

| Method | Description |
|--------|-------------|
| `none` | Use defaults / kwargs |
| `grid` | Limited cartesian grid |
| `random` | Random search |
| `bayesian` / `optuna` | Optuna TPE + MedianPruner |

```yaml
optimization:
  method: optuna
  n_trials: 25
  pruning: true
  parallel: true
  early_stopping: true
```

Search space covers `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `reg_lambda`.

## Time-series validation

`validation.strategy`:

- `walk_forward` / `rolling` / `expanding`
- `blocked`
- `purged_kfold`
- `embargo`

Used both for HPO scoring and `model.cross_validate()`.

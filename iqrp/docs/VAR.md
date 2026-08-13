# VAR / VARMAX

## VAR

`VARModel` fits a K-dimensional VAR(p) by equation-wise OLS.

Extras:

- `impulse_response(horizon=...)`
- `fevd(horizon=...)`
- `granger(cause, effect)`

## VARMAX

`VARMAXModel` extends VAR with contemporaneous exogenous regressors and optional residual MA(q).

Configure endogenous / exogenous columns in Hydra:

```yaml
columns:
  endogenous: [y0, y1]
  exogenous: [x0]
```

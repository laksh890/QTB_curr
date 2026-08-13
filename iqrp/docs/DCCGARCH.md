# DCC-GARCH & BEKK

## DCC-GARCH (`dcc_garch`)

Two-step Engle DCC:

1. Fit univariate GARCH/EWMA volatilities per asset → standardized residuals z_t
2. Estimate DCC(a,b) on the Q_t recursion; R_t = diag(Q)^{-1/2} Q diag(Q)^{-1/2}
3. H_t = D_t R_t D_t

### Outputs

- `conditional_volatility()` — first-asset σ path
- `forecast_covariance(horizon)` — H_{T+h} path
- `correlation_path()` — full R_t series
- Forecast metadata includes covariance and correlation tensors

## BEKK (`bekk`)

Scalar BEKK on two assets for numerical stability:

H_t = C C' + A ε_{t-1} ε'_{t-1} A' + B H_{t-1} B'

with A = a I, B = b I.

Multi-step forecasts mean-revert to the unconditional covariance.

## Cross-asset usage

```python
from iqrp.app.forecasting.volatility import create_volatility_model

model = create_volatility_model("dcc_garch")
model.fit(frame, feature_columns=["r0", "r1"])
cov = model.forecast_covariance(horizon=10)
```

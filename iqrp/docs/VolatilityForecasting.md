# Institutional Volatility Forecasting Engine

Production volatility forecasts for IQRP. Downstream **Risk**, **Portfolio**, **Execution**, and **Trading Bot** modules must consume volatility exclusively through this engine.

## Location

`iqrp/app/forecasting/volatility/`

## Supported models

| Name | Class | Notes |
|------|-------|-------|
| `historical_volatility` | HistoricalVolatilityModel | Full-sample σ |
| `rolling_volatility` | RollingVolatilityModel | Rolling realized vol |
| `ewma` | EWMAVolatilityModel | RiskMetrics λ |
| `arch` | ARCHModel | ARCH(p) |
| `garch` | GARCHModel | GARCH(p,q) |
| `egarch` | EGARCHModel | Leverage via log-variance |
| `gjr_garch` | GJRGARCHModel | Threshold leverage |
| `figarch` | FIGARCHModel | Long memory |
| `aparch` | APARCHModel | Power asymmetry |
| `component_garch` | ComponentGARCHModel | Permanent / transitory |
| `dcc_garch` | DCCGARCHModel | Dynamic correlations |
| `bekk` | BEKKModel | Multivariate BEKK |

All models inherit from the Forecasting Framework (`ForecastModel` → `VolatilityModel`).

## Quick start

```python
from iqrp.app.forecasting.volatility import (
    VolatilitySettings,
    VolatilityTrainer,
    create_volatility_model,
)

settings = VolatilitySettings.default()
model = create_volatility_model("garch", settings=settings)
model.fit(frame, target_column="returns")

sigma = model.conditional_volatility()
var = model.conditional_variance()
fc = model.forecast(frame, horizon=5)
cov = model.forecast_covariance(horizon=5)
diag = model.diagnostics()
report = model.evaluate(frame)
```

## API

- `fit` / `partial_fit` — MLE or closed-form estimation; online warm-start
- `predict` — in-sample conditional volatility
- `forecast` — N-step σ path with intervals / scenarios
- `conditional_variance` / `conditional_volatility`
- `forecast_covariance` — univariate variance path or multivariate H_t
- `evaluate` — QLIKE, RMSE, MAE, MSE, log-likelihood
- `diagnostics` — ARCH-LM, Ljung-Box, JB, persistence, half-life
- `save` / `load` — Forecasting Framework serialization

## Configuration

Hydra: `iqrp/configs/forecasting/volatility/default.yaml`

```python
VolatilitySettings.from_hydra(overrides=["order.p=1", "distribution.name=student_t"])
```

## Regime integration

When `regime` column is present and `regime.enabled=true`, models store per-regime parameters and can switch / ensemble-weight forecasts.

## Risk outputs

- Annualized volatility (`annualized_volatility`)
- Conditional / forecast variance and covariance
- Interval and scenario paths via `forecast` metadata

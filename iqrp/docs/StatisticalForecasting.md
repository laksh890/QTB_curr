# Institutional Statistical Forecasting Engine

Production-grade classical econometric forecasting on top of the IQRP Forecasting Framework.

## Models

| Name | Class | Family |
|------|-------|--------|
| `ar` | ARModel | Autoregressive |
| `ma` | MAModel | Moving average |
| `arma` | ARMAModel | ARMA |
| `arima` | ARIMAModel | ARIMA |
| `sarima` | SARIMAModel | Seasonal ARIMA |
| `var` | VARModel | Multivariate |
| `varmax` | VARMAXModel | Multivariate + exogenous |
| `vecm` | VECMModel | Error correction |
| `ses` | SimpleExpSmoothingModel | Exponential |
| `holt` | HoltModel | Exponential trend |
| `holt_winters` | HoltWintersModel | Seasonal exponential |

## Common API

Every model inherits `ForecastModel` / `StatisticalForecastModel`:

`fit`, `partial_fit`, `predict`, `forecast`, `forecast_interval`, `evaluate`, `residuals`, `diagnostics`, `save`, `load`

## Quick start

```python
from iqrp.app.forecasting.statistical import StatisticalTrainer, StatisticalSettings

trainer = StatisticalTrainer(StatisticalSettings.default())
model, result = trainer.fit("arima", frame, target_column="target")
fc = model.forecast(frame, horizon=10)
diag = model.diagnostics()
```

## Identification

Automatic lag / differencing / seasonality via ADF–KPSS heuristics and AIC/AICc/BIC/HQIC ranking (`select_arima_order`, `select_var_lags`).

## Regime integration

When a `regime` column is present, series are optionally demeaned by regime before fitting and the active regime is attached to `Forecast.regime_used`.

## Online learning

`partial_fit` supports expanding, sliding, and rolling windows with warm start.

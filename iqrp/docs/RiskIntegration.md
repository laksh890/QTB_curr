# Risk Integration

Volatility consumption contract for Risk (and Portfolio / Execution). For the full Institutional Risk Intelligence Framework see [RiskFramework.md](RiskFramework.md).

## Contract

All IQRP risk consumers must obtain volatility through:

```python
from iqrp.app.forecasting.volatility import create_volatility_model, VolatilityTrainer
```

Do not re-implement EWMA/GARCH inside Risk, Portfolio, Execution, or the Trading Bot.

## Risk outputs

| Method | Use |
|--------|-----|
| `conditional_volatility()` | Point-in-time σ_t |
| `annualized_volatility()` | σ_t √252 (configurable) |
| `conditional_variance()` | σ²_t for VaR/ES scaling |
| `forecast(horizon=h)` | Forward σ path + intervals |
| `forecast_covariance()` | Multivariate H_{T+h} for portfolio risk |
| `evaluate(...)` | Backtest QLIKE / RMSE vs realized variance |

## Recommended wiring

1. **Risk Engine** — daily `garch` or `ewma` fit; VaR = z_α · σ̂_{t+1} · exposure
2. **Portfolio Optimizer** — `dcc_garch` / `bekk` covariance forecasts
3. **Execution Engine** — short-horizon EWMA for participation / impact scaling
4. **Trading Bot** — regime-conditioned GJR/EGARCH for position sizing

## Online updates

```python
model.partial_fit(new_bars)  # expanding / rolling / warm-start adaptive
```

## Simulation validation

Use `base/processes.py` with the Institutional Market Simulation Engine to generate GARCH / DCC paths and verify parameter recovery before promoting models to production.

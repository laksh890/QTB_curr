# Model Risk

Model-risk monitors under `iqrp.app.risk.model_risk`, aggregated by `RiskIntelligenceEngine.model_risk_assessment()`.

These monitors **do not generate alpha**. They quantify instability of upstream forecasts so Risk can tighten sizing / escalate alerts. Hard portfolio limits remain independent of model confidence.

---

## Capabilities

| Monitor | Function | Signal |
|---------|----------|--------|
| Disagreement | `model_disagreement` | Cross-model dispersion at the latest PIT observation |
| Uncertainty | `forecast_uncertainty` | RMSE / MAE / bias vs realizations (aligned past only) |
| Drift | `model_drift` | Residual mean/vol shift, reference vs recent window |
| Parameter uncertainty | `parameter_uncertainty` | Bootstrap SE of mean or vol |
| Prediction instability | *(hook)* | Rolling forecast variance / path volatility — see below |
| Feature drift / calibration | *(hook)* | Consume Feature Validation / FI calibration metrics |

---

## Disagreement

```python
from iqrp.app.risk.model_risk import model_disagreement
import numpy as np

# Shape (M, T) — models × time
stack = np.vstack([f_lgbm, f_tft, f_garch])
d = model_disagreement(stack, axis=0)
print(d.value, d.parameters["range"], d.parameters["n_models"])
```

---

## Forecast uncertainty

```python
from iqrp.app.risk.model_risk import forecast_uncertainty

u = forecast_uncertainty(forecasts, realizations, window=60)
# value = RMSE; parameters include mae, bias
```

Only trailing aligned pairs — no look-ahead.

---

## Model drift

```python
from iqrp.app.risk.model_risk import model_drift

# residuals = y - ŷ on past bars
dr = model_drift(residuals, reference_window=60, test_window=20)
# Higher score → more residual distribution shift
```

---

## Parameter uncertainty

```python
from iqrp.app.risk.model_risk import parameter_uncertainty

se_mu = parameter_uncertainty(returns, n_bootstrap=500, seed=42, statistic="mean")
se_vol = parameter_uncertainty(returns, statistic="vol")
print(se_mu.value, se_mu.parameters["ci_low"], se_mu.parameters["ci_high"])
```

---

## Engine assessment

```python
from iqrp.app.risk import RiskIntelligenceEngine, RiskSettings

engine = RiskIntelligenceEngine(RiskSettings.default())

out = engine.model_risk_assessment(
    forecasts={"lgbm": f1, "tft": f2, "mean": f3},
    realizations=y,
    residuals=y - f_mean,
)
# out["disagreement"], out["uncertainty"], out["drift"]
```

Feed elevated scores into `build_alerts` / reduce `confidence` passed to `position_size` — never into hard-limit overrides.

---

## Prediction instability hooks

No dedicated symbol is required; compose from Forecast Intelligence outputs:

```python
import numpy as np

# Example: instability = recent std of point forecasts (or interval width)
instability = float(np.std(forecast_path[-20:]))
# Or coefficient of variation of rolling 1-step forecasts
roll = np.lib.stride_tricks.sliding_window_view(forecasts, 10).std(axis=-1)
prediction_instability = float(roll[-1])

# Gate sizing
confidence = float(np.clip(1.0 / (1.0 + prediction_instability), 0.0, 1.0))
engine.position_size(realized_vol=rv, confidence=confidence)
```

From Forecast Intelligence / calibration modules, also map:

- Interval width / predictive variance → `confidence` ↓  
- Calibration degradation (e.g. rising PIT error) → escalate WARNING alerts  
- Feature drift flags from Feature Validation → reject or shrink until refresh  

```python
from iqrp.app.risk.monitoring.alerts import build_alerts
from iqrp.app.risk import RiskState

alerts = build_alerts(
    risk_state=RiskState.CAUTION,
    measures={
        "model_drift": out["drift"],
        "forecast_uncertainty": out["uncertainty"],
        "model_disagreement": out["disagreement"],
    },
)
```

Values `> 2.0` on named model-risk measures emit WARNING-class model_risk alerts.

---

## Integration (import-only)

| Upstream | Use in Risk |
|----------|-------------|
| Forecast Intelligence | Multi-model forecasts, confidence, calibration, drift APIs |
| Volatility Forecasting | Residual diagnostics for vol models |
| Feature Platform / Validation | Feature drift inputs |
| Probability Engine | Optional distributional width for custom instability scores |

Risk imports these packages at the composition root; it does not mutate model registries or retrain models.

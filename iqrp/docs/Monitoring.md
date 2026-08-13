# Forecast Intelligence Monitoring

Online observation of accuracy, latency, calibration, and stability.

## Tracked quantities

- Forecast accuracy (MAE/RMSE/…) over a rolling window  
- Latency p50 / p95  
- Inference throughput  
- Calibration error  
- Feature & prediction stability  
- Alert flags (`mae_high`, `latency_high`, `calibration_poor`, `prediction_unstable`)

## Deployment

`deployment.DeploymentManager` promotes serialized engine state and supports rollback.

## Usage

```python
engine.monitor(y_true=0.01, y_pred=0.012)
snap = engine.monitor()
print(snap.to_dict())
engine.deploy(name="prod_v1")
engine.save("/tmp/fi.json")
loaded = ForecastIntelligenceEngine.load("/tmp/fi.json")
```

## Diagnostics & visualization

`engine.diagnose(frame)` residual report · `engine.visualize(frame)` chart payloads (leaderboard / forecast / drift).

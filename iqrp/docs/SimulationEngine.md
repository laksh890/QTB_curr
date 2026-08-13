# Institutional Market Simulation Engine

Reusable synthetic-market generator for validation, benchmarking, stress testing,
and algorithm development.

**Not for strategy optimization.** Future modules (Markov, HMM, Bayesian models,
XGBoost, risk, execution, portfolio) must be testable entirely against this engine.

## Location

`iqrp/app/simulation/`

## Configuration

Hydra defaults: `iqrp/configs/simulation/default.yaml`

```python
from iqrp.app.simulation import SimulationSettings, MarketSimulator, Scenario

settings = SimulationSettings.from_hydra(overrides=["n_steps=500", "default_model=heston"])
sim = MarketSimulator(settings)
market = sim.simulate(write_charts=True)
```

## Workflow

```python
from iqrp.app.simulation import MarketSimulator, Scenario

sim = MarketSimulator()
print(sim.available_models())  # gbm, abm, ou, merton_jump, heston, ...

scenario = Scenario.from_settings(name="stress", model="merton_jump", n_steps=2000)
market = sim.simulate(scenario, write_charts=True, validate=True)

# Ground truth for scoring future detectors / forecasters
truth = market.ground_truth
print(truth.regime_ids[:10], truth.transition_matrix)

# OHLCV + microstructure
ohlcv = market.ohlcv()
trades = market.trades
book = market.orderbook_snapshots
```

## Ground truth

Every run exposes:

| Field | Meaning |
|-------|---------|
| `regime_ids` / `regime_names` | True latent regime |
| `volatility` | True instantaneous / path volatility |
| `drift` | True drift path |
| `trend` | True trend strength / sign |
| `transition_matrix` | True Markov transition matrix |
| `event_mask` | Per-event injection indicators |

## Presets

```python
sim.simulate_preset("bull")
sim.simulate_preset("bear")
sim.simulate_preset("sideways")
sim.simulate_preset("high_volatility")
sim.simulate_preset("mixed")
```

## Related docs

- [StochasticModels.md](StochasticModels.md)
- [SyntheticMarkets.md](SyntheticMarkets.md)

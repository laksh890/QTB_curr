# Particle Filter

Institutional Sequential Monte Carlo engine for nonlinear / non-Gaussian latent market states.

## Location

`iqrp/app/regimes/particle/`

| Module | Role |
|--------|------|
| `particle.py` | Particle / ParticleCloud |
| `propagation.py` | Transition models + financial templates |
| `weighting.py` | Likelihoods, ESS, weight diagnostics |
| `resampling.py` | Multinomial / systematic / residual / stratified |
| `proposal.py` | Bootstrap & adaptive proposals |
| `rejuvenation.py` | Jitter / MCMC / covariance perturbation |
| `trainer.py` | Bootstrap, SIS, SIR, APF, RBPF, adaptive runners |
| `model.py` | State Space + Regime adapters |

## API

```python
from iqrp.app.regimes.particle import ParticleFilterModel, ParticleSettings

model = ParticleFilterModel(settings=ParticleSettings.from_hydra(
    overrides=["application=nonlinear_trend", "filter_type=bootstrap", "n_particles=300"]
))
model.fit(observations)
filt = model.filter(observations)
sm = model.smooth(observations)
fc = model.forecast(observations, horizon=5)
post = model.posterior()
ess = model.effective_sample_size()
```

Soft bullish/bearish probabilities feed `FilterResult` / `predict` for the regime contract.
Continuous posterior means live in `metadata` and `filtered_means()`.

## Applications

`nonlinear_trend`, `volatility`, `liquidity`, `dynamic_corr`, `market_stress`,
`risk_factors`, `custom`.

## Integration

Registered as `"particle"` on import. Hydra: `iqrp/configs/particle/default.yaml`.

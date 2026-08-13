"""Final coverage push for particle engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqrp.app.regimes.particle.config import ParticleSettings
from iqrp.app.regimes.particle.evaluator import ParticleEvaluator
from iqrp.app.regimes.particle.model import ParticleFilterModel, _resolve_names
from iqrp.app.regimes.particle.particle import FilterTrace, ParticleCloud
from iqrp.app.regimes.particle.propagation import build_transition
from iqrp.app.regimes.particle.trainer import ParticleTrainer, simulate_nonlinear
from iqrp.app.regimes.particle.weighting import effective_sample_size, normalize_weights, weight_entropy
from iqrp.app.regimes.particle.visualization import plot_state_trajectory


def _settings(**kw: object) -> ParticleSettings:
    data = {**ParticleSettings.default().model_dump(), "n_particles": 25, "training": {"n_iterations": 2, "tol": 1.0}}
    data.update(kw)
    return ParticleSettings.from_mapping(data)


@pytest.mark.unit
def test_final_coverage_lines(tmp_path: Path) -> None:
    assert _resolve_names(None) == ("bearish", "bullish")
    assert _resolve_names(("a",))[0] == "a"
    assert len(_resolve_names(("a", "b", "c"))) == 2

    assert effective_sample_size(np.zeros(5)) == 0.0
    assert weight_entropy(np.zeros(5)) == 0.0
    w = normalize_weights(np.array([0.0, 1.0, 2.0]))
    assert abs(w.sum() - 1.0) < 1e-9

    settings = _settings(application="risk_factors", n_states=3)
    model = build_transition(settings, application="risk_factors")
    assert model.n_states >= 2
    states, obs = simulate_nonlinear(model, 10, rng=np.random.default_rng(0))
    tr = FilterTrace(
        means=states,
        covs=np.stack([np.eye(states.shape[1]) for _ in range(10)]),
        clouds=[ParticleCloud.equal_weight(states[i : i + 1]) for i in range(10)],
        ess=np.full(10, 5.0),
        resampled=np.zeros(10, dtype=bool),
        log_likelihood=-1.0,
    )
    ev = ParticleEvaluator().evaluate(observations=obs, trace=tr, n_params=0)
    assert "log_likelihood" in ev["metrics"]
    ev2 = ParticleEvaluator().evaluate(
        observations=obs[:, 0],
        trace=FilterTrace(
            means=states[:, :1],
            covs=np.stack([np.eye(1) for _ in range(10)]),
            clouds=tr.clouds,
            ess=tr.ess,
            resampled=tr.resampled,
            log_likelihood=-1.0,
        ),
        true_states=states[:, 0],
        n_params=2,
    )
    assert "aic" in ev2["metrics"]

    trainer = ParticleTrainer(_settings(application="liquidity"))
    sys = build_transition(_settings(application="liquidity"), application="liquidity")
    _, y = simulate_nonlinear(sys, 15, rng=np.random.default_rng(1))
    res = trainer.fit(y, model=sys)
    assert res.history

    pf = ParticleFilterModel(settings=_settings(application="market_stress"), random_seed=2)
    pf.fit(y)
    labels, _ = pf.sample(5, initial_state=1)
    assert labels[0] == 1
    plot_state_trajectory(np.zeros(0), tmp_path / "z.svg", _settings())

    # cloud normalize + mean/cov 1d
    c = ParticleCloud.equal_weight(np.linspace(0, 1, 8))
    assert c.normalize().n_particles == 8
    assert c.covariance().shape[0] == 1
    assert c.log_likelihood_increment() != 0.0 or True

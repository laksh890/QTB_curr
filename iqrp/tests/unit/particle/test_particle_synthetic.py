"""Synthetic recovery, degeneracy, and stress tests."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.regimes.particle import (
    ParticleFilterModel,
    ParticleSettings,
    build_transition,
    filter_bootstrap,
    simulate_nonlinear,
)
from iqrp.app.regimes.particle.particle import ParticleCloud
from iqrp.app.regimes.particle.resampling import adaptive_resample
from iqrp.app.simulation.stochastic.jump_diffusion import MertonJumpDiffusion
from iqrp.app.simulation.stochastic.ou import OrnsteinUhlenbeck


def _settings(**kw: object) -> ParticleSettings:
    data = {
        **ParticleSettings.default().model_dump(),
        "n_particles": 150,
        "training": {"n_iterations": 1, "tol": 1e-3},
        "system": {
            **ParticleSettings.default().system.model_dump(),
            "process_noise_scale": 0.05,
            "observation_noise_scale": 0.05,
        },
    }
    data.update(kw)
    return ParticleSettings.from_mapping(data)


@pytest.mark.unit
def test_recover_nonlinear_trend() -> None:
    settings = _settings(application="nonlinear_trend", filter_type="bootstrap")
    model = build_transition(settings, application="nonlinear_trend")
    states, obs = simulate_nonlinear(model, 120, rng=np.random.default_rng(11), obs_scale=0.03)
    pf = ParticleFilterModel(settings=settings, transition=model, random_seed=11)
    pf.fit(obs)
    report = pf.evaluate(obs, true_states=states)
    assert report["metrics"]["state_corr"] >= 0.55
    assert report["metrics"]["mean_ess"] > 5
    assert report["metrics"]["posterior_coverage_95"] >= 0.5


@pytest.mark.unit
def test_simulation_engine_ou_and_jumps() -> None:
    ou = OrnsteinUhlenbeck(rng=np.random.default_rng(7))
    path = ou.generate(
        80, x0=100.0, dt=0.01, mean_reversion_speed=0.8, mean_reversion_level=100.0, volatility=3.0
    )
    prices = np.asarray(path.prices, dtype=np.float64).reshape(-1)
    settings = _settings(application="nonlinear_trend", n_particles=100)
    pf = ParticleFilterModel(settings=settings, random_seed=7)
    pf.fit(prices)
    assert pf.effective_sample_size() > 1
    jd = MertonJumpDiffusion(rng=np.random.default_rng(8))
    jpath = jd.generate(60, x0=100.0)
    rets = np.asarray(jpath.returns, dtype=np.float64).reshape(-1)
    pf2 = ParticleFilterModel(
        settings=_settings(application="volatility", likelihood="student_t", n_particles=80),
        random_seed=8,
    )
    pf2.fit(np.abs(rets) + 1e-3)
    assert np.isfinite(pf2.score(np.abs(rets) + 1e-3))


@pytest.mark.unit
def test_degeneracy_and_resampling_correctness() -> None:
    rng = np.random.default_rng(0)
    states = rng.normal(size=(100, 1))
    log_w = np.full(100, -50.0)
    log_w[0] = 0.0
    cloud = ParticleCloud(states=states, log_weights=log_w, likelihoods=np.ones(100))
    assert cloud.ess() < 5
    out, did = adaptive_resample(cloud, ess_threshold=0.5, method="systematic", rng=rng)
    assert did
    assert out.ess() > 50


@pytest.mark.unit
def test_stress_large_series() -> None:
    settings = _settings(
        application="custom",
        n_particles=40,
        filter_type="bootstrap",
        rejuvenation={"enabled": False, "method": "jitter", "scale": 0.01, "mcmc_steps": 1},
    )
    model = build_transition(settings, application="custom")
    _, obs = simulate_nonlinear(model, 5_000, rng=np.random.default_rng(3), obs_scale=0.1)
    tr = filter_bootstrap(obs, model, settings, rng=np.random.default_rng(3))
    assert tr.means.shape[0] == 5_000
    assert np.all(np.isfinite(tr.means))
    assert float(np.mean(tr.ess)) > 1

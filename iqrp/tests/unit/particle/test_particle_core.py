"""Core unit tests for particle filter engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.regimes.particle import (
    Particle,
    ParticleCloud,
    ParticleFilterModel,
    ParticleRegimeModel,
    ParticleSettings,
    build_transition,
    effective_sample_size,
    filter_adaptive,
    filter_auxiliary,
    filter_bootstrap,
    filter_rao_blackwellized,
    filter_sir,
    filter_sis,
    simulate_nonlinear,
    trajectory_smooth,
)
from iqrp.app.regimes.particle.prediction import (
    credible_interval,
    forecast_particles,
    particle_diversity,
)
from iqrp.app.regimes.particle.rejuvenation import rejuvenate
from iqrp.app.regimes.particle.resampling import (
    adaptive_resample,
    multinomial_resample,
    residual_resample,
    stratified_resample,
)
from iqrp.app.regimes.particle.visualization import (
    plot_credible_intervals,
    plot_ess_timeline,
    plot_particle_cloud,
    plot_resampling_timeline,
    plot_state_trajectory,
    plot_weight_histogram,
)
from iqrp.app.regimes.particle.weighting import log_likelihood, update_weights, weight_diagnostics


def _settings(**kw: object) -> ParticleSettings:
    data = {
        **ParticleSettings.default().model_dump(),
        "n_particles": 80,
        "training": {"n_iterations": 1, "tol": 1e-3},
        "rejuvenation": {"enabled": True, "method": "jitter", "scale": 0.02, "mcmc_steps": 1},
    }
    data.update(kw)
    return ParticleSettings.from_mapping(data)


@pytest.mark.unit
def test_particle_cloud_and_weights() -> None:
    cloud = ParticleCloud.equal_weight(np.random.default_rng(0).normal(size=(50, 2)))
    assert cloud.n_particles == 50 and cloud.dim == 2
    assert abs(cloud.weights.sum() - 1.0) < 1e-9
    assert cloud.ess() > 1
    parts = cloud.to_particles()
    assert len(parts) == 50
    cloud2 = ParticleCloud.from_particles(parts)
    assert cloud2.n_particles == 50
    d = cloud.to_dict()
    assert ParticleCloud.from_dict(d).n_particles == 50
    p = Particle(state=np.array([1.0, 2.0]), weight=0.5, log_weight=np.log(0.5))
    assert p.copy().state.shape[0] == 2
    ll = log_likelihood(np.array([0.1]), np.zeros((10, 1)), scale=0.2, kind="gaussian")
    assert ll.shape[0] == 10
    for kind in ("student_t", "laplace"):
        assert log_likelihood(np.array([0.0]), np.zeros((5, 1)), kind=kind).shape[0] == 5  # type: ignore[arg-type]
    upd = update_weights(cloud, np.zeros(50))
    assert abs(upd.weights.sum() - 1.0) < 1e-8
    assert "ess" in weight_diagnostics(upd)
    assert effective_sample_size(upd.weights) > 0


@pytest.mark.unit
def test_resampling_schemes() -> None:
    rng = np.random.default_rng(1)
    w = rng.random(40)
    w = w / w.sum()
    for fn in (multinomial_resample, residual_resample, stratified_resample):
        idx = fn(w, rng=rng)
        assert idx.shape[0] == 40
    cloud = ParticleCloud.equal_weight(rng.normal(size=(40, 1)))
    # force low ESS
    log_w = np.full(40, -1e6)
    log_w[0] = 0.0
    cloud = ParticleCloud(states=cloud.states, log_weights=log_w, likelihoods=np.ones(40))
    out, did = adaptive_resample(cloud, ess_threshold=0.9, method="systematic", rng=rng)
    assert did and out.ess() > cloud.ess() * 0.5


@pytest.mark.unit
def test_filters_and_smoothing() -> None:
    settings = _settings(application="nonlinear_trend", filter_type="bootstrap")
    model = build_transition(settings, application="nonlinear_trend")
    states, obs = simulate_nonlinear(model, 40, rng=np.random.default_rng(2), obs_scale=0.05)
    rng = np.random.default_rng(2)
    for runner in (
        filter_bootstrap,
        filter_sis,
        filter_sir,
        filter_auxiliary,
        filter_adaptive,
        filter_rao_blackwellized,
    ):
        tr = runner(obs, model, settings, rng=rng)
        assert tr.means.shape[0] == obs.shape[0]
        assert np.isfinite(tr.log_likelihood)
    tr = filter_bootstrap(obs, model, settings, rng=np.random.default_rng(3))
    sm = trajectory_smooth(tr, model, n_trajectories=20, rng=np.random.default_rng(3))
    assert sm.means.shape[0] == obs.shape[0]
    means, covs, clouds = forecast_particles(tr.clouds[-1], model, horizon=3, rng=rng)
    assert means.shape == (3, model.n_states)
    lo, hi = credible_interval(tr.clouds[-1])
    assert lo <= hi
    assert 0 < particle_diversity(tr.clouds[-1]) <= 1
    rej = rejuvenate(tr.clouds[-1], method="mcmc", model=model, observation=obs[-1], rng=rng)
    assert rej.n_particles == tr.clouds[-1].n_particles


@pytest.mark.unit
def test_model_api(tmp_path: Path) -> None:
    settings = _settings(application="nonlinear_trend", n_particles=60)
    model = build_transition(settings, application="nonlinear_trend")
    states, obs = simulate_nonlinear(model, 50, rng=np.random.default_rng(4), obs_scale=0.05)
    pf = ParticleFilterModel(settings=settings, transition=model, random_seed=4)
    pf.fit(obs)
    pred = pf.predict(obs)
    proba = pf.predict_proba(obs)
    assert pred.shape[0] == obs.shape[0] and proba.shape == (obs.shape[0], 2)
    assert np.isfinite(pf.score(obs))
    sm = pf.smooth(obs)
    assert sm.smoothed_states.shape[0] == obs.shape[0]
    fc = pf.forecast(obs, horizon=3)
    assert fc.horizon == 3
    labels, samp = pf.sample(20)
    assert labels.shape[0] == 20 and samp.shape[0] == 20
    assert "mean" in pf.posterior()
    lo, hi = pf.credible_interval()
    assert lo <= hi
    assert pf.effective_sample_size() > 0
    pf.resample()
    x = pf.update(obs[-1])
    assert x.shape[0] >= 1
    path = pf.save(tmp_path / "pf.json")
    loaded = ParticleFilterModel.load(path)
    assert loaded.is_fitted
    diag = pf.diagnostics(obs)
    assert "ess" in diag
    report = pf.evaluate(obs, true_states=states)
    assert "state_corr" in report["metrics"] or "log_likelihood" in report["metrics"]
    plot_particle_cloud(pf._cloud.states, pf._cloud.weights, tmp_path / "c.svg")  # type: ignore[union-attr]
    plot_weight_histogram(pf._cloud.weights, tmp_path / "w.svg")  # type: ignore[union-attr]
    plot_state_trajectory(pf.filtered_means(), tmp_path / "t.svg", observations=obs[:, 0])
    plot_credible_intervals(
        pf.filtered_means()[:, 0],
        pf.filtered_means()[:, 0] - 0.1,
        pf.filtered_means()[:, 0] + 0.1,
        tmp_path / "ci.svg",
    )
    plot_ess_timeline(pf._trace.ess, tmp_path / "e.svg")  # type: ignore[union-attr]
    plot_resampling_timeline(pf._trace.resampled, tmp_path / "r.svg")  # type: ignore[union-attr]
    frame = pl.DataFrame({"open_time": list(range(obs.shape[0])), "close": obs[:, 0]})
    regime = ParticleRegimeModel(settings=settings, random_seed=5)
    regime.fit(frame, feature_columns=["close"])
    assert regime.predict_proba(frame, feature_columns=["close"]).shape[1] == 2
    assert regime.forecast(frame, steps=2).steps == 2


@pytest.mark.unit
def test_applications() -> None:
    for app in (
        "nonlinear_trend",
        "volatility",
        "liquidity",
        "dynamic_corr",
        "market_stress",
        "risk_factors",
        "custom",
    ):
        s = _settings(application=app)
        m = build_transition(s, application=app)  # type: ignore[arg-type]
        assert m.n_states >= 1
        _, obs = simulate_nonlinear(m, 15, rng=np.random.default_rng(6))
        assert obs.shape[0] == 15

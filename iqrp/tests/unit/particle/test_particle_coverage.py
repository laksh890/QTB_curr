"""Broad coverage tests for particle modules."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from iqrp.app.core.exceptions import ConfigurationError, ValidationError
from iqrp.app.regimes.particle.adaptive import resize_cloud, suggest_n_particles
from iqrp.app.regimes.particle.config import ParticleSettings
from iqrp.app.regimes.particle.model import ParticleFilterModel, ParticleRegimeModel, _soft_trend_proba
from iqrp.app.regimes.particle.particle import ParticleCloud
from iqrp.app.regimes.particle.prediction import posterior_summary
from iqrp.app.regimes.particle.propagation import TransitionModel, build_transition
from iqrp.app.regimes.particle.proposal import adaptive_proposal, proposal_diagnostics
from iqrp.app.regimes.particle.rejuvenation import (
    adaptive_perturbation,
    covariance_perturbation,
    gaussian_jitter,
    rejuvenate,
)
from iqrp.app.regimes.particle.serializer import _json_default
from iqrp.app.regimes.particle.smoothing import backward_smooth_means, trajectory_smooth
from iqrp.app.regimes.particle.trainer import ParticleTrainer, initialize_cloud, run_filter, simulate_nonlinear
from iqrp.app.regimes.particle.visualization import _ensure, plot_particle_cloud, plot_state_trajectory
from iqrp.app.regimes.particle.weighting import log_likelihood


def _settings(**kw: object) -> ParticleSettings:
    data = {
        **ParticleSettings.default().model_dump(),
        "n_particles": 40,
        "training": {"n_iterations": 2, "tol": 1.0},
    }
    data.update(kw)
    return ParticleSettings.from_mapping(data)


@pytest.mark.unit
def test_config_paths_and_errors() -> None:
    s = ParticleSettings.default()
    assert s.enabled
    s2 = ParticleSettings.from_mapping(OmegaConf.create({"n_particles": 10, "n_states": 1}))
    assert s2.n_particles == 10
    with pytest.raises(ConfigurationError):
        ParticleSettings.from_mapping("bad")  # type: ignore[arg-type]
    s3 = ParticleSettings.from_hydra(overrides=["filter_type=sir", "n_particles=20"])
    assert s3.filter_type == "sir"
    bad = Path("/tmp/pf_bad_cfg.yaml")
    bad.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        ParticleSettings.from_hydra(bad)


@pytest.mark.unit
def test_trainer_filter_types_and_likelihoods() -> None:
    trainer = ParticleTrainer(_settings(application="custom"))
    model = trainer.build_model()
    _, obs = simulate_nonlinear(model, 20, rng=np.random.default_rng(1))
    for ft in ("bootstrap", "sis", "sir", "auxiliary", "rao_blackwellized", "adaptive"):
        tr = run_filter(obs, model, _settings(filter_type=ft), rng=np.random.default_rng(1))  # type: ignore[arg-type]
        assert tr.means.shape[0] == 20
    for kind in ("gaussian", "student_t", "laplace"):
        ll = log_likelihood(np.array([0.1]), np.zeros((5, 1)), kind=kind)  # type: ignore[arg-type]
        assert ll.shape[0] == 5
    custom = log_likelihood(
        np.array([0.0]),
        np.zeros((3, 1)),
        kind="custom",
        custom_fn=lambda z, y: -0.5 * (y[:, 0] - z[0]) ** 2,
    )
    assert custom.shape[0] == 3
    res = trainer.fit(obs, model=model, rng=np.random.default_rng(2))
    assert len(res.history) >= 1


@pytest.mark.unit
def test_proposal_rejuvenation_adaptive() -> None:
    rng = np.random.default_rng(3)
    settings = _settings(application="nonlinear_trend")
    model = build_transition(settings, application="nonlinear_trend")
    cloud = initialize_cloud(30, 1, settings, rng=rng)
    prop, ratio = adaptive_proposal(cloud, model, np.array([0.5]), rng=rng)
    assert prop.n_particles == 30 and ratio.shape[0] == 30
    assert "n_particles" in proposal_diagnostics(prop)
    assert gaussian_jitter(cloud, scale=0.01, rng=rng).n_particles == 30
    assert covariance_perturbation(cloud, scale=0.01, rng=rng).n_particles == 30
    assert adaptive_perturbation(cloud, scale=0.01, rng=rng).n_particles == 30
    assert rejuvenate(cloud, method="covariance", rng=rng).n_particles == 30
    assert rejuvenate(cloud, method="adaptive", rng=rng).n_particles == 30
    n = suggest_n_particles(5, 100, min_particles=50, max_particles=200)
    assert 50 <= n <= 200
    big = resize_cloud(cloud, 60, rng=rng)
    assert big.n_particles == 60


@pytest.mark.unit
def test_model_online_errors_serialize(tmp_path: Path) -> None:
    settings = _settings(application="denoise") if False else _settings(application="custom")
    model = ParticleFilterModel(settings=settings, random_seed=4)
    with pytest.raises(ValidationError):
        model.posterior()
    y = np.cumsum(np.random.default_rng(4).normal(0, 0.1, size=30))
    model.fit(y[:15])
    model.partial_fit(y[15:25])
    settings2 = _settings(online={"warm_start": False, "checkpoint_every": 0})
    m2 = ParticleFilterModel(settings=settings2, random_seed=5)
    m2.fit(y[:10])
    m2.partial_fit(y[10:20])
    with pytest.raises(ValidationError):
        ParticleFilterModel(settings=settings).diagnostics()
    frame = pl.DataFrame({"open_time": list(range(20)), "close": y[:20], "symbol": ["X"] * 20})
    m3 = ParticleFilterModel(settings=_settings(application="custom"), random_seed=6)
    m3.fit(frame)
    assert np.isfinite(m3.aic(frame)) and np.isfinite(m3.bic(frame))
    with pytest.raises(ValidationError):
        ParticleFilterModel(settings=_settings())._extract_obs(pl.DataFrame({"symbol": ["a"]}))
    p = _soft_trend_proba(np.array([1.0, -1.0]), np.stack([np.eye(1), np.eye(1)]))
    assert p.shape == (2, 2)
    assert isinstance(_json_default(np.array([1.0])), list)
    assert isinstance(_json_default(np.float64(1.0)), float)
    assert isinstance(_json_default(np.bool_(True)), bool)
    with pytest.raises(TypeError):
        _json_default(object())
    off = _settings(visualization={"enabled": False, "max_points": 10, "max_particles_plot": 10})
    plot_state_trajectory(np.zeros(5), tmp_path / "off.svg", off)
    _ensure(tmp_path / "x.svg", off)
    # transition dict roundtrip
    tr = TransitionModel.from_dict(model.transition.to_dict())  # type: ignore[union-attr]
    assert tr.application == model.transition.application  # type: ignore[union-attr]
    summary = posterior_summary(model._cloud)  # type: ignore[arg-type]
    assert "credible_intervals" in summary


@pytest.mark.unit
def test_smoothing_empty_and_regime() -> None:
    from iqrp.app.regimes.particle.particle import FilterTrace

    empty = FilterTrace(
        means=np.zeros((0, 1)),
        covs=np.zeros((0, 1, 1)),
        clouds=[],
        ess=np.zeros(0),
        resampled=np.zeros(0, dtype=bool),
        log_likelihood=0.0,
    )
    sm = trajectory_smooth(empty, build_transition(_settings()))
    assert sm.means.shape[0] == 0
    settings = _settings(application="custom")
    model = build_transition(settings)
    _, obs = simulate_nonlinear(model, 12, rng=np.random.default_rng(9))
    tr = run_filter(obs, model, settings, rng=np.random.default_rng(9))
    means, covs = backward_smooth_means(tr)
    assert means.shape[0] == 12
    frame = pl.DataFrame({"x": obs.reshape(-1)})
    regime = ParticleRegimeModel(settings=settings, random_seed=10)
    regime.fit(frame, feature_columns=["x"])
    st = regime._algorithm_state()
    regime2 = ParticleRegimeModel(settings=settings)
    regime2._load_algorithm_state(st)
    assert regime2.is_fitted
    assert regime.predict(frame, feature_columns=["x"]).shape[0] == 12

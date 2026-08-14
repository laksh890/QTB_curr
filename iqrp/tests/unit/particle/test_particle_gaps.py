"""Gap-filling tests for particle coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

import iqrp.app.regimes.particle
from iqrp.app.regimes.base.registry import get_registry as get_regime_registry
from iqrp.app.regimes.particle import (
    ParticleFilterModel,
    ParticleSettings,
    build_transition,
    simulate_nonlinear,
)
from iqrp.app.regimes.particle.particle import ParticleCloud
from iqrp.app.regimes.particle.propagation import TransitionModel
from iqrp.app.regimes.particle.rejuvenation import mcmc_rejuvenation
from iqrp.app.regimes.particle.resampling import apply_resampling, residual_resample
from iqrp.app.regimes.particle.serializer import ParticleSerializer
from iqrp.app.regimes.particle.trainer import initialize_cloud, run_filter
from iqrp.app.regimes.particle.visualization import (
    plot_ess_timeline,
    plot_particle_cloud,
    plot_posterior_evolution,
    plot_resampling_timeline,
    plot_state_trajectory,
    plot_weight_histogram,
)
from iqrp.app.state_space import get_registry as get_ss_registry


def _settings(**kw: object) -> ParticleSettings:
    data = {
        **ParticleSettings.default().model_dump(),
        "n_particles": 30,
        "training": {"n_iterations": 1, "tol": 1e-3},
    }
    data.update(kw)
    return ParticleSettings.from_mapping(data)


@pytest.mark.unit
def test_registry_and_linear_propagate() -> None:
    assert "particle" in get_ss_registry().list_names()
    assert "particle" in get_regime_registry().list_names()
    model = TransitionModel(f=np.eye(2), q_scale=0.1)
    rng = np.random.default_rng(0)
    out = model.propagate(np.zeros((5, 2)), rng=rng)
    assert out.shape == (5, 2)
    assert model.observe(out).shape[1] == 1


@pytest.mark.unit
def test_residual_pad_and_cholesky_fallback() -> None:
    rng = np.random.default_rng(1)
    w = np.array([0.99, 0.01])
    idx = residual_resample(w, rng=rng)
    assert idx.shape[0] == 2
    cloud = ParticleCloud.equal_weight(rng.normal(size=(20, 2)))
    with patch("numpy.linalg.cholesky", side_effect=np.linalg.LinAlgError("fail")):
        from iqrp.app.regimes.particle.rejuvenation import covariance_perturbation

        out = covariance_perturbation(cloud, scale=0.1, rng=rng)
        assert out.n_particles == 20


@pytest.mark.unit
def test_columns_and_vol_roundtrip(tmp_path: Path) -> None:
    settings = _settings(
        application="volatility",
        filter_type="bootstrap",
        columns={"timestamp": "t", "observation_columns": ("px",)},
        rejuvenation={"enabled": True, "method": "mcmc", "scale": 0.05, "mcmc_steps": 1},
    )
    model = build_transition(settings, application="volatility")
    _, obs = simulate_nonlinear(model, 25, rng=np.random.default_rng(2), obs_scale=0.05)
    obs = np.abs(obs) + 0.01
    frame = pl.DataFrame({"t": list(range(25)), "px": obs[:, 0]})
    pf = ParticleFilterModel(settings=settings, transition=model, random_seed=2)
    pf.fit(frame)
    pf.smooth(frame)
    path = ParticleSerializer().save(pf, tmp_path / "vol.json")
    loaded = ParticleSerializer().load(path, model_cls=ParticleFilterModel)
    assert loaded.transition is not None
    assert loaded.transition.transition_fn is not None


@pytest.mark.unit
def test_viz_empty_and_bands(tmp_path: Path) -> None:
    settings = _settings(
        visualization={"enabled": True, "max_points": 50, "max_particles_plot": 50}
    )
    plot_state_trajectory(np.array([]), tmp_path / "empty.svg", settings)
    plot_weight_histogram(np.array([]), tmp_path / "wh.svg", settings)
    plot_particle_cloud(np.zeros((0, 2)), np.array([]), tmp_path / "pc.svg", settings)
    plot_ess_timeline(np.linspace(1, 10, 10), tmp_path / "ess.svg", settings)
    plot_resampling_timeline(np.array([0, 1, 0, 1]), tmp_path / "rs.svg", settings)
    plot_posterior_evolution(np.linspace(0, 1, 10), tmp_path / "pe.svg", settings)


@pytest.mark.unit
def test_mcmc_and_apply_resample() -> None:
    settings = _settings(application="custom")
    model = build_transition(settings)
    rng = np.random.default_rng(3)
    cloud = initialize_cloud(15, 1, settings, rng=rng)
    out = mcmc_rejuvenation(cloud, model, np.array([0.0]), scale=0.1, steps=2, rng=rng)
    assert out.n_particles == 15
    rs = apply_resampling(cloud, method="multinomial", rng=rng)
    assert "resample_indices" in rs.metadata
    # custom transition without f
    bare = TransitionModel(q_scale=0.2)
    x = bare.propagate(np.zeros((4, 1)), rng=rng)
    assert x.shape == (4, 1)


@pytest.mark.unit
def test_default_config_missing(tmp_path: Path) -> None:
    with patch(
        "iqrp.app.regimes.particle.config._default_config_path",
        return_value=tmp_path / "missing.yaml",
    ):
        s = ParticleSettings.default()
        assert s.n_particles == 500

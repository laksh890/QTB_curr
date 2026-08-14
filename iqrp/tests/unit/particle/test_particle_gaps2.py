"""Additional coverage gaps for particle engine (>98%)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqrp.app.core.exceptions import ValidationError
from iqrp.app.regimes.particle.config import ParticleSettings
from iqrp.app.regimes.particle.diagnostics import ParticleDiagnostics
from iqrp.app.regimes.particle.evaluator import ParticleEvaluator
from iqrp.app.regimes.particle.model import ParticleFilterModel
from iqrp.app.regimes.particle.particle import FilterTrace, ParticleCloud
from iqrp.app.regimes.particle.propagation import TransitionModel, build_transition
from iqrp.app.regimes.particle.resampling import resample_indices, residual_resample
from iqrp.app.regimes.particle.smoothing import SmoothTrace, trajectory_smooth
from iqrp.app.regimes.particle.trainer import (
    filter_adaptive,
    filter_bootstrap,
    filter_sir,
    initialize_cloud,
    run_filter,
    simulate_nonlinear,
)
from iqrp.app.regimes.particle.visualization import (
    _line_plot,
    plot_particle_cloud,
    plot_weight_histogram,
)
from iqrp.app.regimes.particle.weighting import log_likelihood


def _settings(**kw: object) -> ParticleSettings:
    data = {
        **ParticleSettings.default().model_dump(),
        "n_particles": 30,
        "training": {"n_iterations": 1, "tol": 1e-3},
    }
    data.update(kw)
    return ParticleSettings.from_mapping(data)


@pytest.mark.unit
def test_diagnostics_smooth_and_evaluator_smooth() -> None:
    settings = _settings(application="custom")
    model = build_transition(settings)
    states, obs = simulate_nonlinear(model, 12, rng=np.random.default_rng(0))
    tr = filter_bootstrap(obs, model, settings, rng=np.random.default_rng(0))
    sm = trajectory_smooth(tr, model, n_trajectories=8, rng=np.random.default_rng(0))
    diag = ParticleDiagnostics().report(model, tr, smooth=sm, history=[-1.0])
    assert "smoothed" in diag
    # constant truth → state_corr 0
    ev = ParticleEvaluator().evaluate(
        observations=obs,
        trace=tr,
        smooth=sm,
        true_states=np.zeros((12, 1)),
        n_params=2,
    )
    assert ev["metrics"]["state_corr"] == 0.0
    assert "smooth_mse" in ev["metrics"]


@pytest.mark.unit
def test_model_n_particles_override_and_diag_paths() -> None:
    settings = _settings(application="custom", n_particles=40)
    pf = ParticleFilterModel(n_particles=35, settings=settings, random_seed=1)
    y = np.cumsum(np.random.default_rng(1).normal(0, 0.1, size=20))
    pf.fit(y)
    assert pf._pf_settings.n_particles == 35
    # partial_fit sets train_obs when None already set - clear and refill
    pf2 = ParticleFilterModel(settings=_settings(application="custom"), random_seed=2)
    pf2.fit(y[:10])
    pf2._train_obs = None
    pf2.partial_fit(y[10:15])
    assert pf2._train_obs is not None
    # diagnostics via train_obs
    pf2._trace = None
    d = pf2.diagnostics()
    assert "ess" in d
    with pytest.raises(ValidationError):
        ParticleFilterModel(settings=_settings()).diagnostics()


@pytest.mark.unit
def test_resample_pad_and_methods() -> None:
    rng = np.random.default_rng(3)
    # residual with tiny weights that may need pad
    w = np.array([1.0 - 1e-15, 1e-15])
    idx = residual_resample(w, rng=rng)
    assert idx.shape[0] == 2
    assert resample_indices(w, method="stratified", rng=rng).shape[0] == 2
    assert resample_indices(w, method="systematic", rng=rng).shape[0] == 2


@pytest.mark.unit
def test_propagation_hooks_and_1d_cloud() -> None:
    def trans(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return x + 0.01

    def obs(x: np.ndarray) -> np.ndarray:
        return x[:, :1] * 2

    m = TransitionModel(transition_fn=trans, observe_fn=obs, metadata={"n_states": 3}, q_scale=0.1)
    assert m.n_states == 3
    rng = np.random.default_rng(4)
    out = m.propagate(np.zeros(3), rng=rng)  # 1d input
    assert out.shape[0] >= 1
    assert m.observe(np.zeros(3)).shape[-1] == 1
    cloud = ParticleCloud(
        states=np.linspace(0, 1, 5), log_weights=np.zeros(5), likelihoods=np.ones(5)
    )
    assert cloud.states.ndim == 2


@pytest.mark.unit
def test_smoothing_identity_f_and_custom_ll() -> None:
    model = TransitionModel(f=None, q_scale=0.1, metadata={"n_states": 1})
    settings = _settings()
    cloud = initialize_cloud(10, 1, settings, rng=np.random.default_rng(5))
    tr = FilterTrace(
        means=np.zeros((3, 1)),
        covs=np.stack([np.eye(1)] * 3),
        clouds=[cloud, cloud, cloud],
        ess=np.ones(3) * 10,
        resampled=np.zeros(3, dtype=bool),
        log_likelihood=0.0,
    )
    sm = trajectory_smooth(tr, model, n_trajectories=4, rng=np.random.default_rng(5))
    assert sm.means.shape[0] == 3
    # custom likelihood without fn falls through to gaussian
    ll = log_likelihood(np.array([0.0]), np.zeros((4, 1)), kind="custom", custom_fn=None)
    assert ll.shape[0] == 4


@pytest.mark.unit
def test_trainer_branches_and_viz(tmp_path: Path) -> None:
    settings = _settings(
        filter_type="sir",
        rejuvenation={"enabled": True, "method": "jitter", "scale": 0.01, "mcmc_steps": 1},
        resampling={"adaptive": True, "ess_threshold": 0.99, "method": "systematic"},
    )
    model = build_transition(settings)
    _, obs = simulate_nonlinear(model, 15, rng=np.random.default_rng(6))
    tr = filter_sir(obs, model, settings, rng=np.random.default_rng(6))
    assert tr.metadata["filter"] == "sir"
    # adaptive with proposal_adapt false
    s2 = _settings(
        filter_type="adaptive",
        adaptive={
            "enabled": True,
            "min_particles": 20,
            "max_particles": 80,
            "target_ess_fraction": 0.5,
            "proposal_adapt": False,
        },
    )
    tr2 = filter_adaptive(obs, model, s2, rng=np.random.default_rng(7))
    assert tr2.metadata["filter"] == "adaptive"
    # run_filter default bootstrap
    tr3 = run_filter(obs, model, _settings(filter_type="bootstrap"), rng=np.random.default_rng(8))
    assert tr3.means.shape[0] == 15
    # viz branches: empty series with bands, 1d states for cloud
    settings_v = _settings()
    _line_plot([], tmp_path / "e.svg", title="t", settings=settings_v)
    plot_particle_cloud(np.arange(10), np.ones(10) / 10, tmp_path / "1d.svg", settings_v)
    plot_weight_histogram(np.linspace(0, 1, 20), tmp_path / "h.svg", settings_v)
    # disabled viz early return in plot_particle_cloud
    off = _settings(visualization={"enabled": False, "max_points": 10, "max_particles_plot": 10})
    plot_particle_cloud(np.zeros((5, 2)), np.ones(5) / 5, tmp_path / "off.svg", off)

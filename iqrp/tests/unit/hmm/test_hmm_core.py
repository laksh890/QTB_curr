"""Core unit tests for the HMM engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.regimes.hmm import (
    DiscreteEmissionModel,
    GaussianEmissionModel,
    HiddenMarkovModel,
    HMMSettings,
    HMMTrainer,
    baum_welch,
    forward,
    forward_backward,
    viterbi,
)
from iqrp.app.regimes.hmm.initialization import initialize_parameters
from iqrp.app.regimes.hmm.model import HMMRegimeModel
from iqrp.app.regimes.hmm.visualization import (
    plot_emission_means,
    plot_hidden_state_timeline,
    plot_likelihood_curve,
    plot_posterior_heatmap,
    plot_state_duration_histogram,
    plot_transition_heatmap,
    plot_viterbi_path,
)
from iqrp.app.simulation.regimes.regime_switching import RegimeSwitchingSimulator


def _gauss_series(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = RegimeSwitchingSimulator.mixed_transition(2, 0.92)
    states = [0]
    for _ in range(n - 1):
        states.append(int(rng.choice(2, p=p[states[-1]])))
    states_a = np.asarray(states, dtype=np.int64)
    y = np.where(states_a == 0, rng.normal(-1.5, 0.3, n), rng.normal(1.5, 0.3, n))
    return states_a, y.reshape(-1, 1)


@pytest.mark.unit
def test_settings_hydra() -> None:
    s = HMMSettings.from_hydra(overrides=["n_states=4", "training.max_iter=50"])
    assert s.n_states == 4
    assert s.training.max_iter == 50
    assert HMMSettings.default().enabled is True


@pytest.mark.unit
def test_emissions_discrete_and_gaussian() -> None:
    de = DiscreteEmissionModel(2, 3, alpha=1.0)
    log_p = de.log_prob(np.array([0, 1, 2, 0]))
    assert log_p.shape == (4, 2)
    samp = de.sample(np.array([0, 1, 0]), rng=np.random.default_rng(0))
    assert samp.shape[0] == 3
    gamma = np.full((4, 2), 0.5)
    de.m_step(np.array([0, 1, 2, 0]), gamma)
    assert DiscreteEmissionModel.from_dict(de.to_dict()).n_symbols == 3

    ge = GaussianEmissionModel(2, 1, means=[[-1.0], [1.0]], covars=[[0.25], [0.25]])
    assert ge.log_prob(np.linspace(-2, 2, 20)).shape == (20, 2)
    ge.m_step(np.linspace(-2, 2, 40).reshape(-1, 1), np.tile([0.7, 0.3], (40, 1)))
    assert GaussianEmissionModel.from_dict(ge.to_dict()).covariance_type == "diag"

    full = GaussianEmissionModel(2, 2, covariance_type="full")
    y = np.random.default_rng(1).normal(size=(50, 2))
    full.m_step(y, np.full((50, 2), 0.5))
    assert full.log_prob(y).shape == (50, 2)


@pytest.mark.unit
def test_forward_backward_viterbi_stable() -> None:
    _, y = _gauss_series(120, seed=1)
    trans, emis = initialize_parameters(y, 2, method="kmeans", rng=np.random.default_rng(1))
    log_e = emis.log_prob(y)
    alpha, _scales, ll = forward(log_e, trans.transition, initial=trans.initial)
    assert alpha.shape == (120, 2)
    assert np.isfinite(ll)
    fb = forward_backward(log_e, trans.transition, initial=trans.initial)
    assert fb.gamma.shape == (120, 2)
    assert fb.xi.shape == (119, 2, 2)
    vit = viterbi(log_e, trans.transition, initial=trans.initial)
    assert vit.states.shape == (120,)
    assert vit.confidence.min() >= 0


@pytest.mark.unit
def test_baum_welch_and_model_api(tmp_path: Path) -> None:
    truth, y = _gauss_series(250, seed=2)
    result = baum_welch(y, 2, n_restarts=2, max_iter=40, tol=1e-4, rng=np.random.default_rng(2))
    assert result.n_iter >= 1
    assert np.isfinite(result.log_likelihood)

    model = HiddenMarkovModel(n_states=2, random_seed=2)
    model.fit(y)
    assert model.is_fitted
    assert model.decode(y).shape == (len(y),)
    assert model.predict_proba(y).shape == (len(y), 2)
    assert model.filter(y).n_steps == len(y)
    assert model.smooth(y).n_steps == len(y)
    alpha, _scales, ll = model.forward(y)
    assert alpha.shape[0] == len(y) and np.isfinite(ll)
    assert model.backward(y).shape == alpha.shape
    fc = model.forecast(y, horizon=4)
    assert fc.horizon == 4
    states, obs = model.sample(30, initial_state=0)
    assert states.shape == (30,) and obs.shape[0] == 30
    assert np.isfinite(model.log_likelihood(y))
    assert np.isfinite(model.aic(y)) and np.isfinite(model.bic(y))
    report = model.evaluate(y, true_states=truth)
    assert report["metrics"]["prediction_accuracy"] >= 0.7
    diag = model.diagnostics(y)
    assert "convergence" in diag
    path = model.save(tmp_path / "hmm.json")
    loaded = HiddenMarkovModel.load(path)
    assert loaded.is_fitted
    assert np.allclose(loaded.transition_matrix(), model.transition_matrix(), atol=1e-6)

    frame = pl.DataFrame({"open_time": list(range(len(y))), "feat": y.ravel()})
    model2 = HiddenMarkovModel(n_states=2, random_seed=3)
    model2.fit(frame, observation_columns=["feat"])
    model2.partial_fit(frame[:40], observation_columns=["feat"])


@pytest.mark.unit
def test_regime_adapter_and_trainer() -> None:
    _, y = _gauss_series(150, seed=4)
    frame = pl.DataFrame({"x": y.ravel()})
    reg = HMMRegimeModel(n_states=2, random_seed=4)
    reg.fit(frame, feature_columns=["x"])
    assert reg.predict(frame, feature_columns=["x"]).shape == (len(y),)
    assert reg.forecast(frame, steps=3).steps == 3
    trainer = HMMTrainer(
        HMMSettings.from_mapping(
            {
                **HMMSettings.default().model_dump(),
                "initialization": {"method": "random", "n_restarts": 2, "dirichlet_alpha": 1.0},
                "training": {
                    "max_iter": 25,
                    "tol": 1e-3,
                    "early_stopping": True,
                    "min_covar": 1e-6,
                    "n_jobs": 1,
                },
                "model_selection": {"min_states": 2, "max_states": 3, "criterion": "bic"},
            }
        )
    )
    sel = trainer.select_n_states(y, rng=np.random.default_rng(4))
    assert sel["best_n_states"] in (2, 3)


@pytest.mark.unit
def test_visualization(tmp_path: Path) -> None:
    settings = HMMSettings.default()
    p = np.array([[0.9, 0.1], [0.2, 0.8]])
    proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])
    plot_hidden_state_timeline([0, 1, 1, 0], tmp_path / "tl.svg", settings)
    plot_posterior_heatmap(proba, tmp_path / "hm.svg", settings)
    plot_transition_heatmap(p, tmp_path / "tm.svg", settings)
    plot_likelihood_curve([-100.0, -90.0, -88.0], tmp_path / "ll.svg", settings)
    plot_viterbi_path([0, 0, 1], tmp_path / "vt.svg", settings)
    plot_state_duration_histogram([1, 2, 2, 4], tmp_path / "dur.svg", settings)
    plot_emission_means([[-1.0], [1.0]], tmp_path / "em.svg", settings)
    disabled = HMMSettings.from_mapping(
        {**settings.model_dump(), "visualization": {"enabled": False, "max_points": 10}}
    )
    plot_hidden_state_timeline([0], tmp_path / "off.svg", disabled)
    assert (tmp_path / "tl.svg").read_text().startswith("<svg")

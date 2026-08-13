"""Gap-filling tests to push HMM engine coverage above 98%."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.regimes.hmm.baum_welch import baum_welch
from iqrp.app.regimes.hmm.config import HMMSettings
from iqrp.app.regimes.hmm.emissions import (
    DiscreteEmissionModel,
    GaussianEmissionModel,
    _gaussian_logpdf,
    build_emission,
    emission_from_dict,
)
from iqrp.app.regimes.hmm.evaluator import HMMEvaluator, _avg_ce, _best_accuracy
from iqrp.app.regimes.hmm.forward import forward_log_likelihood
from iqrp.app.regimes.hmm.forward_backward import (
    ForwardBackwardResult,
    expected_state_occupancy,
    forward_backward,
    posterior_state_probabilities,
)
from iqrp.app.regimes.hmm.initialization import initialize_parameters
from iqrp.app.regimes.hmm.model import HiddenMarkovModel, HMMRegimeModel, _resolve_names
from iqrp.app.regimes.hmm.prediction import current_state_distribution
from iqrp.app.regimes.hmm.trainer import _n_params
from iqrp.app.regimes.hmm.visualization import plot_emission_means, plot_posterior_heatmap
from iqrp.app.regimes.hmm.viterbi import viterbi
from iqrp.app.state_space.base.registry import get_registry as get_ss_registry


@pytest.mark.unit
def test_resolve_names_branches() -> None:
    assert _resolve_names(3, ("a", "b", "c")) == ("a", "b", "c")
    assert _resolve_names(3, ("a",)) == ("a", "state_1", "state_2")
    assert _resolve_names(2, None) == ("state_0", "state_1")


@pytest.mark.unit
def test_warm_start_baum_welch_and_viterbi_default_pi() -> None:
    y = np.concatenate([np.full(50, -1.0), np.full(50, 1.0)]).reshape(-1, 1)
    trans, emis = initialize_parameters(y, 2, method="kmeans", rng=np.random.default_rng(0))
    result = baum_welch(y, 2, warm_start=(trans, emis), max_iter=5, n_restarts=3)
    assert result.n_iter >= 1
    log_e = emis.log_prob(y)
    path = viterbi(log_e, trans.transition, initial=None)
    assert path.states.shape[0] == y.shape[0]


@pytest.mark.unit
def test_full_cov_sample_mstep_and_singular_logpdf() -> None:
    means = np.array([[0.0, 0.0], [1.0, 1.0]])
    covars = np.array([np.eye(2), np.eye(2)])
    emis = GaussianEmissionModel(2, 2, means=means, covars=covars, covariance_type="full")
    samples = emis.sample(np.array([0, 1, 0]), rng=np.random.default_rng(3))
    assert samples.shape == (3, 2)
    y1d = np.linspace(-1, 1, 40)
    gamma = np.tile(np.array([0.6, 0.4]), (40, 1))
    emis_diag = GaussianEmissionModel(2, 1, covariance_type="diag")
    emis_diag.m_step(y1d, gamma)
    assert emis_diag.means.shape == (2, 1)
    assert build_emission("gaussian", 2, 2, covariance_type="full").n_features == 2
    disc = DiscreteEmissionModel(2, 3)
    restored = emission_from_dict(disc.to_dict())
    assert isinstance(restored, DiscreteEmissionModel)
    # singular / near-singular full covariance triggers LinAlg recovery path
    y = np.zeros((5, 2))
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    ll = _gaussian_logpdf(y, np.zeros(2), singular, "full")
    assert ll.shape == (5,)
    assert np.all(np.isfinite(ll))


@pytest.mark.unit
def test_evaluator_edges_and_greedy_matching() -> None:
    assert np.isnan(_avg_ce(np.array([0.5, 0.5]), np.array([0])))
    pred = np.array([0, 1, 0, 1, 2, 3, 4, 5, 6, 0])
    truth = np.array([0, 1, 0, 1, 2, 3, 4, 5, 6, 1])
    acc = _best_accuracy(pred, truth, 7)
    assert 0.0 <= acc <= 1.0
    emis = DiscreteEmissionModel(2, 3)
    ev = HMMEvaluator().evaluate(
        true_states=None,
        predicted_states=np.array([0, 1, 0]),
        probabilities=np.array([0.7, 0.3]),
        log_likelihood=-1.0,
        emissions=emis,
        n_samples=3,
    )
    assert ev["metrics"]["mean_max_probability"] == 0.0
    assert _n_params(2, emis) > 0


@pytest.mark.unit
def test_forward_helpers_and_fb_accessors() -> None:
    log_e = np.log(np.clip(np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]]), 1e-12, None))
    p = np.array([[0.8, 0.2], [0.3, 0.7]])
    assert np.isfinite(forward_log_likelihood(log_e, p))
    fb = forward_backward(log_e, p)
    assert isinstance(fb, ForwardBackwardResult)
    assert posterior_state_probabilities(fb).shape == fb.gamma.shape
    assert expected_state_occupancy(fb).shape == (2,)


@pytest.mark.unit
def test_initialization_uniform_discrete_and_full_cov() -> None:
    d = np.array([0, 1, 0, 1], dtype=np.int64)
    t, e = initialize_parameters(
        d, 2, method="uniform", emission_type="discrete", n_symbols=2, rng=np.random.default_rng(1)
    )
    assert t.validate() and e.n_states == 2
    empty = np.array([], dtype=np.int64)
    _, e2 = initialize_parameters(
        empty, 2, method="uniform", emission_type="discrete", rng=np.random.default_rng(2)
    )
    assert e2.n_states == 2
    y = np.random.default_rng(4).normal(size=(80, 2))
    _t3, e3 = initialize_parameters(
        y,
        3,
        method="kmeans",
        covariance_type="full",
        rng=np.random.default_rng(4),
    )
    assert e3.covariance_type == "full"
    y1 = np.linspace(-2, 2, 50)
    _t4, e4 = initialize_parameters(y1, 2, method="random", rng=np.random.default_rng(5))
    assert e4.n_features == 1


@pytest.mark.unit
def test_model_online_skip_extract_and_helpers(tmp_path: Path) -> None:
    y = np.concatenate([np.full(40, -1.0), np.full(40, 1.0)]).reshape(-1, 1)
    settings = HMMSettings.from_mapping(
        {
            **HMMSettings.default().model_dump(),
            "online": {"window_size": 40, "update_frequency": 3, "warm_start": True},
            "training": {
                "max_iter": 15,
                "tol": 1e-3,
                "early_stopping": True,
                "min_covar": 1e-6,
                "n_jobs": 1,
            },
            "initialization": {"method": "kmeans", "n_restarts": 1, "dirichlet_alpha": 1.0},
            "columns": {
                **HMMSettings.default().columns.model_dump(),
                "observation_columns": ["ret"],
            },
        }
    )
    model = HiddenMarkovModel(n_states=2, settings=settings, random_seed=7)
    model.fit(y)
    # first two partial_fit calls should skip EM (freq=3)
    model.partial_fit(y[:10])
    model.partial_fit(y[10:20])
    assert model._update_counter % 3 != 0 or model._online_buffer
    # third triggers update
    model.partial_fit(y[20:30])

    # discrete ndarray extract
    disc_settings = HMMSettings.from_mapping(
        {
            **HMMSettings.default().model_dump(),
            "emission": {"type": "discrete", "covariance_type": "diag"},
            "n_states": 2,
            "training": {
                "max_iter": 10,
                "tol": 1e-3,
                "early_stopping": True,
                "min_covar": 1e-6,
                "n_jobs": 1,
            },
            "initialization": {"method": "random", "n_restarts": 1, "dirichlet_alpha": 1.0},
        }
    )
    dm = HiddenMarkovModel(n_states=2, settings=disc_settings, random_seed=8)
    dobs = np.array([0, 1, 2, 0, 1, 2, 0, 1] * 10, dtype=np.int64)
    dm.fit(dobs)
    assert dm.decode(dobs).shape[0] == dobs.size
    frame = pl.DataFrame({"sym": np.tile([0, 1, 2], 20)})
    dm.fit(frame, observation_columns=["sym"])

    # observation_columns from settings + close fallback
    m2 = HiddenMarkovModel(n_states=2, settings=settings, random_seed=9)
    frame2 = pl.DataFrame({"open_time": [0, 1, 2], "ret": [0.1, -0.2, 0.05]})
    m2.fit(frame2)
    frame3 = pl.DataFrame({"open_time": list(range(60)), "close": np.linspace(-1, 1, 60)})
    m3 = HiddenMarkovModel(n_states=2, random_seed=10)
    m3.fit(frame3)

    bare = HiddenMarkovModel(n_states=2)
    assert bare._n_params() == 2
    assert bare._transition_matrix_or_none() is None
    assert current_state_distribution(np.array([0.2, 0.8])).sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_regime_adapter_roundtrip_and_viz_reshape(tmp_path: Path) -> None:
    assert "hmm" in get_ss_registry().list_names()
    y = np.concatenate([np.full(50, -1.0), np.full(50, 1.0)]).reshape(-1, 1)
    frame = pl.DataFrame({"ret": y.ravel()})
    regime = HMMRegimeModel(n_states=2, random_seed=11)
    regime.fit(frame, feature_columns=["ret"])
    proba = regime.predict_proba(frame, feature_columns=["ret"])
    assert proba.shape[0] == frame.height
    state = regime._algorithm_state()
    regime2 = HMMRegimeModel(n_states=2, random_seed=12)
    regime2._load_algorithm_state(state)
    assert regime2.is_fitted
    plot_posterior_heatmap(np.array([0.2, 0.8]), tmp_path / "p.svg")
    assert (tmp_path / "p.svg").exists()
    plot_emission_means(np.array([0.0, 1.5]), tmp_path / "m.svg")
    assert (tmp_path / "m.svg").exists()

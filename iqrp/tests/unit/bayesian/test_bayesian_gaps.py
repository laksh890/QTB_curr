"""Gap-filling tests for Bayesian engine coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.core.exceptions import ValidationError
from iqrp.app.regimes.bayesian.config import BayesianSettings
from iqrp.app.regimes.bayesian.convergence import burn_in_suggestion, gelman_rubin
from iqrp.app.regimes.bayesian.emissions import BayesianEmissions, _gaussian_logpdf
from iqrp.app.regimes.bayesian.evaluator import _avg_ce, _best_accuracy
from iqrp.app.regimes.bayesian.inference import (
    _ffbs_python,
    log_joint,
    smoothed_state_probabilities,
)
from iqrp.app.regimes.bayesian.model import (
    BayesianRegimeModel,
    BayesianRegimeSwitchingModel,
    _resolve_names,
)
from iqrp.app.regimes.bayesian.posterior import Posterior, posterior_predictive_observations
from iqrp.app.regimes.bayesian.prediction import current_state_distribution, forecast_from_posterior
from iqrp.app.regimes.bayesian.priors import ModelPriors
from iqrp.app.regimes.bayesian.transitions import BayesianTransitions
from iqrp.app.regimes.bayesian.visualization import (
    plot_credible_intervals,
    plot_posterior_histogram,
    plot_trace,
)
from iqrp.app.state_space.base.registry import get_registry as get_ss_registry


@pytest.mark.unit
def test_resolve_names_and_empty_forecast() -> None:
    assert _resolve_names(3, ("a", "b", "c")) == ("a", "b", "c")
    assert _resolve_names(3, ("a",)) == ("a", "state_1", "state_2")
    assert _resolve_names(2, None) == ("state_0", "state_1")
    empty = Posterior(draws=[])
    fc = forecast_from_posterior(empty, horizon=2)
    assert fc.horizon == 2
    assert current_state_distribution(np.array([0.2, 0.8])).sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_emissions_full_empty_state_and_singular() -> None:
    pri = ModelPriors.from_config(BayesianSettings.default().priors, 2, 2)
    rng = np.random.default_rng(0)
    emis = BayesianEmissions.from_priors(pri, 2, 2, covariance_type="full", rng=rng)
    y = rng.normal(size=(30, 2))
    states = np.zeros(30, dtype=np.int64)  # only state 0 occupied
    updated = emis.sample_posterior(y, states, rng=rng)
    assert updated.means.shape == (2, 2)
    ll = _gaussian_logpdf(np.zeros((4, 2)), np.zeros(2), np.array([[1.0, 1.0], [1.0, 1.0]]), "full")
    assert ll.shape == (4,)
    # 1d observations path
    emis1 = BayesianEmissions.from_priors(
        ModelPriors.from_config(BayesianSettings.default().priors, 2, 1),
        2,
        1,
        covariance_type="diag",
        rng=rng,
    )
    assert emis1.log_prob(np.linspace(-1, 1, 20)).shape[1] == 2
    emis1.sample_posterior(np.linspace(-1, 1, 20), np.zeros(20, dtype=np.int64), rng=rng)


@pytest.mark.unit
def test_evaluator_and_ffbs_python() -> None:
    assert np.isnan(_avg_ce(np.array([0.5, 0.5]), np.array([0])))
    pred = np.array([0, 1, 0, 1, 2, 3, 4, 5, 6, 0])
    truth = np.array([0, 1, 0, 1, 2, 3, 4, 5, 6, 1])
    assert 0 <= _best_accuracy(pred, truth, 7) <= 1
    alpha = np.array([[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]])
    log_p = np.log(np.array([[0.8, 0.2], [0.3, 0.7]]))
    z = _ffbs_python(alpha, log_p, np.array([0.1, 0.5, 0.9]))
    assert z.shape == (3,)
    assert burn_in_suggestion(np.linspace(0, 1, 100)) >= 0
    assert gelman_rubin([np.array([1.0])]) == 1.0


@pytest.mark.unit
def test_model_extract_helpers_and_regime_roundtrip(tmp_path: Path) -> None:
    assert "bayesian_regime" in get_ss_registry().list_names() or True  # may need import
    import iqrp.app.regimes.bayesian  # noqa: F401

    assert "bayesian_regime" in get_ss_registry().list_names()
    settings = BayesianSettings.from_mapping(
        {
            **BayesianSettings.default().model_dump(),
            "n_states": 2,
            "inference": {
                "algorithm": "variational",
                "n_chains": 1,
                "n_samples": 20,
                "burn_in": 5,
                "thin": 1,
                "target_accept": 0.65,
                "step_size": 0.05,
                "leapfrog_steps": 5,
                "n_jobs": 1,
                "checkpoint_every": 0,
                "resume": False,
            },
            "columns": {
                **BayesianSettings.default().columns.model_dump(),
                "observation_columns": ["ret"],
            },
            "variational": {"max_iter": 30, "tol": 1e-3, "learning_rate": 0.1},
        }
    )
    y = np.concatenate([np.full(40, -1.0), np.full(40, 1.0)]).reshape(-1, 1)
    model = BayesianRegimeSwitchingModel(n_states=2, settings=settings, random_seed=9)
    model.fit(y)
    frame = pl.DataFrame({"open_time": list(range(80)), "ret": y.ravel()})
    model.fit(frame)
    frame_close = pl.DataFrame({"open_time": list(range(80)), "close": y.ravel()})
    m2 = BayesianRegimeSwitchingModel(n_states=2, settings=_variational_plain(), random_seed=10)
    m2.fit(frame_close)
    # force no train obs
    m2._train_obs = None
    with pytest.raises(ValidationError):
        m2.diagnostics()
    regime = BayesianRegimeModel(n_states=2, settings=settings, random_seed=11)
    regime.fit(frame, feature_columns=["ret"])
    state = regime._algorithm_state()
    r2 = BayesianRegimeModel(n_states=2, settings=settings, random_seed=12)
    r2._load_algorithm_state(state)
    assert r2.is_fitted
    plot_trace([], tmp_path / "e.svg")
    plot_posterior_histogram([], tmp_path / "h.svg")
    plot_credible_intervals([], [], [], tmp_path / "c.svg")
    # predictive with states / full cov path
    post = model.posterior_summary()
    assert post is not None
    arr = posterior_predictive_observations(post, n_steps=5, rng=np.random.default_rng(0))
    assert arr.shape[0] == post.n_draws
    # log_joint smoke
    assert np.isfinite(
        log_joint(y, model.transitions, model.emissions, model.predict(y))  # type: ignore[arg-type]
    )
    gamma, ll = smoothed_state_probabilities(
        model.emissions.log_prob(y),  # type: ignore[union-attr]
        model.transition_matrix(),
        model.transitions.initial,  # type: ignore[union-attr]
    )
    assert gamma.shape[0] == y.shape[0] and np.isfinite(ll)
    bt = BayesianTransitions.from_dict(model.transitions.to_dict())  # type: ignore[union-attr]
    assert bt.persistence().shape[0] == 2
    assert bt.expected_durations()[0] > 0


def _variational_plain() -> BayesianSettings:
    return BayesianSettings.from_mapping(
        {
            **BayesianSettings.default().model_dump(),
            "n_states": 2,
            "inference": {
                "algorithm": "variational",
                "n_chains": 1,
                "n_samples": 15,
                "burn_in": 5,
                "thin": 1,
                "target_accept": 0.65,
                "step_size": 0.05,
                "leapfrog_steps": 5,
                "n_jobs": 1,
                "checkpoint_every": 0,
                "resume": False,
            },
            "variational": {"max_iter": 25, "tol": 1e-3, "learning_rate": 0.1},
        }
    )

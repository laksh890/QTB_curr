"""Coverage and edge-case tests for Bayesian engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from iqrp.app.core.exceptions import ConfigurationError, ValidationError
from iqrp.app.regimes.bayesian import BayesianRegimeSwitchingModel, BayesianSettings
from iqrp.app.regimes.bayesian.posterior import ParameterDraw, Posterior
from iqrp.app.regimes.bayesian.priors import UserDefinedPrior, mvn_prior, sample_wishart
from iqrp.app.regimes.bayesian.serializer import _json_default
from iqrp.app.regimes.bayesian.trainer import (
    bayes_factor,
    loo_cv,
    marginal_likelihood_harmonic,
    waic,
)
from iqrp.app.regimes.bayesian.visualization import (
    plot_posterior_predictive_check,
    plot_transition_uncertainty,
)


@pytest.mark.unit
def test_config_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigurationError):
        BayesianSettings.from_mapping([1, 2])  # type: ignore[arg-type]
    assert BayesianSettings.from_mapping(OmegaConf.create({"n_states": 2})).n_states == 2
    bad = tmp_path / "bad.yaml"
    bad.write_text("- a\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        BayesianSettings.from_hydra(bad)
    monkeypatch.setattr(
        "iqrp.app.regimes.bayesian.config._default_config_path",
        lambda: tmp_path / "missing.yaml",
    )
    assert BayesianSettings.default().n_states == 3


@pytest.mark.unit
def test_serializer_and_not_fitted() -> None:
    assert _json_default(np.array([1.0])) == [1.0]
    assert _json_default(np.float64(1.2)) == 1.2
    with pytest.raises(TypeError):
        _json_default(object())
    model = BayesianRegimeSwitchingModel(n_states=2)
    with pytest.raises(ValidationError):
        model.predict(np.zeros((5, 1)))
    with pytest.raises(ValidationError):
        model.fit(pl.DataFrame({"open_time": [0, 1], "symbol": ["a", "b"]}))
    user = UserDefinedPrior(lambda x: 0.0)
    with pytest.raises(ValueError):
        user.rvs()


@pytest.mark.unit
def test_partial_fit_checkpoint_compare(tmp_path: Path) -> None:
    y = np.concatenate([np.full(50, -1.0), np.full(50, 1.0)]).reshape(-1, 1)
    settings = BayesianSettings.from_mapping(
        {
            **BayesianSettings.default().model_dump(),
            "n_states": 2,
            "inference": {
                "algorithm": "gibbs",
                "n_chains": 1,
                "n_samples": 25,
                "burn_in": 5,
                "thin": 1,
                "target_accept": 0.65,
                "step_size": 0.05,
                "leapfrog_steps": 5,
                "n_jobs": 1,
                "checkpoint_every": 10,
                "resume": False,
            },
            "online": {"window_size": 40, "update_frequency": 2, "warm_start": True},
            "model_comparison": {"min_states": 2, "max_states": 2, "criterion": "loo"},
            "store_dir": str(tmp_path / "store"),
        }
    )
    model = BayesianRegimeSwitchingModel(n_states=2, settings=settings, random_seed=4)
    model.fit(y)
    model.partial_fit(y[:20])  # skip (freq=2 -> counter=1)
    model.partial_fit(y[20:40])  # update
    settings2 = BayesianSettings.from_mapping(
        {
            **settings.model_dump(),
            "online": {"window_size": 0, "update_frequency": 1, "warm_start": False},
        }
    )
    m2 = BayesianRegimeSwitchingModel(n_states=2, settings=settings2, random_seed=5)
    m2.fit(y)
    m2.partial_fit(y[:30])
    cmp_ = model.compare_models(y)
    assert cmp_["best_n_states"] >= 2
    assert waic(np.array([-1.0, -1.2]))["waic"] < 0
    assert loo_cv(np.array([-1.0, -1.2, -0.8]))["loo"] != 0
    assert np.isfinite(marginal_likelihood_harmonic(model.posterior_summary()))  # type: ignore[arg-type]
    assert bayes_factor(-10.0, -12.0) > 1.0
    # empty posterior helpers
    empty = Posterior(draws=[])
    assert empty.state_occupancy().sum() > 0
    assert empty.posterior_state_probabilities(5).shape[0] == 5
    draw = ParameterDraw(
        transition=np.eye(2),
        initial=np.array([0.5, 0.5]),
        means=np.zeros((2, 1)),
        covars=np.ones((2, 1)),
        states=None,
    )
    p1 = Posterior(draws=[draw])
    assert p1.credible_intervals("initial")["mean"].shape[0] == 2
    assert p1.credible_intervals("covars")["mean"].shape[0] == 2
    assert p1.marginal_summary("means")["mean"].shape[0] == 2
    rng = np.random.default_rng(0)
    assert sample_wishart(3, np.eye(2), rng).shape == (2, 2)
    assert mvn_prior([0, 0], np.eye(2)).rvs(rng=rng).shape


@pytest.mark.unit
def test_viz_disabled_and_ppc(tmp_path: Path) -> None:
    settings = BayesianSettings.from_mapping(
        {
            **BayesianSettings.default().model_dump(),
            "visualization": {"enabled": False, "max_points": 5},
        }
    )
    plot_transition_uncertainty(
        np.eye(2), np.zeros((2, 2)), np.ones((2, 2)), tmp_path / "tu.svg", settings
    )
    y = np.linspace(-1, 1, 40)
    pred = np.random.default_rng(0).normal(size=(10, 40))
    settings_on = BayesianSettings.default()
    plot_posterior_predictive_check(y, pred, tmp_path / "ppc.svg", settings_on)
    assert (tmp_path / "ppc.svg").exists()

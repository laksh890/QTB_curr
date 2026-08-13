"""Final coverage push for Bayesian engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.regimes.bayesian.config import BayesianSettings
from iqrp.app.regimes.bayesian.convergence import burn_in_suggestion, gelman_rubin
from iqrp.app.regimes.bayesian.emissions import BayesianEmissions, _gaussian_logpdf
from iqrp.app.regimes.bayesian.hmc import _mean_grad
from iqrp.app.regimes.bayesian.inference import log_joint
from iqrp.app.regimes.bayesian.model import BayesianRegimeSwitchingModel
from iqrp.app.regimes.bayesian.posterior import (
    ParameterDraw,
    Posterior,
    posterior_predictive_observations,
)
from iqrp.app.regimes.bayesian.prediction import forecast_from_posterior
from iqrp.app.regimes.bayesian.priors import ModelPriors
from iqrp.app.regimes.bayesian.trainer import BayesianTrainer, loo_cv, pointwise_log_likelihood
from iqrp.app.regimes.bayesian.transitions import BayesianTransitions
from iqrp.app.regimes.bayesian.visualization import (
    plot_credible_intervals,
    plot_posterior_histogram,
    plot_posterior_predictive_check,
    plot_regime_timeline,
    plot_trace,
    plot_transition_uncertainty,
)


@pytest.mark.unit
def test_final_gaps(tmp_path: Path) -> None:
    # gelman short chain length mismatch path + burn_in small window path
    assert gelman_rubin([np.array([1.0]), np.array([1.0, 2.0])]) == 1.0
    assert burn_in_suggestion(np.linspace(0, 1, 200), window=20) >= 0

    # LinAlgError recovery in gaussian logpdf
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert _gaussian_logpdf(np.zeros((2, 2)), np.zeros(2), singular, "full").shape == (2,)

    # HMC grad with empty mask and full cov
    y = np.random.default_rng(0).normal(size=(20, 2))
    states = np.zeros(20, dtype=np.int64)
    means = np.zeros((2, 2))
    cov = np.array([np.eye(2), np.eye(2)])
    g = _mean_grad(y, states, means, cov, "full")
    assert g.shape == (2, 2)
    g2 = _mean_grad(y, states, means, np.ones((2, 2)), "diag")
    assert g2.shape == (2, 2)

    settings = BayesianSettings.from_mapping(
        {
            **BayesianSettings.default().model_dump(),
            "n_states": 2,
            "inference": {
                "algorithm": "gibbs",
                "n_chains": 1,
                "n_samples": 15,
                "burn_in": 3,
                "thin": 1,
                "target_accept": 0.65,
                "step_size": 0.05,
                "leapfrog_steps": 5,
                "n_jobs": 1,
                "checkpoint_every": 0,
                "resume": False,
            },
            "emission": {"type": "multivariate_gaussian", "covariance_type": "full"},
            "online": {"window_size": 30, "update_frequency": 3, "warm_start": True},
        }
    )
    y1 = np.concatenate([np.full(40, -1.0), np.full(40, 1.0)]).reshape(-1, 1)
    # partial_fit before fit
    m0 = BayesianRegimeSwitchingModel(n_states=2, settings=settings, random_seed=1)
    m0.partial_fit(y1)
    assert m0.is_fitted
    # predict_proba fallback when draw lengths mismatch
    m0._posterior.draws[0].states = np.array([0, 1])  # type: ignore[union-attr]
    proba = m0.predict_proba(y1)
    assert proba.shape[0] == y1.shape[0]
    # sample with full covariance emissions
    m0.emissions.covariance_type = "full"  # type: ignore[union-attr]
    m0.emissions.n_features = 1  # type: ignore[union-attr]
    m0.emissions.means = np.array([[-1.0], [1.0]])  # type: ignore[union-attr]
    m0.emissions.covars = np.array([[[0.2]], [[0.2]]])  # type: ignore[union-attr]
    states, obs = m0.sample(10)
    assert obs.shape == (10, 1)
    # diagnostics with explicit observations None uses train obs
    assert m0.diagnostics()["n_draws"] > 0

    # forecast with states present
    d = ParameterDraw(
        transition=np.array([[0.9, 0.1], [0.1, 0.9]]),
        initial=np.array([0.5, 0.5]),
        means=np.array([[0.0], [1.0]]),
        covars=np.ones((2, 1)),
        states=np.array([0, 1, 0, 1]),
    )
    fc = forecast_from_posterior(Posterior(draws=[d]), horizon=2)
    assert fc.horizon == 2
    assert posterior_predictive_observations(Posterior(draws=[]), n_steps=5).shape[0] == 0

    # trainer checkpoint path + loo short + pointwise empty
    s2 = BayesianSettings.from_mapping(
        {
            **settings.model_dump(),
            "inference": {
                **settings.inference.model_dump(),
                "algorithm": "hmc",
                "checkpoint_every": 5,
                "n_samples": 10,
                "burn_in": 2,
            },
            "store_dir": str(tmp_path / "store"),
        }
    )
    BayesianTrainer(s2).fit(y1, n_states=2, rng=np.random.default_rng(0))
    assert loo_cv(np.array([-1.0]))["loo"] == -1.0
    assert pointwise_log_likelihood(y1, Posterior(draws=[])).shape[0] == y1.shape[0]

    # log_joint 1d obs
    pri = ModelPriors.from_config(BayesianSettings.default().priors, 2, 1)
    trans = BayesianTransitions.from_priors(pri, rng=np.random.default_rng(0))
    emis = BayesianEmissions.from_priors(pri, 2, 1, rng=np.random.default_rng(0))
    assert np.isfinite(log_joint(y1.reshape(-1), trans, emis, states))

    # visualization enabled empty / 1d branches
    plot_trace([], tmp_path / "t.svg")
    plot_posterior_histogram([], tmp_path / "h.svg")
    plot_credible_intervals([], [], [], tmp_path / "c.svg")
    plot_regime_timeline(np.linspace(0, 1, 10), tmp_path / "r.svg")
    plot_posterior_predictive_check(
        np.linspace(0, 1, 20), np.linspace(0, 1, 20), tmp_path / "p.svg"
    )
    plot_posterior_predictive_check(
        np.linspace(0, 1, 20),
        np.random.default_rng(0).normal(size=(5, 1, 20)),
        tmp_path / "p2.svg",
    )
    off = BayesianSettings.from_mapping(
        {
            **BayesianSettings.default().model_dump(),
            "visualization": {"enabled": False, "max_points": 3},
        }
    )
    plot_transition_uncertainty(
        np.eye(2), np.zeros((2, 2)), np.ones((2, 2)), tmp_path / "tu.svg", off
    )
    # frame close fallback already covered; ensure numeric column auto-detect
    frame = pl.DataFrame({"open_time": list(range(80)), "feat": y1.ravel()})
    m1 = BayesianRegimeSwitchingModel(
        n_states=2,
        settings=BayesianSettings.from_mapping(
            {
                **BayesianSettings.default().model_dump(),
                "n_states": 2,
                "inference": {
                    "algorithm": "variational",
                    "n_chains": 1,
                    "n_samples": 10,
                    "burn_in": 2,
                    "thin": 1,
                    "target_accept": 0.65,
                    "step_size": 0.05,
                    "leapfrog_steps": 5,
                    "n_jobs": 1,
                    "checkpoint_every": 0,
                    "resume": False,
                },
                "variational": {"max_iter": 15, "tol": 1e-3, "learning_rate": 0.1},
            }
        ),
        random_seed=4,
    )
    m1.fit(frame)
    assert m1.is_fitted

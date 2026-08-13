"""Core unit tests for Bayesian Regime Switching Engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from iqrp.app.regimes.bayesian import (
    BayesianRegimeModel,
    BayesianRegimeSwitchingModel,
    BayesianSettings,
    ModelPriors,
)
from iqrp.app.regimes.bayesian.convergence import (
    autocorrelation,
    effective_sample_size,
    gelman_rubin,
)
from iqrp.app.regimes.bayesian.inference import ffbs, forward_filter
from iqrp.app.regimes.bayesian.priors import (
    UserDefinedPrior,
    beta_prior,
    dirichlet_prior,
    gamma_prior,
    inverse_gamma_prior,
    normal_prior,
    wishart_prior,
)
from iqrp.app.regimes.bayesian.visualization import (
    plot_credible_intervals,
    plot_posterior_histogram,
    plot_regime_timeline,
    plot_trace,
)


def _fast_settings(**overrides: object) -> BayesianSettings:
    base = {
        **BayesianSettings.default().model_dump(),
        "n_states": 2,
        "inference": {
            "algorithm": "gibbs",
            "n_chains": 2,
            "n_samples": 40,
            "burn_in": 10,
            "thin": 1,
            "target_accept": 0.65,
            "step_size": 0.05,
            "leapfrog_steps": 5,
            "n_jobs": 2,
            "checkpoint_every": 0,
            "resume": False,
        },
        "model_comparison": {"min_states": 2, "max_states": 2, "criterion": "waic"},
    }
    base.update(overrides)
    return BayesianSettings.from_mapping(base)


@pytest.mark.unit
def test_priors_and_convergence_helpers() -> None:
    rng = np.random.default_rng(0)
    assert beta_prior(2, 2).rvs(rng=rng).shape
    assert gamma_prior(2, 1).rvs(rng=rng).shape
    assert normal_prior(0, 1).rvs(rng=rng).shape
    assert dirichlet_prior([1, 1, 1]).rvs(rng=rng).shape
    ig = inverse_gamma_prior(3, 1)
    assert float(np.asarray(ig.rvs(rng=rng)).reshape(-1)[0]) > 0
    w = wishart_prior(4, np.eye(2))
    assert w.rvs(rng=rng).shape == (2, 2)
    user = UserDefinedPrior(
        lambda x: -0.5 * float(np.sum(np.asarray(x) ** 2)), lambda r: r.normal(size=2)
    )
    assert np.isfinite(user.logpdf(np.zeros(2)))
    assert user.rvs(rng).shape == (2,)
    chains = [rng.normal(size=50), rng.normal(size=50)]
    assert gelman_rubin(chains) >= 0.9
    assert effective_sample_size(chains[0]) >= 1
    assert autocorrelation(chains[0], max_lag=5).shape[0] == 5


@pytest.mark.unit
def test_ffbs_and_gibbs_fit(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(-1.5, 0.3, 80), rng.normal(1.5, 0.3, 80)]).reshape(-1, 1)
    settings = _fast_settings()
    model = BayesianRegimeSwitchingModel(n_states=2, settings=settings, random_seed=1)
    model.fit(y)
    assert model.is_fitted
    pred = model.predict(y)
    proba = model.predict_proba(y)
    assert pred.shape[0] == y.shape[0]
    assert proba.shape == (y.shape[0], 2)
    assert np.isfinite(model.log_likelihood(y))
    filt = model.filter(y)
    assert filt.n_states == 2
    sm = model.smooth(y)
    assert sm.smoothed_states.shape[0] == y.shape[0]
    fc = model.forecast(y, horizon=3)
    assert fc.horizon == 3
    ci = model.credible_intervals("means")
    assert "low" in ci and "high" in ci
    draws = model.sample_posterior(5)
    assert len(draws) == 5
    y_rep = model.posterior_predictive(n_steps=20)
    assert y_rep.ndim == 3
    states, obs = model.sample(30)
    assert states.shape == (30,) and obs.shape[0] == 30
    path = model.save(tmp_path / "bayes.json")
    loaded = BayesianRegimeSwitchingModel.load(path)
    assert loaded.is_fitted
    diag = model.diagnostics(y)
    assert "convergence" in diag
    plot_trace(model._history, tmp_path / "trace.svg", settings)
    plot_posterior_histogram([d["means"][0][0] for d in draws], tmp_path / "hist.svg", settings)
    plot_regime_timeline(proba, tmp_path / "tl.svg", settings)
    plot_credible_intervals(ci["mean"], ci["low"], ci["high"], tmp_path / "ci.svg", settings)
    # FFBS smoke
    log_e = model.emissions.log_prob(y)  # type: ignore[union-attr]
    alpha, ll = forward_filter(log_e, model.transition_matrix(), model.transitions.initial)  # type: ignore[union-attr]
    z, _ = ffbs(log_e, model.transition_matrix(), model.transitions.initial, rng=rng)  # type: ignore[union-attr]
    assert alpha.shape[0] == y.shape[0] and z.shape[0] == y.shape[0] and np.isfinite(ll)


@pytest.mark.unit
def test_algorithms_and_regime_adapter() -> None:
    rng = np.random.default_rng(2)
    y = np.concatenate([rng.normal(-1.0, 0.4, 60), rng.normal(1.0, 0.4, 60)]).reshape(-1, 1)
    for algo in ("metropolis", "hmc", "variational"):
        settings = _fast_settings(
            inference={
                **_fast_settings().inference.model_dump(),
                "algorithm": algo,
                "n_chains": 1,
                "n_samples": 30,
                "burn_in": 8,
                "n_jobs": 1,
            }
        )
        model = BayesianRegimeSwitchingModel(n_states=2, settings=settings, random_seed=2)
        model.fit(y)
        assert model.posterior_summary() is not None and model.posterior_summary().n_draws > 0

    frame = pl.DataFrame({"ret": y.ravel()})
    regime = BayesianRegimeModel(n_states=2, settings=_fast_settings(), random_seed=3)
    regime.fit(frame, feature_columns=["ret"])
    assert regime.predict(frame, feature_columns=["ret"]).shape[0] == frame.height
    assert regime.predict_proba(frame, feature_columns=["ret"]).shape[1] == 2
    fc = regime.forecast(frame, steps=2)
    assert fc.steps == 2
    priors = ModelPriors.from_config(BayesianSettings.default().priors, 2, 1)
    assert priors.to_dict()["mean_strength"] > 0

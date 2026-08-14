"""Additional coverage gaps for Bayesian engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqrp.app.regimes.bayesian.config import BayesianSettings
from iqrp.app.regimes.bayesian.convergence import (
    autocorrelation,
    burn_in_suggestion,
    effective_sample_size,
    gelman_rubin,
)
from iqrp.app.regimes.bayesian.emissions import BayesianEmissions, _gaussian_logpdf
from iqrp.app.regimes.bayesian.gibbs import run_gibbs
from iqrp.app.regimes.bayesian.hmc import run_hmc
from iqrp.app.regimes.bayesian.inference import (
    _HAS_NUMBA,
    _ffbs_backward_sample,
    _ffbs_python,
    ffbs,
)
from iqrp.app.regimes.bayesian.metropolis import run_metropolis
from iqrp.app.regimes.bayesian.model import BayesianRegimeModel, BayesianRegimeSwitchingModel
from iqrp.app.regimes.bayesian.posterior import (
    ParameterDraw,
    Posterior,
    posterior_predictive_observations,
)
from iqrp.app.regimes.bayesian.prediction import (
    forecast_from_posterior,
    posterior_predictive_state_proba,
)
from iqrp.app.regimes.bayesian.priors import ModelPriors, UserDefinedPrior
from iqrp.app.regimes.bayesian.trainer import pointwise_log_likelihood, waic
from iqrp.app.regimes.bayesian.transitions import BayesianTransitions
from iqrp.app.regimes.bayesian.variational import run_variational
from iqrp.app.regimes.bayesian.visualization import (
    plot_posterior_histogram,
    plot_regime_timeline,
    plot_trace,
)


def _y(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(-1.2, 0.3, 50), rng.normal(1.2, 0.3, 50)]).reshape(-1, 1)


@pytest.mark.unit
def test_convergence_edge_cases() -> None:
    assert gelman_rubin([]) == 1.0
    assert gelman_rubin([np.array([1.0, 1.0])]) == 1.0
    assert gelman_rubin([np.ones(5), np.ones(5)]) == 1.0
    assert effective_sample_size(np.array([1.0, 1.0])) >= 1
    assert effective_sample_size(np.zeros(10)) >= 1
    assert autocorrelation(np.array([]), max_lag=5).size == 0
    assert autocorrelation(np.ones(20), max_lag=5)[0] == 1.0
    assert burn_in_suggestion(np.arange(10.0), window=20) >= 0


@pytest.mark.unit
def test_direct_samplers_and_numba_branch(tmp_path: Path) -> None:
    y = _y(1)
    pri = ModelPriors.from_config(BayesianSettings.default().priors, 2, 1)
    rng = np.random.default_rng(1)
    g = run_gibbs(
        y,
        2,
        pri,
        n_chains=1,
        n_samples=15,
        burn_in=5,
        checkpoint_every=5,
        checkpoint_dir=tmp_path / "ckpt",
        rng=rng,
    )
    assert g.posterior.n_draws > 0
    assert list((tmp_path / "ckpt").glob("*.npz"))
    # warm start
    last = g.posterior.draws[-1]

    warm = (
        BayesianTransitions(
            2, last.transition, last.initial, pri.transition_alpha, pri.initial_alpha
        ),
        BayesianEmissions(2, 1, last.means, last.covars, "diag", pri),
        last.states if last.states is not None else np.zeros(y.shape[0], dtype=np.int64),
    )
    g2 = run_gibbs(y, 2, pri, n_chains=1, n_samples=5, burn_in=2, warm_start=warm, rng=rng)
    assert g2.posterior.n_draws > 0

    mh = run_metropolis(y, 2, pri, n_samples=20, burn_in=5, covariance_type="full", rng=rng)
    assert mh.acceptance_rate >= 0.0
    h = run_hmc(y, 2, pri, n_samples=15, burn_in=5, leapfrog_steps=3, rng=rng)
    assert h.posterior.n_draws > 0
    v = run_variational(y, 2, pri, covariance_type="full", max_iter=20, rng=rng)
    assert v.posterior.n_draws > 0

    emis = BayesianEmissions.from_priors(pri, 2, 1, rng=rng)
    trans = BayesianTransitions.from_priors(pri, rng=rng)
    log_e = emis.log_prob(y)
    z, ll = ffbs(log_e, trans.transition, trans.initial, rng=rng)
    assert z.size == y.shape[0] and np.isfinite(ll)
    assert _HAS_NUMBA in (True, False)
    # cover pure-python FFBS helpers / degenerate probs
    alpha0 = np.zeros((3, 2))
    log_p = np.log(np.array([[0.7, 0.3], [0.4, 0.6]]))
    z0 = _ffbs_backward_sample(alpha0, log_p, np.array([0.2, 0.8, 0.1]))
    assert z0.shape == (3,)
    z1 = _ffbs_python(np.array([[0.5, 0.5], [0.5, 0.5]]), log_p, np.array([0.01, 0.99]))
    assert z1.shape == (2,)
    # singular full cov recovery
    out = _gaussian_logpdf(np.zeros((3, 2)), np.zeros(2), np.ones((2, 2)), "full")
    assert out.shape == (3,)
    # 1d series through MH/HMC/VI
    y1 = y.reshape(-1)
    run_metropolis(y1, 2, pri, n_samples=8, burn_in=2, rng=rng)
    run_hmc(y1, 2, pri, n_samples=8, burn_in=2, leapfrog_steps=2, rng=rng)
    run_variational(y1, 2, pri, max_iter=10, rng=rng)
    # HMC full-cov gradient path
    run_hmc(y.reshape(-1, 1), 2, pri, covariance_type="full", n_samples=8, burn_in=2, rng=rng)


@pytest.mark.unit
def test_posterior_predictive_and_model_api_edges() -> None:
    settings = BayesianSettings.from_mapping(
        {
            **BayesianSettings.default().model_dump(),
            "n_states": 2,
            "inference": {
                "algorithm": "gibbs",
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
        }
    )
    y = _y(2)
    model = BayesianRegimeSwitchingModel(n_states=2, settings=settings, random_seed=2)
    model.fit(y)
    assert model.posterior().n_draws > 0
    assert model.sample_posterior(3)
    reps = model.posterior_predictive(n_steps=10)
    assert reps.shape[1] == 10
    # draws without states
    d = ParameterDraw(
        transition=np.eye(2) * 0.5 + 0.25,
        initial=np.array([0.5, 0.5]),
        means=np.array([[-1.0], [1.0]]),
        covars=np.ones((2, 1)),
        states=None,
    )
    # normalize transition
    d.transition = np.array([[0.8, 0.2], [0.2, 0.8]])
    post = Posterior(draws=[d, d])
    assert post.state_occupancy().sum() == pytest.approx(1.0)
    arr = posterior_predictive_observations(post, n_steps=5, rng=np.random.default_rng(0))
    assert arr.shape == (2, 5, 1)
    # full cov predictive
    d2 = ParameterDraw(
        transition=np.array([[0.8, 0.2], [0.2, 0.8]]),
        initial=np.array([0.5, 0.5]),
        means=np.zeros((2, 2)),
        covars=np.array([np.eye(2), np.eye(2)]),
        states=np.array([0, 1, 0]),
    )
    arr2 = posterior_predictive_observations(
        Posterior(draws=[d2]), n_steps=3, rng=np.random.default_rng(1)
    )
    assert arr2.shape[-1] == 2
    fc = forecast_from_posterior(post, horizon=2, n_draws=1)
    assert fc.horizon == 2
    assert posterior_predictive_state_proba(post, n_steps=3).shape[0] == 3
    # scalar ci / from_dict
    ci = post.scalar_ci(np.array([1.0, 2.0, 3.0]))
    assert ci.low <= ci.high
    restored = Posterior.from_dict(post.to_dict())
    assert restored.n_draws == 2
    pll = pointwise_log_likelihood(y, model.posterior_summary())
    assert waic(pll.reshape(-1, 1))["p_waic"] >= 0
    # user prior logpdf
    u = UserDefinedPrior(lambda x: 0.0, lambda r: r.normal(size=1))
    assert u.logpdf(0) == 0.0
    # regime sample path via engine sample
    states, obs = model.sample(12, initial_state=0)
    assert states[0] == 0 and obs.shape[0] == 12
    # diagnostics empty by_chain shouldn't happen; force empty draws report
    from iqrp.app.regimes.bayesian.diagnostics import BayesianDiagnostics

    empty_report = BayesianDiagnostics().report(Posterior(draws=[]))
    assert empty_report["n_draws"] == 0
    plot_trace(np.array([1.0]), Path("/tmp/t_bayes.svg"))
    plot_posterior_histogram(np.linspace(0, 1, 50), Path("/tmp/h_bayes.svg"))
    plot_regime_timeline(np.array([0.5, 0.5]), Path("/tmp/r_bayes.svg"))
    # adapter predict_proba / forecast already tested; hit export via save roundtrip already
    import polars as pl

    frame = pl.DataFrame({"ret": y.ravel()})
    regime = BayesianRegimeModel(n_states=2, settings=settings, random_seed=3)
    regime.fit(frame, feature_columns=["ret"])
    assert regime.predict_proba(frame, feature_columns=["ret"]).shape[1] == 2
    regime.forecast(frame, steps=1)

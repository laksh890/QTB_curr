"""Synthetic recovery, stability, and stress tests."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.regimes.hmm import HiddenMarkovModel, baum_welch, forward
from iqrp.app.regimes.hmm.initialization import initialize_parameters
from iqrp.app.simulation.regimes.hidden_regime import HiddenRegimeSimulator
from iqrp.app.simulation.regimes.regime_switching import RegimeSwitchingSimulator


@pytest.mark.unit
def test_recover_hidden_regimes() -> None:
    k = 2
    true_p = RegimeSwitchingSimulator.mixed_transition(k, 0.93)
    obs = HiddenRegimeSimulator(np.random.default_rng(11)).simulate(
        1500,
        transition_matrix=true_p,
        state_names=("a", "b"),
        emission_means=(-2.0, 2.0),
        emission_stds=(0.35, 0.35),
    )
    y = obs.observations.reshape(-1, 1)
    model = HiddenMarkovModel(n_states=2, random_seed=11)
    settings = model._hmm_settings
    from iqrp.app.regimes.hmm.config import HMMSettings

    model = HiddenMarkovModel(
        n_states=2,
        random_seed=11,
        settings=HMMSettings.from_mapping(
            {
                **settings.model_dump(),
                "initialization": {"method": "kmeans", "n_restarts": 3, "dirichlet_alpha": 1.0},
                "training": {
                    "max_iter": 80,
                    "tol": 1e-5,
                    "early_stopping": True,
                    "min_covar": 1e-6,
                    "n_jobs": 2,
                },
            }
        ),
    )
    model.fit(y)
    report = model.evaluate(y, true_states=obs.latent.state_ids)
    assert report["metrics"]["prediction_accuracy"] >= 0.85
    est = model.transition_matrix()
    # diagonals (persistence) recovered approximately
    assert abs(float(np.diag(est).mean()) - float(np.diag(true_p).mean())) < 0.12
    assert model.diagnostics(y)["convergence"]["history"]
    assert len(model._history) >= 2
    # likelihood should improve over first iterations
    assert model._history[-1] >= model._history[0] - 1e-6


@pytest.mark.unit
def test_numerical_stability_extreme() -> None:
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(-50, 0.01, 80), rng.normal(50, 0.01, 80)]).reshape(-1, 1)
    trans, emis = initialize_parameters(y, 2, method="kmeans", rng=rng)
    log_e = emis.log_prob(y)
    assert np.all(np.isfinite(log_e))
    alpha, _scales, ll = forward(log_e, trans.transition, initial=trans.initial)
    assert np.all(np.isfinite(alpha)) and np.isfinite(ll)
    result = baum_welch(y, 2, max_iter=30, n_restarts=1, rng=rng)
    assert np.isfinite(result.log_likelihood)


@pytest.mark.unit
def test_stress_large_series() -> None:
    rng = np.random.default_rng(3)
    n = 200_000
    # two well-separated Gaussians with sticky transitions
    states = np.empty(n, dtype=np.int64)
    states[0] = 0
    p = np.array([[0.98, 0.02], [0.03, 0.97]])
    u = rng.random(n - 1)
    for t in range(n - 1):
        states[t + 1] = 0 if u[t] < p[states[t], 0] else 1
    y = np.where(states == 0, rng.normal(-1.0, 0.2, n), rng.normal(1.0, 0.2, n)).reshape(-1, 1)
    from iqrp.app.regimes.hmm.config import HMMSettings

    model = HiddenMarkovModel(
        n_states=2,
        settings=HMMSettings.from_mapping(
            {
                **HMMSettings.default().model_dump(),
                "initialization": {"method": "kmeans", "n_restarts": 1, "dirichlet_alpha": 1.0},
                "training": {
                    "max_iter": 15,
                    "tol": 1e-3,
                    "early_stopping": True,
                    "min_covar": 1e-6,
                    "n_jobs": 1,
                },
            }
        ),
        random_seed=3,
    )
    model.fit(y)
    assert model.predict(y[:1000]).shape == (1000,)

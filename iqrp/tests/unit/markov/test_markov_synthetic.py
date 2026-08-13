"""Synthetic regime recovery and stress tests using the Market Simulation Engine."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.math.stochastic.markov_utils import stationary_distribution
from iqrp.app.regimes.markov import MarkovChainModel, MarkovSettings
from iqrp.app.simulation.regimes.hidden_regime import HiddenRegimeSimulator
from iqrp.app.simulation.regimes.regime_switching import RegimeSwitchingSimulator


@pytest.mark.unit
def test_recover_transition_persistence_stationary() -> None:
    k = 3
    true_p = RegimeSwitchingSimulator.mixed_transition(k, persistence=0.92)
    true_pi = stationary_distribution(true_p)
    sim = RegimeSwitchingSimulator(np.random.default_rng(42))
    path = sim.simulate(
        5000,
        transition_matrix=true_p,
        state_names=("s0", "s1", "s2"),
        drifts=(0.1, 0.0, -0.1),
        volatilities=(0.2, 0.15, 0.3),
        initial_state=0,
    )
    model = MarkovChainModel(n_states=k, state_names=("s0", "s1", "s2"), random_seed=42)
    model.fit(path.state_ids)
    est = model.transition_matrix()
    # Frobenius relative error on transitions
    err = np.linalg.norm(est - true_p) / np.linalg.norm(true_p)
    assert err < 0.15
    # Persistence (diagonal)
    assert np.allclose(np.diag(est), np.diag(true_p), atol=0.08)
    # Stationary
    est_pi = model.stationary_distribution()
    assert np.allclose(est_pi, true_pi, atol=0.08)
    # Forecast: most likely next often matches persistence
    fc = model.forecast(path.state_ids[-50:], horizon=1)
    assert fc.probability_distribution.sum() == pytest.approx(1.0)


@pytest.mark.unit
def test_hidden_regime_labels_via_latent() -> None:
    """Use HiddenRegimeSimulator latent states as discrete observations."""
    k = 2
    true_p = RegimeSwitchingSimulator.mixed_transition(k, 0.9)
    obs = HiddenRegimeSimulator(np.random.default_rng(7)).simulate(
        2000,
        transition_matrix=true_p,
        state_names=("a", "b"),
        emission_means=(-1.0, 1.0),
        emission_stds=(0.25, 0.25),
    )
    model = MarkovChainModel(n_states=k)
    model.fit(obs.latent.state_ids)
    assert np.linalg.norm(model.transition_matrix() - true_p) / np.linalg.norm(true_p) < 0.2
    pred = model.predict(obs.latent.state_ids)
    assert float(np.mean(pred == obs.latent.state_ids)) == pytest.approx(1.0)
    report = model.evaluate(obs.latent.state_ids, true_states=obs.latent.state_ids)
    assert report["metrics"]["prediction_accuracy"] == pytest.approx(1.0)


@pytest.mark.unit
def test_stress_million_scale_vectorized() -> None:
    """Stress: ~1e6 observations should fit quickly with vectorized counts."""
    k = 5
    true_p = RegimeSwitchingSimulator.mixed_transition(k, 0.85)
    rng = np.random.default_rng(1)
    n = 1_000_000
    # Fast path simulation
    states = np.empty(n, dtype=np.int64)
    states[0] = 0
    # Use cumulative choice via searchsorted for speed
    cdf = np.cumsum(true_p, axis=1)
    u = rng.random(n - 1)
    for t in range(n - 1):
        states[t + 1] = int(np.searchsorted(cdf[states[t]], u[t], side="right"))
        if states[t + 1] >= k:
            states[t + 1] = k - 1
    settings = MarkovSettings.from_mapping(
        {
            **MarkovSettings.default().model_dump(),
            "n_states": k,
            "estimation": {
                "method": "mle",
                "laplace_alpha": 0.0,
                "dirichlet_alpha": 1.0,
                "forgetting_factor": 1.0,
                "window_size": 0,
                "min_count_warning": 10,
            },
        }
    )
    model = MarkovChainModel(n_states=k, settings=settings)
    model.fit(states)
    assert model.estimator.matrix.n_transitions == n - 1
    assert np.allclose(model.transition_matrix().sum(axis=1), 1.0)


@pytest.mark.unit
def test_many_states_sparse_path() -> None:
    k = 20
    p = RegimeSwitchingSimulator.mixed_transition(k, 0.9)
    path = RegimeSwitchingSimulator(np.random.default_rng(3)).simulate(
        3000,
        transition_matrix=p,
        state_names=tuple(f"s{i}" for i in range(k)),
        drifts=[0.0] * k,
        volatilities=[0.2] * k,
    )
    model = MarkovChainModel(n_states=k)
    model.fit(path.state_ids)
    sp = model.estimator.matrix.sparse_probability_matrix()
    assert sp.shape == (k, k)
    assert model.stationary_distribution().shape == (k,)

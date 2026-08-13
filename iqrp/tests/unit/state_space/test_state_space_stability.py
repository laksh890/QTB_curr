"""Synthetic latent-state and numerical stability tests."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.math.utils.numerical_stability import logsumexp
from iqrp.app.state_space.base.probabilities import (
    backward_probabilities,
    forward_probabilities,
    state_occupancy_probabilities,
)
from iqrp.app.state_space.filtering.forward_filter import ForwardFilter
from iqrp.app.state_space.models.mock import MockDiscreteStateSpaceModel


@pytest.mark.unit
def test_synthetic_latent_recovery() -> None:
    rng = np.random.default_rng(11)
    p = np.array([[0.95, 0.05], [0.05, 0.95]])
    states = [0]
    for _ in range(199):
        states.append(int(rng.choice(2, p=p[states[-1]])))
    states_arr = np.asarray(states, dtype=np.int64)
    y = np.where(states_arr == 0, rng.normal(-2.0, 0.2, 200), rng.normal(2.0, 0.2, 200))

    model = MockDiscreteStateSpaceModel(n_states=2, random_seed=11)
    model.fit(y)
    pred = model.predict(y)
    # Allow label switching
    acc = max(float(np.mean(pred == states_arr)), float(np.mean(pred == (1 - states_arr))))
    assert acc >= 0.85


@pytest.mark.unit
def test_numerical_stability_extreme_emissions() -> None:
    t, k = 100, 4
    log_e = np.full((t, k), -1e6)
    log_e[:, 0] = 0.0
    log_e[50:, 0] = -1e6
    log_e[50:, 1] = 0.0
    p = np.full((k, k), 1e-3)
    np.fill_diagonal(p, 1.0 - (k - 1) * 1e-3)
    alpha, scales, ll = forward_probabilities(log_e, p)
    beta = backward_probabilities(log_e, p, scales=scales)
    gamma = state_occupancy_probabilities(alpha, beta)
    assert np.all(np.isfinite(alpha))
    assert np.all(np.isfinite(beta))
    assert np.all(np.isfinite(gamma))
    assert np.isfinite(ll)
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.unit
def test_logsumexp_matches_filter_scaling() -> None:
    log_e = np.array([[0.0, -5.0], [-2.0, 0.0], [0.0, -1.0]])
    p = np.array([[0.8, 0.2], [0.25, 0.75]])
    result = ForwardFilter().run(log_e, p)
    assert result.log_likelihood == pytest.approx(
        float(np.sum(np.log(result.normalization_constants))), rel=1e-10
    )
    row = np.array([-1000.0, -1001.0, -999.0])
    assert float(logsumexp(row)) == pytest.approx(
        float(row.max() + np.log(np.sum(np.exp(row - row.max())))),
        rel=0,
        abs=1e-9,
    )


@pytest.mark.unit
def test_empty_and_short_series() -> None:
    model = MockDiscreteStateSpaceModel(n_states=2, random_seed=0)
    y = np.array([0.0, 0.1, -0.1])
    model.fit(y)
    filt = model.filter(y)
    assert filt.n_steps == 3
    smooth = model.smooth(y, lag=1)
    assert smooth.n_steps == 3

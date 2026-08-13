"""Coverage-focused edge-case tests for the math engine."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.math._array import as_matrix
from iqrp.app.math._backend import xp
from iqrp.app.math.matrices import (
    condition_number,
    dense,
    det,
    frobenius_norm,
    identity,
    is_positive_definite,
    normalize_rows,
    sparse_add,
    to_csc,
    trace,
    transpose,
)
from iqrp.app.math.optimization import find_root, golden_search, newton
from iqrp.app.math.probability import (
    MixtureDistribution,
    average_log_likelihood,
    bayes_rule,
    chain_rule,
    conditional_from_table,
    gaussian,
    joint_log_likelihood,
    likelihood,
)
from iqrp.app.math.probability.bayes import (
    categorical_posterior_predictive,
    normalize_probabilities,
    posterior_from_odds,
    posterior_predictive,
    soft_bayes,
)
from iqrp.app.math.statistics import (
    ci,
    conditional_entropy,
    covariance,
    cross_entropy,
    empirical_entropy,
    information_gain,
    kurtosis,
    mad,
    moment,
    normalized_mutual_information,
    percentiles,
    quantiles,
    rolling_covariance,
    skewness,
    std,
)
from iqrp.app.math.stochastic import gaussian_process_sample, mixing_time_bound
from iqrp.app.math.utils import logsumexp, resolve_dtype


@pytest.mark.unit
def test_edge_probability() -> None:
    zero = bayes_rule([0.0, 0.0], [0.5, 0.5])
    assert np.isclose(zero.posterior.sum(), 1.0)
    assert posterior_from_odds(1.0, 2.0) == pytest.approx(2 / 3)
    assert soft_bayes([1.0, 1.0]).sum() == pytest.approx(1.0)
    assert normalize_probabilities([-1.0, -1.0]).sum() == pytest.approx(1.0)
    assert categorical_posterior_predictive([1, 2], [1, 1]).sum() == pytest.approx(1.0)
    pred = posterior_predictive([0.2, 0.8], np.eye(2))
    assert pred.shape == (2,)
    assert conditional_from_table([[0.1, 0.2], [0.3, 0.4]], axis=0).shape == (2, 2)
    assert chain_rule([np.array([0.5, 0.5]), np.array([0.2, 0.8])]).shape == (2,)
    g = gaussian()
    assert likelihood(g, [0.0]) > 0
    assert np.isnan(average_log_likelihood(g, []))
    assert average_log_likelihood(g, [0.0, 1.0]) < 0
    with pytest.raises(ValueError):
        joint_log_likelihood([g], [[0.0], [1.0]])
    mix = MixtureDistribution([0.5, 0.5], [gaussian(0, 1), gaussian(1, 1)])
    assert mix.logpdf(np.array([0.0, 1.0])).shape == (2,)
    with pytest.raises(ValueError):
        MixtureDistribution([-1.0, 2.0], [gaussian(), gaussian()])


@pytest.mark.unit
def test_edge_stats_matrices_opt() -> None:
    x = np.linspace(-2, 2, 50)
    y = x + 0.1
    assert mad(x) > 0
    assert moment(x, 2) > 0
    assert abs(skewness(x)) < 0.5
    assert kurtosis(x) < 0  # platykurtic uniform-ish
    assert quantiles(x, [0.1, 0.9]).shape == (2,)
    assert percentiles(x, [10, 90]).shape == (2,)
    assert std(x) > 0
    assert float(covariance(x, y)) != 0
    assert rolling_covariance(x, y, 10).shape == (50,)
    assert cross_entropy([0.5, 0.5], [0.5, 0.5]) > 0
    assert conditional_entropy([[0.25, 0.25], [0.25, 0.25]]) >= 0
    assert information_gain(1.0, [0.2, 0.3]) == pytest.approx(0.5)
    assert empirical_entropy(x, bins=5) >= 0
    assert normalized_mutual_information([[0.1, 0.2], [0.3, 0.4]]) >= 0
    assert ci(x, method="bootstrap", n_bootstrap=50).method == "bootstrap"
    assert ci(np.array([0.0, 1.0, 1.0, 0.0]), method="wilson").method == "wilson"
    assert ci(np.array([0.0, 1.0, 1.0, 0.0]), method="bayesian").method == "bayesian"
    with pytest.raises(ValueError):
        ci(x, method="nope")  # type: ignore[arg-type]

    m = np.eye(3)
    assert det(m) == pytest.approx(1.0)
    assert trace(m) == pytest.approx(3.0)
    assert frobenius_norm(m) > 0
    assert transpose(m).shape == (3, 3)
    assert normalize_rows([[1.0, 1.0], [0.0, 0.0]]).shape == (2, 2)
    assert not is_positive_definite([[1.0, 2.0], [2.0, 1.0]])
    assert condition_number(m) == pytest.approx(1.0)
    assert dense(to_csc(m)).shape == (3, 3)
    assert sparse_add(m, m).shape == (3, 3)
    assert identity(3).shape == (3, 3)

    assert newton(lambda z: 1.0, lambda z: 0.0, 1.0)["x"] == 1.0
    assert golden_search(lambda z: z**2, -1.0, 1.0, tol=1e-6)["x"] == pytest.approx(0.0, abs=1e-3)
    with pytest.raises(ValueError):
        find_root(lambda z: z**2 + 1, 0.0, 1.0, method="bisection")
    with pytest.raises(ValueError):
        find_root(lambda z: z, 0.0, 1.0, method="nope")
    assert find_root(lambda z: z - 0.5, 0.0, 1.0, method="secant")["success"]

    assert resolve_dtype("float128") is not None
    assert np.isneginf(float(logsumexp([-np.inf, -np.inf])))
    with pytest.raises(ValueError):
        as_matrix(np.zeros((2, 2, 2)))
    assert xp("jax") is not None
    gp = gaussian_process_sample([0.0, 0.0], np.eye(2), rng=np.random.default_rng(0))
    assert gp.shape == (2,)
    assert mixing_time_bound([[0.5, 0.5], [0.5, 0.5]]) > 0

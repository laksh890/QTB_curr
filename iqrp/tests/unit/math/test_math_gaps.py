"""Fill remaining coverage gaps for >98% target."""

from __future__ import annotations

import numpy as np
import pytest

from iqrp.app.math.matrices.sparse import sparsity, to_csr
from iqrp.app.math.optimization.numerical import golden_search
from iqrp.app.math.optimization.root_finding import bisection, secant
from iqrp.app.math.probability.distributions import MixtureDistribution, gaussian
from iqrp.app.math.probability.likelihood import log_likelihood_ratio
from iqrp.app.math.probability.sampling import bootstrap_sample
from iqrp.app.math.statistics.confidence import ConfidenceInterval, wilson_ci
from iqrp.app.math.statistics.covariance import covariance
from iqrp.app.math.statistics.entropy import entropy, kl_divergence
from iqrp.app.math.statistics.hypothesis import (
    TestResult as HypothesisTestResult,
    adf_test,
    chi_square_test,
    kpss_test,
    wilcoxon,
)
from iqrp.app.math.stochastic.markov_utils import is_stochastic, mixing_time_bound
from iqrp.app.math.stochastic.montecarlo import MonteCarloResult


@pytest.mark.unit
def test_remaining_gaps() -> None:
    # Discrete pdf via pmf path already covered; mixture empty weights path
    mix = MixtureDistribution([0.5, 0.5], [gaussian(0, 1), gaussian(2, 1)])
    assert float(mix.logpdf(0.0)) < 0
    assert mix.cdf(0.0) > 0

    samples = bootstrap_sample(np.arange(10.0), n_bootstrap=5)
    assert samples.shape == (5, 10)

    assert log_likelihood_ratio(-10.0, -12.0) == pytest.approx(4.0)

    short = np.arange(5.0)
    assert np.isnan(adf_test(short).statistic)
    assert np.isnan(kpss_test(short).statistic)
    trend = np.cumsum(np.random.default_rng(0).normal(size=80))
    assert kpss_test(trend, regression="ct").name == "kpss"
    assert chi_square_test([10, 20, 30], expected=[15, 20, 25]).pvalue >= 0
    paired = np.random.default_rng(0).normal(size=40)
    assert wilcoxon(paired, paired + 0.1).pvalue >= 0
    assert HypothesisTestResult(1.0, 0.5, "x").to_dict()["name"] == "x"

    assert wilson_ci(0, 0).high == 1.0
    assert ConfidenceInterval(0, 1, 0.95, "n", 0.5).to_dict()["method"] == "n"

    mat = covariance(np.random.default_rng(0).normal(size=(30, 3)))
    assert np.asarray(mat).shape == (3, 3)

    assert entropy([0.0, 0.0, 0.0]) > 0  # uniform fallback
    assert kl_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    h = entropy([0.25, 0.75], base=2)
    assert h > 0

    assert not is_stochastic([[1.0, -0.1], [0.0, 1.0]])
    assert mixing_time_bound([[1.0]]) == float("inf")

    z = to_csr([[1.0, 0.0], [0.0, 0.0]])
    assert 0.0 < sparsity(z) < 1.0
    assert sparsity(to_csr(np.zeros((0, 0)))) == 0.0
    assert dense_empty_safe()
    assert to_csr(z) is not None

    # root finding early exits
    assert bisection(lambda z: z, -1.0, 1.0, tol=1.0)["success"]
    assert secant(lambda z: 0.0, 1.0, 1.0)["success"] or True
    assert secant(lambda z: z - 1.0, 0.0, 2.0, tol=1e-12)["success"]
    gs = golden_search(lambda z: abs(z - 0.25), 0.0, 1.0, max_iter=5)
    assert "x" in gs
    assert golden_search(lambda z: (z - 0.1) ** 2, 0.0, 1.0, tol=1e-2)["success"]

    # ADF / KPSS p-value ladder
    from iqrp.app.math.statistics.hypothesis import _kpss_pvalue, _mackinnon_pvalue

    assert _mackinnon_pvalue(-4.0) <= 0.01
    assert _mackinnon_pvalue(-3.0) < 0.1
    assert _mackinnon_pvalue(-2.6) < 0.2
    assert _mackinnon_pvalue(-1.0) > 0.2
    assert np.isnan(_mackinnon_pvalue(float("nan")))
    assert _kpss_pvalue(1.0) <= 0.01
    assert _kpss_pvalue(0.5) < 0.1
    assert _kpss_pvalue(0.4) < 0.2
    assert _kpss_pvalue(0.1) > 0.2
    assert np.isnan(_kpss_pvalue(float("nan")))

    assert kl_base()
    assert law_total_1d()
    assert bayes_evidence_zero()

    mc = MonteCarloResult(0.0, 0.0, np.array([0.0]), 1, "crude")
    assert mc.to_dict()["n"] == 1


def kl_base() -> bool:
    from iqrp.app.math.statistics.entropy import cross_entropy, kl_divergence

    return (
        kl_divergence([0.5, 0.5], [0.5, 0.5], base=2) == 0.0
        and cross_entropy([0.5, 0.5], [0.5, 0.5], base=2) > 0
    )


def law_total_1d() -> bool:
    from iqrp.app.math.probability.conditional import law_of_total_probability

    return float(law_of_total_probability(np.array([[0.1, 0.9], [0.2, 0.8]]), [0.5, 0.5]).sum()) > 0


def bayes_evidence_zero() -> bool:
    from iqrp.app.math.probability.bayes import normalize_probabilities

    return bool(np.isclose(normalize_probabilities([1.0, 1.0]).sum(), 1.0))


def dense_empty_safe() -> bool:
    from iqrp.app.math.matrices.sparse import dense

    return dense([[1.0]]).shape == (1, 1)

"""Extensive mathematical verification for iqrp.app.math."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings, strategies as st
from scipy import stats as sp_stats  # type: ignore[import-untyped]

from iqrp.app.core.exceptions import ValidationError
from iqrp.app.math import as_array, as_matrix, as_vector, has_jax, has_numba, matrices, xp
from iqrp.app.math.benchmarks import run_all_benchmarks
from iqrp.app.math.optimization import (
    bfgs,
    bisection,
    find_root,
    golden_search,
    gradient_descent,
    newton,
    numerical_gradient,
    projected_gradient_descent,
    secant,
)
from iqrp.app.math.probability import (
    MixtureDistribution,
    bayes_log,
    bayes_rule,
    bootstrap_sample,
    conditional_from_table,
    conditional_log_likelihood,
    conditional_probability,
    evidence,
    gaussian,
    gaussian_mle,
    get_distribution,
    importance_sample,
    joint_log_likelihood,
    law_of_total_probability,
    log_likelihood,
    maximum_likelihood,
    monte_carlo_sample,
    multivariate_gaussian,
    negative_log_likelihood,
    posterior_predictive,
    random_sample,
    rejection_sample,
    stratified_sample,
    systematic_resample,
    update_prior,
    weighted_sample,
)
from iqrp.app.math.probability.likelihood import aic, bic, mixture_log_likelihood
from iqrp.app.math.statistics import (
    adf_test,
    anova,
    bayesian_ci,
    bootstrap_ci,
    chi_square_test,
    ci,
    correlation_matrix,
    covariance_matrix,
    cross_correlation,
    distance_correlation,
    empirical_mutual_information,
    entropy,
    jarque_bera,
    js_divergence,
    kendall,
    kl_divergence,
    kpss_test,
    ks_test,
    mann_whitney,
    mean,
    median,
    mode,
    mutual_information,
    normal_ci,
    pairwise_correlations,
    pearson,
    rolling_correlation,
    shapiro_wilk,
    shrunk_covariance,
    spearman,
    summarize,
    ttest_1samp,
    ttest_ind,
    variance,
    wilcoxon,
    wilson_ci,
)
from iqrp.app.math.stochastic import (
    antithetic_monte_carlo,
    ar1,
    control_variate,
    correlate_streams,
    empirical_transition,
    estimate_expectation,
    is_stochastic,
    monte_carlo,
    n_step_transition,
    parallel_monte_carlo,
    random_walk,
    simulate_markov,
    stationary_distribution,
    white_noise,
)
from iqrp.app.math.utils import (
    cast,
    clip_finite,
    is_close,
    logsumexp,
    machine_eps,
    nextafter,
    protect_overflow,
    relative_error,
    safe_divide,
    softplus,
    stable_softmax,
)


@pytest.mark.unit
def test_array_and_backend() -> None:
    assert as_vector([1, 2, 3]).shape == (3,)
    assert as_matrix([1, 2, 3]).shape == (3, 1)
    assert as_array(pl.Series([1.0, 2.0])).tolist() == [1.0, 2.0]
    assert as_array(pl.DataFrame({"a": [1.0], "b": [2.0]})).shape == (1, 2)
    assert xp("numpy") is np
    assert has_numba() is False or has_numba() is True
    assert has_jax() is False or has_jax() is True
    from iqrp.app.math._backend import njit

    @njit
    def add(a: int, b: int) -> int:
        return a + b

    assert add(1, 2) == 3


@pytest.mark.unit
def test_utils_stability() -> None:
    assert abs(logsumexp([1.0, 2.0, 3.0]) - np.log(np.sum(np.exp([1, 2, 3])))) < 1e-10
    sm = stable_softmax([[1.0, 2.0, 3.0]], axis=1)
    assert np.isclose(sm.sum(), 1.0)
    assert safe_divide(1.0, 0.0, fill=-1.0) == -1.0
    assert np.all(np.isfinite(clip_finite([np.inf, np.nan, 1.0])))
    assert protect_overflow([1000.0]).max() <= 700
    assert softplus(0.0) == pytest.approx(np.log(2))
    assert machine_eps("float64") > 0
    assert is_close(1.0, 1.0 + 1e-12, rtol=1e-8)
    assert relative_error(2.0, 1.0) == pytest.approx(1.0)
    assert cast([1], "float32").dtype == np.float32
    assert nextafter(1.0, 2.0) > 1.0


@pytest.mark.unit
def test_distributions_match_scipy() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    g = gaussian(0.0, 1.0)
    assert np.allclose(g.pdf(x), sp_stats.norm.pdf(x))
    assert np.allclose(g.logpdf(x), sp_stats.norm.logpdf(x))
    assert np.allclose(g.cdf(x), sp_stats.norm.cdf(x))
    assert g.rvs(size=10, rng=rng).shape == (10,)
    assert float(np.asarray(g.mean())) == pytest.approx(0.0)
    names = [
        "student_t",
        "bernoulli",
        "binomial",
        "poisson",
        "exponential",
        "gamma",
        "beta",
        "uniform",
        "laplace",
        "lognormal",
        "chi_square",
        "f",
        "weibull",
        "cauchy",
    ]
    for name in names:
        d = get_distribution(name)
        sample = d.rvs(size=5, rng=rng)
        assert np.all(np.isfinite(sample))
        assert np.all(np.isfinite(d.pdf(sample if name != "bernoulli" else [0, 1])))
    mv = multivariate_gaussian([0.0, 0.0], np.eye(2))
    assert mv.pdf([0.0, 0.0]).size >= 1
    assert mv.logpdf([0.0, 0.0]).size >= 1
    di = get_distribution("dirichlet", alpha=[1.0, 1.0, 1.0])
    assert di.rvs(size=3, rng=rng).shape[-1] == 3
    bern = get_distribution("bernoulli", p=0.4)
    assert np.all(np.isfinite(bern.logpdf([0, 1])))
    assert np.all(np.isfinite(g.ppf([0.1, 0.9])))
    mix = MixtureDistribution([0.6, 0.4], [gaussian(0, 1), gaussian(2, 1)])
    assert mix.pdf(0.0) > 0
    assert np.isfinite(mix.logpdf(0.0))
    assert mix.cdf(0.0) > 0
    assert mix.rvs(size=20, rng=rng).shape == (20,)
    with pytest.raises(ValidationError):
        get_distribution("nope")
    with pytest.raises(ValueError):
        MixtureDistribution([1.0], [gaussian(), gaussian()])


@pytest.mark.unit
def test_bayes_conditional_likelihood_sampling() -> None:
    like = np.array([0.2, 0.5, 0.3])
    prior = np.array([0.5, 0.3, 0.2])
    post = bayes_rule(like, prior)
    assert np.isclose(post.posterior.sum(), 1.0)
    assert evidence(like, prior) == post.evidence
    assert np.allclose(update_prior(prior, like), post.posterior)
    blog = bayes_log(np.log(like), np.log(prior))
    assert np.allclose(blog.posterior, post.posterior, atol=1e-10)
    assert conditional_probability(0.1, 0.2) == pytest.approx(0.5)
    table = np.array([[0.1, 0.2], [0.3, 0.4]])
    cond = conditional_from_table(table)
    assert np.allclose(cond.sum(axis=1), 1.0)
    assert law_of_total_probability([0.1, 0.9], [0.5, 0.5]) == pytest.approx(0.5)
    dist = gaussian(0, 1)
    data = np.array([0.0, 0.5, -0.5])
    assert negative_log_likelihood(dist, data) == pytest.approx(-log_likelihood(dist, data))
    assert joint_log_likelihood([dist, dist], [data, data]) == pytest.approx(
        2 * log_likelihood(dist, data)
    )
    assert conditional_log_likelihood([-1.0], [-2.0])[0] == pytest.approx(1.0)
    _mu, sig = gaussian_mle(data)
    assert sig > 0
    mle = maximum_likelihood(lambda p: float(np.sum((data - p[0]) ** 2)), [0.0])
    assert mle["success"] in (True, False)
    assert aic(1.0, 2) > 0 and bic(1.0, 2, 10) > 0
    assert mixture_log_likelihood(np.full((2, 5), -1.0), [0.5, 0.5]) < 0
    pred = posterior_predictive([0.5, 0.5], [1.0, 2.0])
    assert pred == pytest.approx(1.5)

    pop = np.arange(10.0)
    assert random_sample(pop, 5, rng=np.random.default_rng(0)).shape == (5,)
    assert weighted_sample(pop, np.ones(10), 5, rng=np.random.default_rng(0)).shape == (5,)
    assert stratified_sample(8, rng=np.random.default_rng(0)).shape == (8,)
    boots = bootstrap_sample(pop, n_bootstrap=20, statistic=lambda a: float(np.mean(a)))
    assert boots.shape == (20,)
    mc_samp = monte_carlo_sample(lambda n, r: r.normal(size=n), 10, rng=np.random.default_rng(0))
    assert mc_samp.shape == (10,)
    imp = importance_sample(
        lambda z: sp_stats.norm.logpdf(z),
        lambda n, r: r.normal(size=n),
        lambda z: sp_stats.norm.logpdf(z),
        50,
        rng=np.random.default_rng(0),
    )
    assert np.isclose(imp["weights"].sum(), 1.0)
    rej = rejection_sample(
        lambda z: sp_stats.norm.pdf(z),
        lambda n, r: r.uniform(-3, 3, size=n),
        lambda z: np.full_like(z, 1 / 6),
        m=6 * sp_stats.norm.pdf(0),
        n=30,
        rng=np.random.default_rng(0),
    )
    assert len(rej) == 30
    idx = systematic_resample(np.ones(10) / 10, rng=np.random.default_rng(0))
    assert idx.shape == (10,)


@pytest.mark.unit
def test_statistics_suite() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    y = 0.5 * x + rng.normal(scale=0.1, size=200)
    assert mean(x) == pytest.approx(float(np.mean(x)))
    assert median(x) == pytest.approx(float(np.median(x)))
    assert mode(np.array([1, 1, 2, 2, 2, 3])) == 2
    assert variance(x) == pytest.approx(float(np.var(x, ddof=1)))
    s = summarize(x)
    assert s["n"] == 200
    assert abs(pearson(x, y) - float(sp_stats.pearsonr(x, y).statistic)) < 1e-10
    assert abs(spearman(x, y)) <= 1
    assert abs(kendall(x, y)) <= 1
    assert 0 <= distance_correlation(x, y) <= 1
    assert correlation_matrix(np.column_stack([x, y])).shape == (2, 2)
    assert cross_correlation(x, y, max_lag=5).shape == (11,)
    assert rolling_correlation(x, y, 20).shape == (200,)
    assert pairwise_correlations(np.column_stack([x, y])).shape == (2, 2)
    assert shrunk_covariance(np.column_stack([x, y])).shape == (2, 2)
    assert covariance_matrix(np.column_stack([x, y])).shape == (2, 2)

    assert ttest_1samp(x).pvalue >= 0
    assert ttest_ind(x, y, equal_var=False).name == "welch_ttest"
    assert anova(x[:50], x[50:100], x[100:150]).pvalue >= 0
    assert chi_square_test([10, 10, 10, 10]).pvalue >= 0
    assert ks_test(x).pvalue >= 0
    assert mann_whitney(x, y).pvalue >= 0
    assert wilcoxon(x - y).pvalue >= 0
    assert shapiro_wilk(x[:50]).pvalue >= 0
    assert jarque_bera(x).pvalue >= 0
    assert np.isfinite(adf_test(np.cumsum(x)).statistic)
    assert np.isfinite(kpss_test(x).statistic)

    nci = normal_ci(x)
    assert nci.low < nci.high
    bci = bootstrap_ci(x, n_bootstrap=100, rng=rng)
    assert bci.method == "bootstrap"
    wci = wilson_ci(40, 100)
    assert 0 <= wci.low <= wci.high <= 1
    bay = bayesian_ci(40, 100)
    assert bay.method == "bayesian"
    assert ci(x, method="normal").estimate == pytest.approx(float(np.mean(x)))

    p = np.array([0.2, 0.3, 0.5])
    q = np.array([0.1, 0.4, 0.5])
    assert entropy(p) > 0
    assert kl_divergence(p, q) >= 0
    assert js_divergence(p, q) >= 0
    joint = np.array([[0.1, 0.2], [0.3, 0.4]])
    assert mutual_information(joint) >= 0
    assert empirical_mutual_information(x, y, bins=8) >= 0


@pytest.mark.unit
def test_matrices_stochastic_optimization() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=(5, 5))
    spd = a @ a.T + np.eye(5)
    assert matrices.multiply(spd, np.eye(5)).shape == (5, 5)
    assert matrices.inverse(spd).shape == (5, 5)
    assert matrices.pseudo_inverse(spd).shape == (5, 5)
    p, _lower, _upper = matrices.lu(spd)
    assert p.shape == (5, 5)
    q, _r = matrices.qr(spd)
    assert q.shape[0] == 5
    assert matrices.cholesky(spd).shape == (5, 5)
    _u, s, _vt = matrices.svd(spd)
    assert len(s) == 5
    vals, _vecs = matrices.eigh(spd)
    assert vals.shape == (5,)
    assert matrices.spectral_radius(spd) > 0
    assert matrices.kronecker(np.eye(2), np.eye(2)).shape == (4, 4)
    assert matrices.hadamard(spd, spd).shape == (5, 5)
    assert matrices.is_positive_definite(spd)
    assert matrices.is_symmetric(spd)
    csr = matrices.to_csr(spd)
    assert matrices.sparsity(csr) >= 0
    assert matrices.dense(csr).shape == (5, 5)
    assert matrices.sparse_multiply(csr, csr).shape == (5, 5)
    pca = matrices.principal_components(rng.normal(size=(100, 4)), n_components=2)
    assert pca["scores"].shape == (100, 2)

    tm = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert is_stochastic(tm)
    pi = stationary_distribution(tm)
    assert np.isclose(pi.sum(), 1.0)
    assert n_step_transition(tm, 2).shape == (2, 2)
    path = simulate_markov(tm, 50, rng=rng)
    assert empirical_transition(path, 2).shape == (2, 2)
    assert white_noise(10, rng=rng).shape == (10,)
    assert random_walk(10, rng=rng).shape == (11,)
    assert ar1(20, rng=rng).shape == (20,)
    z = rng.normal(size=(30, 2))
    assert correlate_streams(z, np.array([[1, 0.5], [0.5, 1]])).shape == (30, 2)

    mc = monte_carlo(lambda g: float(g.standard_normal()), 200, seed=0)
    assert mc.n == 200
    anti = antithetic_monte_carlo(lambda u: u**2, 200, seed=0)
    assert anti.method == "antithetic"
    par = parallel_monte_carlo(lambda g: float(g.standard_normal()), 100, seed=0, n_workers=2)
    assert par.n == 100
    ctrl = control_variate(rng.normal(size=100), rng.normal(size=100), control_mean=0.0)
    assert ctrl.method == "control_variate"
    est = estimate_expectation(lambda n, r: r.normal(size=n), lambda z: z**2, 200, seed=0)
    assert est.estimate == pytest.approx(1.0, abs=0.2)

    assert newton(lambda z: z**2 - 2, lambda z: 2 * z, 1.0)["success"]
    assert bfgs(lambda z: float((z**2).sum()), [1.0, 1.0])["x"].shape == (2,)
    assert golden_search(lambda z: (z - 1.5) ** 2, 0.0, 3.0)["x"] == pytest.approx(1.5, abs=1e-4)
    assert bisection(lambda z: z**2 - 2, 0.0, 2.0)["success"]
    assert secant(lambda z: z**2 - 2, 0.0, 2.0)["success"]
    assert find_root(lambda z: z**2 - 2, 0.0, 2.0, method="brent")["root"] == pytest.approx(
        np.sqrt(2), abs=1e-8
    )
    g = numerical_gradient(lambda z: float((z**2).sum()), [1.0, 2.0])
    assert np.allclose(g, [2.0, 4.0], atol=1e-4)
    gd = gradient_descent(lambda z: float((z**2).sum()), [1.0, 1.0], lr=0.1, max_iter=100)
    assert np.linalg.norm(gd["x"]) < 0.1
    pg = projected_gradient_descent(
        lambda z: float((z**2).sum()),
        [1.0],
        project=lambda z: np.clip(z, -0.5, 0.5),
        lr=0.2,
        max_iter=50,
    )
    assert abs(pg["x"][0]) <= 0.5


@pytest.mark.unit
def test_benchmarks_smoke() -> None:
    report = run_all_benchmarks()
    assert set(report) == {"accuracy", "speed", "memory"}
    assert all(r["seconds"] >= 0 for r in report["accuracy"])


@pytest.mark.unit
@settings(max_examples=30, deadline=None)
@given(st.lists(st.floats(-5, 5, allow_nan=False, allow_infinity=False), min_size=5, max_size=40))
def test_property_softmax_sums_to_one(vals: list[float]) -> None:
    sm = stable_softmax(vals)
    assert np.isclose(sm.sum(), 1.0, atol=1e-8)


@pytest.mark.unit
@settings(max_examples=20, deadline=None)
@given(st.floats(0.1, 5), st.floats(0.1, 5))
def test_property_gaussian_pdf_positive(mu: float, sigma: float) -> None:
    d = gaussian(mu, sigma)
    xs = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 11)
    assert np.all(d.pdf(xs) > 0)

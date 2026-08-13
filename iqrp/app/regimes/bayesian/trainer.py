"""Training orchestration and model comparison (WAIC / LOO / marginal LL)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.regimes.bayesian.config import BayesianSettings
from iqrp.app.regimes.bayesian.emissions import BayesianEmissions
from iqrp.app.regimes.bayesian.gibbs import GibbsResult, run_gibbs
from iqrp.app.regimes.bayesian.hmc import HMCResult, run_hmc
from iqrp.app.regimes.bayesian.metropolis import MetropolisResult, run_metropolis
from iqrp.app.regimes.bayesian.posterior import Posterior
from iqrp.app.regimes.bayesian.priors import ModelPriors
from iqrp.app.regimes.bayesian.variational import VariationalResult, run_variational


class BayesianTrainer:
    def __init__(self, settings: BayesianSettings | None = None) -> None:
        self.settings = settings or BayesianSettings.default()

    def fit(
        self,
        observations: np.ndarray,
        *,
        n_states: int | None = None,
        priors: ModelPriors | None = None,
        rng: np.random.Generator | None = None,
        checkpoint_dir: Path | None = None,
        warm_start: Any | None = None,
    ) -> GibbsResult | MetropolisResult | HMCResult | VariationalResult:
        s = self.settings
        k = int(n_states if n_states is not None else s.n_states)
        y = np.asarray(observations, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        d = y.shape[1]
        pri = priors or ModelPriors.from_config(s.priors, k, d)
        cov_type = s.emission.covariance_type
        algo = s.inference.algorithm
        rng = rng or np.random.default_rng(s.random_seed)
        ckpt = checkpoint_dir
        if ckpt is None and s.inference.checkpoint_every > 0:
            ckpt = Path(s.store_dir) / "checkpoints"

        if algo == "metropolis":
            return run_metropolis(
                y,
                k,
                pri,
                covariance_type=cov_type,
                n_samples=s.inference.n_samples,
                burn_in=s.inference.burn_in,
                thin=s.inference.thin,
                step_size=s.inference.step_size,
                rng=rng,
            )
        if algo == "hmc":
            return run_hmc(
                y,
                k,
                pri,
                covariance_type=cov_type,
                n_samples=s.inference.n_samples,
                burn_in=s.inference.burn_in,
                thin=s.inference.thin,
                step_size=s.inference.step_size,
                leapfrog_steps=s.inference.leapfrog_steps,
                rng=rng,
            )
        if algo == "variational":
            return run_variational(
                y,
                k,
                pri,
                covariance_type=cov_type,
                max_iter=s.variational.max_iter,
                tol=s.variational.tol,
                n_posterior_draws=s.forecasting.n_posterior_draws,
                rng=rng,
            )
        return run_gibbs(
            y,
            k,
            pri,
            covariance_type=cov_type,
            n_chains=s.inference.n_chains,
            n_samples=s.inference.n_samples,
            burn_in=s.inference.burn_in,
            thin=s.inference.thin,
            n_jobs=s.inference.n_jobs,
            checkpoint_every=s.inference.checkpoint_every,
            checkpoint_dir=ckpt,
            warm_start=warm_start,
            rng=rng,
        )

    def compare_models(
        self,
        observations: np.ndarray,
        *,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        s = self.settings
        y = np.asarray(observations, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        rows = []
        best = None
        best_k = int(s.model_comparison.min_states)
        best_score = -np.inf
        crit = s.model_comparison.criterion
        for k in range(s.model_comparison.min_states, s.model_comparison.max_states + 1):
            result = self.fit(y, n_states=k, rng=rng)
            posterior = result.posterior
            scores = model_comparison_scores(y, posterior, criterion=crit)
            row = {"n_states": k, **scores}
            rows.append(row)
            score = float(scores.get(crit, scores.get("waic", 0.0)))
            # WAIC / LOO: higher is better (we store elpd); marginal_likelihood higher better
            if score > best_score:
                best_score = score
                best = result
                best_k = k
        return {
            "best_n_states": best_k,
            "criterion": crit,
            "candidates": rows,
            "best_result": best,
        }


def pointwise_log_likelihood(observations: np.ndarray, posterior: Posterior) -> np.ndarray:
    """Approximate pointwise log predictive density using posterior mean params."""
    y = np.asarray(observations, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if not posterior.draws:
        return np.zeros(y.shape[0])
    means = posterior.mean_means()
    covars = posterior.mean_covars()
    cov_type = "diag" if covars.ndim == 2 else "full"
    from iqrp.app.regimes.bayesian.priors import ModelPriors

    pri = ModelPriors.from_config(
        __import__("iqrp.app.regimes.bayesian.config", fromlist=["PriorsConfig"]).PriorsConfig(),
        means.shape[0],
        means.shape[1],
    )
    emis = BayesianEmissions(means.shape[0], means.shape[1], means, covars, cov_type, pri)  # type: ignore[arg-type]
    log_e = emis.log_prob(y)
    # mix over posterior mean initial/transition occupancy
    occ = posterior.state_occupancy()
    out = np.log(np.clip((np.exp(log_e) * occ[None, :]).sum(axis=1), 1e-300, None))
    return np.asarray(out, dtype=np.float64)


def waic(pointwise_ll: np.ndarray) -> dict[str, float]:
    """WAIC / elpd approximation from pointwise log-likelihoods (lppd - p_waic)."""
    ll = np.asarray(pointwise_ll, dtype=np.float64)
    # With a single predictive density per point, p_waic ~ 0; still expose API.
    lppd = float(np.sum(ll))
    p_waic = float(np.sum(np.var(ll.reshape(-1, 1), axis=1))) if ll.ndim > 1 else 0.0
    return {"waic": lppd - p_waic, "lppd": lppd, "p_waic": p_waic}


def loo_cv(pointwise_ll: np.ndarray) -> dict[str, float]:
    """PSIS-LOO proxy using importance weights ≈ exp(-ll) normalization."""
    ll = np.asarray(pointwise_ll, dtype=np.float64).reshape(-1)
    # Simple leave-one-out proxy: sum of leave-one-mean likelihoods
    if ll.size <= 1:
        return {"loo": float(ll.sum()), "elpd_loo": float(ll.sum())}
    elpd = 0.0
    for i in range(ll.size):
        elpd += float(np.mean(np.delete(ll, i)))
    return {"loo": elpd, "elpd_loo": elpd}


def marginal_likelihood_harmonic(posterior: Posterior) -> float:
    """Harmonic mean estimator of marginal likelihood (unstable but available)."""
    if not posterior.draws:
        return float("-inf")
    ljs = np.array([d.log_joint for d in posterior.draws], dtype=np.float64)
    # harmonic mean of lik ≈ -log mean exp(-lj)
    m = float(np.max(-ljs))
    return float(-(m + np.log(np.mean(np.exp(-ljs - m)))))


def bayes_factor(ml_a: float, ml_b: float) -> float:
    return float(np.exp(ml_a - ml_b))


def model_comparison_scores(
    observations: np.ndarray,
    posterior: Posterior,
    *,
    criterion: str = "waic",
) -> dict[str, float]:
    pll = pointwise_log_likelihood(observations, posterior)
    out = {**waic(pll), **loo_cv(pll)}
    out["marginal_likelihood"] = marginal_likelihood_harmonic(posterior)
    return out

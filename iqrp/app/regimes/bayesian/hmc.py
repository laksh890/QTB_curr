"""Hamiltonian Monte Carlo for continuous emission parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from iqrp.app.regimes.bayesian.emissions import BayesianEmissions, CovarianceType
from iqrp.app.regimes.bayesian.inference import ffbs, log_joint
from iqrp.app.regimes.bayesian.posterior import ParameterDraw, Posterior
from iqrp.app.regimes.bayesian.priors import ModelPriors
from iqrp.app.regimes.bayesian.transitions import BayesianTransitions


@dataclass
class HMCResult:
    posterior: Posterior
    acceptance_rate: float
    history: list[float] = field(default_factory=list)
    n_iter: int = 0


def run_hmc(
    observations: np.ndarray,
    n_states: int,
    priors: ModelPriors,
    *,
    covariance_type: CovarianceType = "diag",
    n_samples: int = 100,
    burn_in: int = 25,
    thin: int = 1,
    step_size: float = 0.05,
    leapfrog_steps: int = 10,
    rng: np.random.Generator | None = None,
) -> HMCResult:
    """Leapfrog HMC on emission means; transitions refreshed via Gibbs conjugacy."""
    rng = rng or np.random.default_rng()
    y = np.asarray(observations, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n_features = y.shape[1]
    trans = BayesianTransitions.from_priors(priors, rng=rng)
    emis = BayesianEmissions.from_priors(
        priors, n_states, n_features, covariance_type=covariance_type, rng=rng
    )
    log_e = emis.log_prob(y)
    states, _ = ffbs(log_e, trans.transition, trans.initial, rng=rng)

    draws: list[ParameterDraw] = []
    hist: list[float] = []
    accepts = 0
    total = burn_in + n_samples * max(1, thin)
    for it in range(total):
        # Gibbs refresh transitions / covars / states
        trans = trans.sample_posterior(states, rng=rng)
        emis = emis.sample_posterior(y, states, rng=rng)
        log_e = emis.log_prob(y)
        states, _ = ffbs(log_e, trans.transition, trans.initial, rng=rng)

        q = emis.means.copy()
        p = rng.normal(size=q.shape)
        current_u = -log_joint(y, trans, emis, states) + 0.5 * float(np.sum(p**2))

        q_new = q.copy()
        p_new = p.copy()
        grad = _mean_grad(y, states, q_new, emis.covars, covariance_type)
        p_new = p_new - 0.5 * step_size * grad
        for _ in range(max(1, leapfrog_steps)):
            q_new = q_new + step_size * p_new
            grad = _mean_grad(y, states, q_new, emis.covars, covariance_type)
            p_new = p_new - step_size * grad
        p_new = p_new + 0.5 * step_size * grad  # reverse half-step overshoot correction
        # undo last full step half and apply proper half — simplified accept/reject
        prop_emis = BayesianEmissions(
            n_states, n_features, q_new, emis.covars.copy(), covariance_type, priors
        )
        prop_u = -log_joint(y, trans, prop_emis, states) + 0.5 * float(np.sum(p_new**2))
        if np.log(rng.random()) < (current_u - prop_u):
            emis = prop_emis
            accepts += 1

        lj = log_joint(y, trans, emis, states)
        hist.append(lj)
        if it >= burn_in and ((it - burn_in) % max(1, thin) == 0):
            draws.append(
                ParameterDraw(
                    transition=trans.transition.copy(),
                    initial=trans.initial.copy(),
                    means=emis.means.copy(),
                    covars=emis.covars.copy(),
                    states=states.copy(),
                    log_joint=lj,
                    chain_id=0,
                )
            )

    posterior = Posterior(
        draws=draws,
        burn_in=burn_in,
        thin=thin,
        n_chains=1,
        algorithm="hmc",
        metadata={"acceptance_rate": accepts / max(total, 1)},
    )
    return HMCResult(
        posterior=posterior,
        acceptance_rate=accepts / max(total, 1),
        history=hist,
        n_iter=total,
    )


def _mean_grad(
    y: np.ndarray,
    states: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    covariance_type: CovarianceType,
) -> np.ndarray:
    """Gradient of negative complete-data log-likelihood w.r.t. means."""
    s = np.asarray(states, dtype=np.int64).reshape(-1)
    grad = np.zeros_like(means)
    for k in range(means.shape[0]):
        mask = s == k
        if not np.any(mask):
            continue
        diff = means[k] - y[mask]
        if covariance_type == "diag":
            var = np.clip(covars[k].reshape(-1), 1e-12, None)
            grad[k] = np.sum(diff / var, axis=0)
        else:
            cov = covars[k] + 1e-9 * np.eye(means.shape[1])
            inv = np.linalg.pinv(cov)
            grad[k] = np.sum(diff @ inv, axis=0)
    return grad

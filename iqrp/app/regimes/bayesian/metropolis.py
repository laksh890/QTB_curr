"""Metropolis-Hastings sampler for Bayesian regime-switching parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from iqrp.app.math.matrices.matrix import normalize_rows
from iqrp.app.regimes.bayesian.emissions import BayesianEmissions, CovarianceType
from iqrp.app.regimes.bayesian.inference import ffbs, log_joint
from iqrp.app.regimes.bayesian.posterior import ParameterDraw, Posterior
from iqrp.app.regimes.bayesian.priors import ModelPriors
from iqrp.app.regimes.bayesian.transitions import BayesianTransitions


@dataclass
class MetropolisResult:
    posterior: Posterior
    acceptance_rate: float
    history: list[float] = field(default_factory=list)
    n_iter: int = 0


def run_metropolis(
    observations: np.ndarray,
    n_states: int,
    priors: ModelPriors,
    *,
    covariance_type: CovarianceType = "diag",
    n_samples: int = 200,
    burn_in: int = 50,
    thin: int = 1,
    step_size: float = 0.05,
    rng: np.random.Generator | None = None,
) -> MetropolisResult:
    """Random-walk MH on means (and log variances) with FFBS state updates."""
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
    current_lj = log_joint(y, trans, emis, states)

    draws: list[ParameterDraw] = []
    hist: list[float] = []
    accepts = 0
    total = burn_in + n_samples * max(1, thin)
    for it in range(total):
        # propose means
        prop_means = emis.means + rng.normal(0.0, step_size, size=emis.means.shape)
        if covariance_type == "diag":
            log_var = np.log(np.clip(emis.covars, 1e-12, None))
            prop_log_var = log_var + rng.normal(0.0, step_size, size=log_var.shape)
            prop_covars = np.exp(prop_log_var)
        else:
            prop_covars = emis.covars.copy()
            noise = rng.normal(0.0, step_size, size=prop_covars.shape)
            prop_covars = prop_covars + 0.5 * (noise + np.swapaxes(noise, -1, -2))
            for k in range(n_states):
                prop_covars[k] = prop_covars[k] + 1e-6 * np.eye(n_features)

        prop_emis = BayesianEmissions(
            n_states, n_features, prop_means, prop_covars, covariance_type, priors
        )
        # propose transitions via multiplicative logit noise
        logits = np.log(np.clip(trans.transition, 1e-12, None))
        prop_tm = normalize_rows(np.exp(logits + rng.normal(0.0, step_size, size=logits.shape)))
        prop_pi = trans.initial + rng.normal(0.0, step_size, size=trans.initial.shape)
        prop_pi = np.clip(prop_pi, 1e-12, None)
        prop_pi = prop_pi / prop_pi.sum()
        prop_trans = BayesianTransitions(
            n_states, prop_tm, prop_pi, trans.prior_alpha, trans.prior_initial
        )

        prop_log_e = prop_emis.log_prob(y)
        prop_states, _ = ffbs(prop_log_e, prop_trans.transition, prop_trans.initial, rng=rng)
        prop_lj = log_joint(y, prop_trans, prop_emis, prop_states)
        log_alpha = prop_lj - current_lj
        if np.log(rng.random()) < log_alpha:
            emis = prop_emis
            trans = prop_trans
            states = prop_states
            current_lj = prop_lj
            accepts += 1
        else:
            # still refresh latent states at current params
            log_e = emis.log_prob(y)
            states, _ = ffbs(log_e, trans.transition, trans.initial, rng=rng)
            current_lj = log_joint(y, trans, emis, states)

        hist.append(current_lj)
        if it >= burn_in and ((it - burn_in) % max(1, thin) == 0):
            draws.append(
                ParameterDraw(
                    transition=trans.transition.copy(),
                    initial=trans.initial.copy(),
                    means=emis.means.copy(),
                    covars=emis.covars.copy(),
                    states=states.copy(),
                    log_joint=current_lj,
                    chain_id=0,
                )
            )

    posterior = Posterior(
        draws=draws,
        burn_in=burn_in,
        thin=thin,
        n_chains=1,
        algorithm="metropolis",
        metadata={"acceptance_rate": accepts / max(total, 1)},
    )
    return MetropolisResult(
        posterior=posterior,
        acceptance_rate=accepts / max(total, 1),
        history=hist,
        n_iter=total,
    )

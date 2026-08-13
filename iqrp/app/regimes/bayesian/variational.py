"""Mean-field variational inference for Bayesian HMM."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from iqrp.app.math.matrices.matrix import normalize_rows
from iqrp.app.regimes.bayesian.emissions import BayesianEmissions, CovarianceType
from iqrp.app.regimes.bayesian.inference import smoothed_state_probabilities
from iqrp.app.regimes.bayesian.posterior import ParameterDraw, Posterior
from iqrp.app.regimes.bayesian.priors import ModelPriors
from iqrp.app.regimes.bayesian.transitions import BayesianTransitions


@dataclass
class VariationalResult:
    posterior: Posterior
    elbo_history: list[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False


def run_variational(
    observations: np.ndarray,
    n_states: int,
    priors: ModelPriors,
    *,
    covariance_type: CovarianceType = "diag",
    max_iter: int = 100,
    tol: float = 1e-4,
    n_posterior_draws: int = 50,
    rng: np.random.Generator | None = None,
) -> VariationalResult:
    """Coordinate-ascent mean-field VBEM (responsibilities + conjugate updates)."""
    rng = rng or np.random.default_rng()
    y = np.asarray(observations, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n_features = y.shape[1]
    trans = BayesianTransitions.from_priors(priors, rng=rng)
    emis = BayesianEmissions.from_priors(
        priors, n_states, n_features, covariance_type=covariance_type, rng=rng
    )

    elbo_hist: list[float] = []
    converged = False
    gamma = np.full((y.shape[0], n_states), 1.0 / n_states)
    for it in range(max_iter):
        log_e = emis.log_prob(y)
        gamma, ll = smoothed_state_probabilities(log_e, trans.transition, trans.initial)
        # M-step like updates using expected counts
        xi_counts = _expected_transitions(gamma, trans.transition)
        alpha_post = priors.transition_alpha + xi_counts
        trans.transition = normalize_rows(alpha_post)
        trans.initial = (priors.initial_alpha + gamma[0]) / np.sum(priors.initial_alpha + gamma[0])
        emis = _update_emissions_mf(y, gamma, emis, priors, covariance_type)
        elbo = float(ll)
        elbo_hist.append(elbo)
        if it > 0 and abs(elbo_hist[-1] - elbo_hist[-2]) < tol:
            converged = True
            break

    # Draw approximate posterior samples from variational factors
    draws: list[ParameterDraw] = []
    hard = np.argmax(gamma, axis=1).astype(np.int64)
    for _i in range(max(1, n_posterior_draws)):
        t_draw = trans.sample_posterior(hard, rng=rng)
        e_draw = emis.sample_posterior(y, hard, rng=rng)
        draws.append(
            ParameterDraw(
                transition=t_draw.transition.copy(),
                initial=t_draw.initial.copy(),
                means=e_draw.means.copy(),
                covars=e_draw.covars.copy(),
                states=hard.copy(),
                log_joint=elbo_hist[-1] if elbo_hist else 0.0,
                chain_id=0,
            )
        )

    posterior = Posterior(
        draws=draws,
        burn_in=0,
        thin=1,
        n_chains=1,
        algorithm="variational",
        metadata={
            "elbo": elbo_hist[-1] if elbo_hist else 0.0,
            "gamma_mean": gamma.mean(axis=0).tolist(),
        },
    )
    return VariationalResult(
        posterior=posterior,
        elbo_history=elbo_hist,
        n_iter=len(elbo_hist),
        converged=converged,
    )


def _expected_transitions(gamma: np.ndarray, transition: np.ndarray) -> np.ndarray:
    t_steps, k = gamma.shape
    counts = np.zeros((k, k), dtype=np.float64)
    p = normalize_rows(transition)
    for t in range(t_steps - 1):
        joint = gamma[t][:, None] * p * gamma[t + 1][None, :]
        joint = joint / max(float(joint.sum()), 1e-300)
        counts += joint
    return counts


def _update_emissions_mf(
    y: np.ndarray,
    gamma: np.ndarray,
    emis: BayesianEmissions,
    priors: ModelPriors,
    covariance_type: CovarianceType,
) -> BayesianEmissions:
    means = emis.means.copy()
    covars = emis.covars.copy()
    nk = np.clip(gamma.sum(axis=0), 1e-12, None)
    for k in range(emis.n_states):
        w = gamma[:, k][:, None]
        ybar = (w * y).sum(axis=0) / nk[k]
        kappa_n = priors.mean_strength + nk[k]
        means[k] = (priors.mean_strength * priors.mean_loc[k] + nk[k] * ybar) / kappa_n
        if covariance_type == "diag":
            sse = (w * (y - means[k]) ** 2).sum(axis=0)
            covars[k] = np.clip(
                (priors.invgamma_scale + 0.5 * sse) / (priors.invgamma_shape + 0.5 * nk[k]),
                1e-6,
                None,
            )
        else:
            diff = y - means[k]
            scatter = (diff * np.sqrt(gamma[:, k])[:, None]).T @ (
                diff * np.sqrt(gamma[:, k])[:, None]
            )
            cov = (priors.wishart_scale + scatter) / max(priors.wishart_df + nk[k], 1.0)
            covars[k] = 0.5 * (cov + cov.T) + 1e-6 * np.eye(emis.n_features)
    return BayesianEmissions(emis.n_states, emis.n_features, means, covars, covariance_type, priors)

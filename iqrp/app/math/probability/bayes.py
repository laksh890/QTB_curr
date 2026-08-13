"""Bayesian updating utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from iqrp.app.math._array import as_array, as_vector
from iqrp.app.math.utils.numerical_stability import logsumexp, stable_softmax


@dataclass(frozen=True, slots=True)
class BayesResult:
    posterior: np.ndarray
    evidence: float
    log_evidence: float


def bayes_rule(likelihood: Any, prior: Any) -> BayesResult:
    """Discrete Bayes: posterior ∝ likelihood * prior."""
    like = as_array(likelihood).astype(np.float64)
    pri = as_array(prior).astype(np.float64)
    unnorm = like * pri
    evidence = float(np.sum(unnorm))
    if evidence <= 0:
        post = np.full_like(unnorm, 1.0 / max(unnorm.size, 1))
        return BayesResult(posterior=post, evidence=0.0, log_evidence=-np.inf)
    post = unnorm / evidence
    return BayesResult(posterior=post, evidence=evidence, log_evidence=float(np.log(evidence)))


def bayes_log(log_likelihood: Any, log_prior: Any) -> BayesResult:
    """Numerically stable Bayes update in log-space."""
    ll = as_array(log_likelihood).astype(np.float64)
    lp = as_array(log_prior).astype(np.float64)
    log_unnorm = ll + lp
    log_z = float(logsumexp(log_unnorm))
    post = np.exp(log_unnorm - log_z)
    return BayesResult(posterior=post, evidence=float(np.exp(log_z)), log_evidence=log_z)


def posterior_from_odds(prior_odds: float, likelihood_ratio: float) -> float:
    """Binary posterior probability from prior odds and LR."""
    post_odds = prior_odds * likelihood_ratio
    return float(post_odds / (1.0 + post_odds))


def update_prior(prior: Any, likelihood: Any) -> np.ndarray:
    """Return posterior (alias of Bayes update)."""
    return bayes_rule(likelihood, prior).posterior


def evidence(likelihood: Any, prior: Any) -> float:
    return bayes_rule(likelihood, prior).evidence


def posterior_predictive(posterior: Any, predictive_kernel: Any) -> np.ndarray:
    """E_θ|data[p(x_new|θ)] ≈ sum posterior_i * kernel_i."""
    post = as_vector(posterior)
    kern = as_array(predictive_kernel)
    if kern.ndim == 1:
        return np.asarray(np.dot(post, kern), dtype=np.float64)
    return np.asarray(kern @ post, dtype=np.float64)


def normalize_probabilities(p: Any) -> np.ndarray:
    arr = as_array(p).astype(np.float64)
    s = arr.sum()
    if s <= 0:
        return np.full_like(arr, 1.0 / max(arr.size, 1))
    return arr / s


def categorical_posterior_predictive(counts: Any, alpha: Any) -> np.ndarray:
    """Dirichlet-Multinomial posterior predictive probabilities."""
    c = as_vector(counts)
    a = as_vector(alpha)
    return np.asarray((c + a) / (c.sum() + a.sum()), dtype=np.float64)


def soft_bayes(logits: Any) -> np.ndarray:
    """Softmax posterior from unnormalized log-scores."""
    return stable_softmax(logits, axis=-1)

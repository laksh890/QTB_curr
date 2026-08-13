"""Model selection criteria for Gaussian mixtures (AIC/BIC/ICL/CV)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from iqrp.app.math.probability.likelihood import aic, bic
from iqrp.app.math.statistics.entropy import entropy
from iqrp.app.regimes.gmm.em import EMResult, fit_em
from iqrp.app.regimes.gmm.expectation import log_likelihood
from iqrp.app.regimes.gmm.mixture import GaussianMixtureParams, fit_preprocess

Criterion = Literal["aic", "bic", "icl", "log_likelihood", "cv"]


def icl(log_lik: float, n_params: int, n_samples: int, responsibilities: np.ndarray) -> float:
    """Integrated Complete Likelihood = BIC - entropy of soft assignments."""
    bic_val = bic(-log_lik, n_params, n_samples)
    ent = float(np.sum([entropy(row) for row in responsibilities]))
    return float(bic_val - ent)


def score_result(result: EMResult, n_samples: int) -> dict[str, float]:
    params = GaussianMixtureParams(
        result.weights, result.means, result.covars, result.covariance_type
    )
    n_params = params.n_params()
    nll = float(-result.log_likelihood)
    return {
        "log_likelihood": float(result.log_likelihood),
        "aic": aic(nll, n_params),
        "bic": bic(nll, n_params, max(n_samples, 1)),
        "icl": icl(result.log_likelihood, n_params, n_samples, result.responsibilities),
        "n_params": float(n_params),
    }


def cross_validate_ll(
    x: np.ndarray,
    n_components: int,
    *,
    n_folds: int = 3,
    covariance_type: str = "full",
    model_type: str = "gmm",
    rng: np.random.Generator | None = None,
    **fit_kwargs: Any,
) -> float:
    rng = rng or np.random.default_rng()
    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n = y.shape[0]
    folds = max(2, int(n_folds))
    idx = rng.permutation(n)
    fold_sizes = np.full(folds, n // folds)
    fold_sizes[: n % folds] += 1
    scores = []
    start = 0
    for fs in fold_sizes:
        test_idx = idx[start : start + fs]
        train_idx = np.concatenate([idx[:start], idx[start + fs :]])
        start += fs
        if train_idx.size < n_components or test_idx.size == 0:
            continue
        result = fit_em(
            y[train_idx],
            n_components,
            model_type=model_type,  # type: ignore[arg-type]
            covariance_type=covariance_type,  # type: ignore[arg-type]
            n_restarts=1,
            n_jobs=1,
            rng=rng,
            **fit_kwargs,
        )
        ll = log_likelihood(
            y[test_idx],
            result.weights,
            result.means,
            result.covars,
            covariance_type=result.covariance_type,
        )
        scores.append(ll / max(test_idx.size, 1))
    return float(np.mean(scores)) if scores else float("-inf")


def select_n_components(
    x: np.ndarray,
    *,
    min_components: int = 1,
    max_components: int = 5,
    criterion: Criterion = "bic",
    cv_folds: int = 3,
    preprocess: bool = True,
    rng: np.random.Generator | None = None,
    **fit_kwargs: Any,
) -> dict[str, Any]:
    rng = rng or np.random.default_rng()
    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if preprocess:
        y, _ = fit_preprocess(y, standardize=True)
    rows = []
    best = None
    best_k = int(min_components)
    minimize = criterion in ("aic", "bic", "icl")
    best_score = np.inf if minimize else -np.inf

    for k in range(int(min_components), int(max_components) + 1):
        result = fit_em(y, k, n_restarts=1, n_jobs=1, rng=rng, **fit_kwargs)
        scores = score_result(result, y.shape[0])
        if criterion == "cv":
            cv_score = cross_validate_ll(
                y,
                k,
                n_folds=cv_folds,
                covariance_type=fit_kwargs.get("covariance_type", "full"),
                model_type=fit_kwargs.get("model_type", "gmm"),
                rng=rng,
                max_iter=fit_kwargs.get("max_iter", 50),
                tol=fit_kwargs.get("tol", 1e-3),
                reg_covar=fit_kwargs.get("reg_covar", 1e-6),
            )
            scores["cv"] = cv_score
            score = cv_score
        else:
            score = float(scores[criterion])
        rows.append({"n_components": k, **scores, "converged": result.converged})
        better = score < best_score if minimize else score > best_score
        if better:
            best_score = score
            best = result
            best_k = k
    return {
        "best_n_components": best_k,
        "criterion": criterion,
        "candidates": rows,
        "best_result": best,
    }

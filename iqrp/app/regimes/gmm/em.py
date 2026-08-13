"""Expectation-Maximization and Bayesian variational EM for GMMs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from iqrp.app.regimes.gmm.covariance import CovarianceType
from iqrp.app.regimes.gmm.expectation import e_step
from iqrp.app.regimes.gmm.initialization import initialize_parameters
from iqrp.app.regimes.gmm.maximization import bayesian_m_step, m_step

ModelType = Literal["gmm", "bayesian_gmm"]


@dataclass
class EMResult:
    weights: np.ndarray
    means: np.ndarray
    covars: np.ndarray
    responsibilities: np.ndarray
    log_likelihood: float
    history: list[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False
    covariance_type: CovarianceType = "full"
    model_type: ModelType = "gmm"
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_em(
    x: np.ndarray,
    n_components: int,
    *,
    model_type: ModelType = "gmm",
    covariance_type: CovarianceType = "full",
    init_method: str = "kmeans",
    max_iter: int = 100,
    tol: float = 1e-4,
    early_stopping: bool = True,
    reg_covar: float = 1e-6,
    n_restarts: int = 1,
    n_jobs: int = 1,
    warm_start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    bayesian_params: dict[str, Any] | None = None,
    user_params: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> EMResult:
    """Fit classical or Bayesian GMM via EM with optional restarts."""
    rng = rng or np.random.default_rng()
    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n_restarts = max(1, int(n_restarts))
    bayesian_params = bayesian_params or {}

    if warm_start is not None:
        return _fit_once(
            y,
            warm_start[0],
            warm_start[1],
            warm_start[2],
            model_type=model_type,
            covariance_type=covariance_type,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            reg_covar=reg_covar,
            bayesian_params=bayesian_params,
        )

    seeds = rng.integers(0, 2**31 - 1, size=n_restarts)

    def _run(seed: int) -> EMResult:
        local = np.random.default_rng(int(seed))
        w, m, c = initialize_parameters(
            y,
            n_components,
            method=init_method,  # type: ignore[arg-type]
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            user_params=user_params,
            rng=local,
        )
        return _fit_once(
            y,
            w,
            m,
            c,
            model_type=model_type,
            covariance_type=covariance_type,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            reg_covar=reg_covar,
            bayesian_params=bayesian_params,
        )

    results: list[EMResult] = []
    n_jobs_use = max(1, int(n_jobs))
    if n_jobs_use == 1 or n_restarts == 1:
        results = [_run(int(s)) for s in seeds]
    else:
        with ThreadPoolExecutor(max_workers=min(n_jobs_use, n_restarts)) as pool:
            futs = [pool.submit(_run, int(s)) for s in seeds]
            for fut in as_completed(futs):
                results.append(fut.result())
    best = max(results, key=lambda r: r.log_likelihood)
    return best


def _fit_once(
    y: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covars: np.ndarray,
    *,
    model_type: ModelType,
    covariance_type: CovarianceType,
    max_iter: int,
    tol: float,
    early_stopping: bool,
    reg_covar: float,
    bayesian_params: dict[str, Any],
) -> EMResult:
    history: list[float] = []
    resp = np.full((y.shape[0], means.shape[0]), 1.0 / means.shape[0])
    prev = -np.inf
    converged = False
    n_iter = 0
    ll = -np.inf
    for it in range(max_iter):
        resp, avg_ll = e_step(y, weights, means, covars, covariance_type=covariance_type)
        ll = float(avg_ll * y.shape[0])
        history.append(ll)
        if model_type == "bayesian_gmm":
            weights, means, covars = bayesian_m_step(
                y,
                resp,
                covariance_type=covariance_type,
                reg_covar=reg_covar,
                weight_concentration_prior=float(
                    bayesian_params.get("weight_concentration_prior", 1.0)
                ),
                mean_precision_prior=float(bayesian_params.get("mean_precision_prior", 1.0)),
                covariance_prior_scale=float(bayesian_params.get("covariance_prior_scale", 1.0)),
            )
        else:
            weights, means, covars = m_step(
                y, resp, covariance_type=covariance_type, reg_covar=reg_covar
            )
        n_iter = it + 1
        if early_stopping and it > 0 and abs(ll - prev) < tol * max(1.0, abs(prev)):
            converged = True
            break
        prev = ll
    return EMResult(
        weights=weights,
        means=means,
        covars=covars,
        responsibilities=resp,
        log_likelihood=ll,
        history=history,
        n_iter=n_iter,
        converged=converged,
        covariance_type=covariance_type,
        model_type=model_type,
    )

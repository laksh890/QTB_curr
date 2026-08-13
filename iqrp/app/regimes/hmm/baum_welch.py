"""Baum-Welch (EM) training for HMMs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

from iqrp.app.regimes.hmm.emissions import EmissionModel
from iqrp.app.regimes.hmm.forward_backward import forward_backward
from iqrp.app.regimes.hmm.initialization import initialize_parameters
from iqrp.app.regimes.hmm.transitions import HMMTransitions


@dataclass
class BaumWelchResult:
    transitions: HMMTransitions
    emissions: EmissionModel
    log_likelihood: float
    history: list[float] = field(default_factory=list)
    n_iter: int = 0
    converged: bool = False


def baum_welch(
    observations: np.ndarray,
    n_states: int,
    *,
    emission_type: str = "gaussian",
    covariance_type: str = "diag",
    n_symbols: int | None = None,
    method: str = "kmeans",
    max_iter: int = 100,
    tol: float = 1e-4,
    early_stopping: bool = True,
    min_covar: float = 1e-6,
    dirichlet_alpha: float = 1.0,
    n_restarts: int = 1,
    n_jobs: int = 1,
    warm_start: tuple[HMMTransitions, EmissionModel] | None = None,
    rng: np.random.Generator | None = None,
) -> BaumWelchResult:
    """Fit HMM via EM with optional multiple random restarts."""
    rng = rng or np.random.default_rng()
    n_restarts = max(1, int(n_restarts))
    if warm_start is not None:
        return _fit_once(
            observations,
            warm_start[0],
            warm_start[1],
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            min_covar=min_covar,
        )

    seeds = rng.integers(0, 2**31 - 1, size=n_restarts)

    def _run(seed: int) -> BaumWelchResult:
        local = np.random.default_rng(int(seed))
        trans, emis = initialize_parameters(
            observations,
            n_states,
            method=method,  # type: ignore[arg-type]
            emission_type=emission_type,
            covariance_type=covariance_type,
            n_symbols=n_symbols,
            dirichlet_alpha=dirichlet_alpha,
            rng=local,
            min_covar=min_covar,
        )
        return _fit_once(
            observations,
            trans,
            emis,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            min_covar=min_covar,
        )

    results: list[BaumWelchResult] = []
    if n_jobs == 1 or n_restarts == 1:
        results = [_run(int(s)) for s in seeds]
    else:
        with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as pool:
            futs = [pool.submit(_run, int(s)) for s in seeds]
            for fut in as_completed(futs):
                results.append(fut.result())
    return max(results, key=lambda r: r.log_likelihood)


def _fit_once(
    observations: np.ndarray,
    transitions: HMMTransitions,
    emissions: EmissionModel,
    *,
    max_iter: int,
    tol: float,
    early_stopping: bool,
    min_covar: float,
) -> BaumWelchResult:
    history: list[float] = []
    prev = -np.inf
    converged = False
    n_iter = 0
    for it in range(max_iter):
        n_iter = it + 1
        log_e = emissions.log_prob(observations)
        fb = forward_backward(log_e, transitions.transition, initial=transitions.initial)
        history.append(fb.log_likelihood)
        transitions.m_step(fb.xi, fb.gamma)
        emissions.m_step(observations, fb.gamma, min_covar=min_covar)
        if early_stopping and abs(fb.log_likelihood - prev) < tol:
            converged = True
            break
        prev = fb.log_likelihood
    final_ll = history[-1] if history else float("nan")
    return BaumWelchResult(
        transitions=transitions,
        emissions=emissions,
        log_likelihood=final_ll,
        history=history,
        n_iter=n_iter,
        converged=converged,
    )


def em_step(
    observations: np.ndarray,
    transitions: HMMTransitions,
    emissions: EmissionModel,
    *,
    min_covar: float = 1e-6,
) -> float:
    """Single EM iteration; returns current log-likelihood after E-step."""
    log_e = emissions.log_prob(observations)
    fb = forward_backward(log_e, transitions.transition, initial=transitions.initial)
    transitions.m_step(fb.xi, fb.gamma)
    emissions.m_step(observations, fb.gamma, min_covar=min_covar)
    return fb.log_likelihood

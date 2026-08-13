"""Gibbs sampling for Bayesian HMM / Markov-switching models."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from iqrp.app.regimes.bayesian.emissions import BayesianEmissions, CovarianceType
from iqrp.app.regimes.bayesian.inference import ffbs, log_joint
from iqrp.app.regimes.bayesian.posterior import ParameterDraw, Posterior
from iqrp.app.regimes.bayesian.priors import ModelPriors
from iqrp.app.regimes.bayesian.transitions import BayesianTransitions


@dataclass
class GibbsResult:
    posterior: Posterior
    acceptance_rate: float = 1.0
    history: list[float] = field(default_factory=list)
    n_iter: int = 0


def run_gibbs(
    observations: np.ndarray,
    n_states: int,
    priors: ModelPriors,
    *,
    covariance_type: CovarianceType = "diag",
    n_chains: int = 2,
    n_samples: int = 200,
    burn_in: int = 50,
    thin: int = 1,
    n_jobs: int = 1,
    checkpoint_every: int = 0,
    checkpoint_dir: Path | None = None,
    warm_start: tuple[BayesianTransitions, BayesianEmissions, np.ndarray] | None = None,
    rng: np.random.Generator | None = None,
) -> GibbsResult:
    """Parallel multi-chain Gibbs sampler with FFBS latent updates."""
    rng = rng or np.random.default_rng()
    y = np.asarray(observations, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    n_features = y.shape[1]
    seeds = rng.integers(0, 2**31 - 1, size=max(1, n_chains))

    def _chain(seed: int, chain_id: int) -> tuple[list[ParameterDraw], list[float]]:
        local = np.random.default_rng(int(seed))
        if warm_start is not None and chain_id == 0:
            trans, emis, states = warm_start
            trans = BayesianTransitions(
                trans.n_states,
                trans.transition.copy(),
                trans.initial.copy(),
                trans.prior_alpha,
                trans.prior_initial,
            )
            emis = BayesianEmissions(
                emis.n_states,
                emis.n_features,
                emis.means.copy(),
                emis.covars.copy(),
                emis.covariance_type,
                emis.priors,
            )
            states = states.copy()
        else:
            trans = BayesianTransitions.from_priors(priors, rng=local)
            emis = BayesianEmissions.from_priors(
                priors, n_states, n_features, covariance_type=covariance_type, rng=local
            )
            log_e = emis.log_prob(y)
            states, _ = ffbs(log_e, trans.transition, trans.initial, rng=local)
        draws: list[ParameterDraw] = []
        hist: list[float] = []
        total = burn_in + n_samples * max(1, thin)
        for it in range(total):
            log_e = emis.log_prob(y)
            states, _ = ffbs(log_e, trans.transition, trans.initial, rng=local)
            trans = trans.sample_posterior(states, rng=local)
            emis = emis.sample_posterior(y, states, rng=local)
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
                        chain_id=chain_id,
                    )
                )
            if (
                checkpoint_every > 0
                and checkpoint_dir is not None
                and (it + 1) % checkpoint_every == 0
            ):
                _write_checkpoint(checkpoint_dir, chain_id, it, draws, hist)
        return draws, hist

    all_draws: list[ParameterDraw] = []
    all_hist: list[float] = []
    n_jobs_use = max(1, int(n_jobs))
    if n_jobs_use == 1 or n_chains <= 1:
        for cid, seed in enumerate(seeds):
            d, h = _chain(int(seed), cid)
            all_draws.extend(d)
            all_hist.extend(h)
    else:
        with ThreadPoolExecutor(max_workers=min(n_jobs_use, len(seeds))) as pool:
            futs = {pool.submit(_chain, int(seed), cid): cid for cid, seed in enumerate(seeds)}
            for fut in as_completed(futs):
                d, h = fut.result()
                all_draws.extend(d)
                all_hist.extend(h)

    posterior = Posterior(
        draws=all_draws,
        burn_in=burn_in,
        thin=thin,
        n_chains=len(seeds),
        algorithm="gibbs",
        metadata={"n_features": n_features, "n_states": n_states},
    )
    return GibbsResult(
        posterior=posterior,
        acceptance_rate=1.0,
        history=all_hist,
        n_iter=burn_in + n_samples * max(1, thin),
    )


def _write_checkpoint(
    directory: Path,
    chain_id: int,
    iteration: int,
    draws: list[ParameterDraw],
    history: list[float],
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"gibbs_chain{chain_id}_iter{iteration}.npz"
    payload: dict[str, Any] = {"history": np.asarray(history, dtype=np.float64)}
    if draws:
        payload["last_means"] = draws[-1].means
        payload["last_transition"] = draws[-1].transition
    np.savez_compressed(path, **payload)

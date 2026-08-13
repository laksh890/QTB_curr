"""Training orchestration and model selection for HMMs."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math.probability.likelihood import aic, bic
from iqrp.app.regimes.hmm.baum_welch import BaumWelchResult, baum_welch
from iqrp.app.regimes.hmm.config import HMMSettings


class HMMTrainer:
    def __init__(self, settings: HMMSettings | None = None) -> None:
        self.settings = settings or HMMSettings.default()
        self.history: list[dict[str, Any]] = []

    def train(
        self,
        observations: np.ndarray,
        *,
        n_states: int | None = None,
        warm_start: tuple[Any, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> BaumWelchResult:
        s = self.settings
        k = int(n_states if n_states is not None else s.n_states)
        result = baum_welch(
            observations,
            k,
            emission_type=(
                s.emission.type if s.emission.type != "multivariate_gaussian" else "gaussian"
            ),
            covariance_type=s.emission.covariance_type,
            method=s.initialization.method,
            max_iter=s.training.max_iter,
            tol=s.training.tol,
            early_stopping=s.training.early_stopping,
            min_covar=s.training.min_covar,
            dirichlet_alpha=s.initialization.dirichlet_alpha,
            n_restarts=s.initialization.n_restarts,
            n_jobs=s.training.n_jobs,
            warm_start=warm_start,
            rng=rng,
        )
        self.history.append(
            {
                "n_states": k,
                "log_likelihood": result.log_likelihood,
                "n_iter": result.n_iter,
                "converged": result.converged,
            }
        )
        return result

    def select_n_states(
        self,
        observations: np.ndarray,
        *,
        min_states: int | None = None,
        max_states: int | None = None,
        criterion: str | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        s = self.settings
        lo = int(min_states if min_states is not None else s.model_selection.min_states)
        hi = int(max_states if max_states is not None else s.model_selection.max_states)
        crit = criterion or s.model_selection.criterion
        rows: list[dict[str, Any]] = []
        best: BaumWelchResult | None = None
        best_score = float("inf")
        best_k = lo
        n = max(int(np.asarray(observations).shape[0]), 1)
        for k in range(lo, hi + 1):
            result = self.train(observations, n_states=k, rng=rng)
            n_params = _n_params(k, result.emissions)
            nll = -result.log_likelihood
            scores = {
                "log_likelihood": result.log_likelihood,
                "aic": aic(nll, n_params),
                "bic": bic(nll, n_params, n),
            }
            score = scores.get(crit, scores["bic"])
            # for LL, higher is better
            rank_score = -scores["log_likelihood"] if crit == "log_likelihood" else score
            rows.append(
                {"n_states": k, **scores, "n_params": n_params, "converged": result.converged}
            )
            if rank_score < best_score:
                best_score = float(rank_score)
                best = result
                best_k = k
        return {
            "best_n_states": best_k,
            "criterion": crit,
            "candidates": rows,
            "best_result": best,
        }


def _n_params(n_states: int, emissions: Any) -> int:
    # transitions: K*(K-1) + initial K-1
    base = n_states * (n_states - 1) + (n_states - 1)
    raw: object = getattr(emissions, "to_dict", lambda: {})()
    payload: dict[str, object] = raw if isinstance(raw, dict) else {}
    kind = str(payload.get("type", "gaussian"))
    if kind == "discrete":
        return base + n_states * (getattr(emissions, "n_symbols", 2) - 1)
    d = int(getattr(emissions, "n_features", 1))
    cov_type = getattr(emissions, "covariance_type", "diag")
    means = n_states * d
    cov = n_states * d if cov_type == "diag" else n_states * d * (d + 1) // 2
    return base + means + cov

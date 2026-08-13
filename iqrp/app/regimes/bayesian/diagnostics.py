"""Diagnostics for Bayesian regime-switching fits."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math.statistics.entropy import entropy
from iqrp.app.regimes.bayesian.convergence import convergence_report
from iqrp.app.regimes.bayesian.posterior import Posterior


class BayesianDiagnostics:
    def report(
        self,
        posterior: Posterior,
        *,
        history: list[float] | None = None,
        acceptance_rate: float | None = None,
        state_proba: np.ndarray | None = None,
    ) -> dict[str, Any]:
        traces = _build_traces(posterior)
        conv = convergence_report(traces, acceptance_rate=acceptance_rate)
        means_ci = posterior.credible_intervals("means")
        trans_ci = posterior.credible_intervals("transition")
        occupancy = posterior.state_occupancy()
        rare = [i for i, o in enumerate(occupancy) if o < 0.05]
        state_entropy = None
        if state_proba is not None and np.asarray(state_proba).ndim == 2:
            state_entropy = float(np.mean([entropy(row) for row in state_proba]))
        return {
            "convergence": conv,
            "history": list(history or []),
            "posterior_means": posterior.mean_means(),
            "posterior_covars": posterior.mean_covars(),
            "posterior_transition": posterior.mean_transition(),
            "means_credible": {
                "low": means_ci["low"],
                "high": means_ci["high"],
                "level": means_ci["level"],
            },
            "transition_credible": {
                "low": trans_ci["low"],
                "high": trans_ci["high"],
                "level": trans_ci["level"],
            },
            "state_occupancy": occupancy,
            "rare_states": rare,
            "mean_state_entropy": state_entropy,
            "n_draws": posterior.n_draws,
            "algorithm": posterior.algorithm,
        }


def _build_traces(posterior: Posterior) -> dict[str, list[np.ndarray]]:
    by_chain: dict[int, list[Any]] = {}
    for d in posterior.draws:
        by_chain.setdefault(d.chain_id, []).append(d)
    traces: dict[str, list[np.ndarray]] = {
        "log_joint": [],
        "mean_0": [],
        "persist_0": [],
    }
    if not by_chain:
        return traces
    for _cid, draws in sorted(by_chain.items()):
        traces["log_joint"].append(np.array([d.log_joint for d in draws], dtype=np.float64))
        traces["mean_0"].append(np.array([float(d.means.reshape(-1)[0]) for d in draws]))
        traces["persist_0"].append(np.array([float(d.transition[0, 0]) for d in draws]))
    return traces

"""Evaluation metrics for Bayesian regime-switching models."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math.statistics.entropy import cross_entropy
from iqrp.app.regimes.bayesian.posterior import Posterior
from iqrp.app.regimes.bayesian.trainer import model_comparison_scores
from iqrp.app.state_space.evaluation.diagnostics import probability_calibration
from iqrp.app.state_space.evaluation.metrics import state_stability, transition_accuracy


class BayesianEvaluator:
    def evaluate(
        self,
        *,
        true_states: np.ndarray | None,
        predicted_states: np.ndarray,
        probabilities: np.ndarray,
        posterior: Posterior,
        observations: np.ndarray,
    ) -> dict[str, Any]:
        pred = np.asarray(predicted_states, dtype=np.int64).reshape(-1)
        proba = np.asarray(probabilities, dtype=np.float64)
        scores = model_comparison_scores(observations, posterior)
        metrics: dict[str, float] = {
            "state_stability": float(state_stability(pred)),
            "mean_max_probability": (float(np.mean(proba.max(axis=1))) if proba.ndim == 2 else 0.0),
            "waic": float(scores["waic"]),
            "loo": float(scores["loo"]),
            "marginal_likelihood": float(scores["marginal_likelihood"]),
            "n_posterior_draws": float(posterior.n_draws),
        }
        details: dict[str, Any] = {"comparison": scores}
        if true_states is not None:
            truth = np.asarray(true_states, dtype=np.int64).reshape(-1)
            k = int(proba.shape[1]) if proba.ndim == 2 else int(truth.max()) + 1
            metrics["prediction_accuracy"] = _best_accuracy(pred, truth, k)
            metrics["transition_accuracy"] = float(transition_accuracy(pred, truth))
            metrics["cross_entropy"] = float(_avg_ce(proba, truth))
            details["calibration"] = probability_calibration(proba, truth)
            # credible interval coverage for recovered means if available
            if posterior.draws:
                means_ci = posterior.credible_intervals("means", level=0.95)
                details["means_credible"] = {
                    "low": means_ci["low"].tolist(),
                    "high": means_ci["high"].tolist(),
                }
        return {"metrics": metrics, "details": details}


def _avg_ce(proba: np.ndarray, truth: np.ndarray) -> float:
    if proba.ndim != 2 or proba.shape[0] != truth.size:
        return float("nan")
    total = 0.0
    for t, s in enumerate(truth):
        q = np.zeros(proba.shape[1])
        if 0 <= int(s) < len(q):
            q[int(s)] = 1.0
        total += float(cross_entropy(q, proba[t]))
    return total / max(len(truth), 1)


def _best_accuracy(pred: np.ndarray, truth: np.ndarray, n_states: int) -> float:
    from itertools import permutations

    k = min(n_states, 6)
    if n_states > 6:
        conf = np.zeros((n_states, n_states), dtype=np.float64)
        for a, b in zip(pred, truth, strict=False):
            if 0 <= a < n_states and 0 <= b < n_states:
                conf[a, b] += 1.0
        mapping = {i: int(np.argmax(conf[i])) for i in range(n_states)}
        mapped = np.array([mapping.get(int(x), int(x)) for x in pred])
        return float(np.mean(mapped == truth))
    best = 0.0
    for perm in permutations(range(k)):
        mapped = np.array([perm[int(x)] if 0 <= int(x) < k else int(x) for x in pred])
        best = max(best, float(np.mean(mapped == truth)))
    return best

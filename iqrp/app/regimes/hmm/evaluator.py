"""Evaluation metrics for fitted HMMs."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math.probability.likelihood import aic, bic
from iqrp.app.math.statistics.entropy import cross_entropy
from iqrp.app.regimes.hmm.trainer import _n_params
from iqrp.app.state_space.evaluation.diagnostics import probability_calibration
from iqrp.app.state_space.evaluation.metrics import state_stability, transition_accuracy


class HMMEvaluator:
    def evaluate(
        self,
        *,
        true_states: np.ndarray | None,
        predicted_states: np.ndarray,
        probabilities: np.ndarray,
        log_likelihood: float,
        emissions: Any,
        n_samples: int,
    ) -> dict[str, Any]:
        pred = np.asarray(predicted_states, dtype=np.int64).reshape(-1)
        proba = np.asarray(probabilities, dtype=np.float64)
        n_params = _n_params(proba.shape[1] if proba.ndim == 2 else 1, emissions)
        nll = float(-log_likelihood)
        metrics: dict[str, float] = {
            "log_likelihood": float(log_likelihood),
            "negative_log_likelihood": nll,
            "aic": aic(nll, n_params),
            "bic": bic(nll, n_params, max(n_samples, 1)),
            "state_stability": float(state_stability(pred)),
            "mean_max_probability": (float(np.mean(proba.max(axis=1))) if proba.ndim == 2 else 0.0),
        }
        details: dict[str, Any] = {"n_params": n_params, "n_samples": n_samples}
        if true_states is not None:
            truth = np.asarray(true_states, dtype=np.int64).reshape(-1)
            # allow label switching via best permutation for small K
            acc = _best_accuracy(
                pred, truth, int(proba.shape[1]) if proba.ndim == 2 else int(truth.max()) + 1
            )
            metrics["prediction_accuracy"] = acc
            metrics["transition_accuracy"] = float(transition_accuracy(pred, truth))
            metrics["cross_entropy"] = float(_avg_ce(proba, truth))
            details["calibration"] = probability_calibration(proba, truth)
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
        # greedy matching by confusion matrix
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

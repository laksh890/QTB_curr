"""Evaluation metrics for Markov chain models."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np

from iqrp.app.math.probability.likelihood import aic, bic
from iqrp.app.math.statistics.entropy import cross_entropy
from iqrp.app.state_space.evaluation.diagnostics import probability_calibration
from iqrp.app.state_space.evaluation.metrics import (
    persistence_stability,
    state_stability,
    transition_accuracy,
)


class MarkovEvaluator:
    def evaluate(
        self,
        *,
        true_states: np.ndarray,
        predicted_states: np.ndarray,
        probabilities: np.ndarray,
        transition: np.ndarray,
        log_likelihood: float,
        n_params: int,
        forecast_true: np.ndarray | None = None,
        forecast_pred: np.ndarray | None = None,
    ) -> dict[str, Any]:
        truth = np.asarray(true_states, dtype=np.int64).reshape(-1)
        pred = np.asarray(predicted_states, dtype=np.int64).reshape(-1)
        proba = np.asarray(probabilities, dtype=np.float64)
        n = int(truth.size)
        nll = float(-log_likelihood)
        metrics: dict[str, float] = {
            "log_likelihood": float(log_likelihood),
            "negative_log_likelihood": nll,
            "aic": aic(nll, n_params),
            "bic": bic(nll, n_params, max(n - 1, 1)),
            "prediction_accuracy": float(np.mean(pred == truth)) if n else 0.0,
            "transition_accuracy": float(transition_accuracy(pred, truth)),
            "cross_entropy": float(_avg_cross_entropy(proba, truth)),
            "state_stability": float(state_stability(pred)),
            "persistence_stability": float(persistence_stability(pred)),
        }
        if forecast_true is not None and forecast_pred is not None:
            ft = np.asarray(forecast_true).reshape(-1)
            fp = np.asarray(forecast_pred).reshape(-1)
            m = min(ft.size, fp.size)
            metrics["forecast_accuracy"] = float(np.mean(ft[:m] == fp[:m])) if m else 0.0
        cal = probability_calibration(proba, truth)
        return {
            "metrics": metrics,
            "calibration": cal,
            "n_params": n_params,
            "n_samples": n,
            "transition_diagonal_mean": float(np.mean(np.diag(transition))),
        }


def _avg_cross_entropy(probabilities: np.ndarray, true_states: np.ndarray) -> float:
    proba = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(true_states, dtype=np.int64).reshape(-1)
    if proba.ndim != 2 or proba.shape[0] != truth.size:
        # one-step-ahead CE from consecutive pairs using rows as predictive
        if proba.ndim == 2 and truth.size >= 2:
            total = 0.0
            for t in range(truth.size - 1):
                q = np.zeros(proba.shape[1])
                j = int(truth[t + 1])
                if 0 <= j < len(q):
                    q[j] = 1.0
                i = int(truth[t])
                p = (
                    proba[i]
                    if 0 <= i < proba.shape[0]
                    else np.full(proba.shape[1], 1.0 / proba.shape[1])
                )
                total += float(cross_entropy(q, p))
            return total / max(truth.size - 1, 1)
        return float("nan")
    total = 0.0
    for t, s in enumerate(truth):
        q = np.zeros(proba.shape[1])
        if 0 <= int(s) < len(q):
            q[int(s)] = 1.0
        total += float(cross_entropy(q, proba[t]))
    return total / max(len(truth), 1)


def next_state_accuracy(states: np.ndarray, transition: np.ndarray) -> float:
    """Accuracy of argmax next-state prediction from ``P``."""
    s = np.asarray(states, dtype=np.int64).reshape(-1)
    p = np.asarray(transition, dtype=np.float64)
    if s.size < 2:
        return 1.0
    correct = 0
    for a, b in pairwise(s):
        if 0 <= a < p.shape[0] and int(np.argmax(p[a])) == int(b):
            correct += 1
    return float(correct / (s.size - 1))

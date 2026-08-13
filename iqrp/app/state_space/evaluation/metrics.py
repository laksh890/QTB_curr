"""Evaluation metrics for latent-state models (math-engine backed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np

from iqrp.app.math.probability.likelihood import aic, bic
from iqrp.app.math.statistics.entropy import cross_entropy, entropy as shannon_entropy


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    metrics: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"metrics": dict(self.metrics), "details": dict(self.details)}


class EvaluationMetrics:
    """Institutional metrics for discrete state-space models."""

    def evaluate(
        self,
        *,
        predicted: np.ndarray,
        probabilities: np.ndarray,
        log_likelihood: float,
        n_params: int,
        n_samples: int,
        true_states: np.ndarray | None = None,
        transition_matrix: np.ndarray | None = None,
    ) -> dict[str, Any]:
        pred = np.asarray(predicted, dtype=np.int64).reshape(-1)
        proba = np.asarray(probabilities, dtype=np.float64)
        nll = float(-log_likelihood)
        metrics: dict[str, float] = {
            "log_likelihood": float(log_likelihood),
            "negative_log_likelihood": nll,
            "aic": aic(nll, n_params),
            "bic": bic(nll, n_params, n_samples),
            "perplexity": float(np.exp(-log_likelihood / max(n_samples, 1))),
            "state_stability": float(state_stability(pred)),
            "persistence_stability": float(persistence_stability(pred)),
            "mean_max_probability": (
                float(np.mean(proba.max(axis=1))) if proba.ndim == 2 else float(np.max(proba))
            ),
            "mean_entropy": float(_mean_entropy(proba)),
        }
        details: dict[str, Any] = {"n_params": n_params, "n_samples": n_samples}

        if true_states is not None:
            truth = np.asarray(true_states, dtype=np.int64).reshape(-1)
            metrics["state_prediction_accuracy"] = float(np.mean(pred == truth))
            metrics["transition_accuracy"] = float(transition_accuracy(pred, truth))
            metrics["cross_entropy"] = float(sequence_cross_entropy(proba, truth))

        if transition_matrix is not None:
            tm = np.asarray(transition_matrix, dtype=np.float64)
            diag = np.diag(tm)
            metrics["mean_self_transition"] = float(np.mean(diag))
            details["expected_durations"] = {
                str(i): float(1.0 / max(1.0 - float(diag[i]), 1e-12)) for i in range(len(diag))
            }

        report = EvaluationReport(metrics=metrics, details=details)
        return report.to_dict()


def state_stability(states: np.ndarray) -> float:
    """Fraction of consecutive equal states (persistence rate)."""
    s = np.asarray(states).reshape(-1)
    if s.size <= 1:
        return 1.0
    return float(np.mean(s[1:] == s[:-1]))


def persistence_stability(states: np.ndarray) -> float:
    """Normalized mean run length relative to series length."""
    s = np.asarray(states).reshape(-1)
    if s.size == 0:
        return 0.0
    runs = 1
    for a, b in pairwise(s):
        if a != b:
            runs += 1
    mean_run = s.size / runs
    return float(min(1.0, mean_run / max(s.size, 1)))


def transition_accuracy(predicted: np.ndarray, true_states: np.ndarray) -> float:
    pred = np.asarray(predicted).reshape(-1)
    truth = np.asarray(true_states).reshape(-1)
    if pred.size < 2:
        return 1.0
    correct = 0
    total = 0
    for i in range(1, len(pred)):
        total += 1
        if (pred[i - 1], pred[i]) == (truth[i - 1], truth[i]):
            correct += 1
    return float(correct / max(total, 1))


def sequence_cross_entropy(probabilities: np.ndarray, true_states: np.ndarray) -> float:
    proba = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(true_states, dtype=np.int64).reshape(-1)
    if proba.ndim != 2 or proba.shape[0] != truth.size:
        return float("nan")
    # Empirical one-hot vs predictive rows — average CE
    total = 0.0
    for t, s in enumerate(truth):
        p = proba[t]
        q = np.zeros_like(p)
        if 0 <= int(s) < len(q):
            q[int(s)] = 1.0
        else:
            q[:] = 1.0 / len(q)
        total += float(cross_entropy(q, p))
    return float(total / max(len(truth), 1))


def _mean_entropy(proba: np.ndarray) -> float:
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim == 1:
        return float(shannon_entropy(p))
    return float(np.mean([shannon_entropy(row) for row in p]))

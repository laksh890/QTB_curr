"""Regime model evaluation metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from iqrp.app.regimes.base.persistence import PersistenceEngine


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    prediction_accuracy: float
    transition_accuracy: float
    log_likelihood: float
    cross_entropy: float
    state_stability: float
    persistence_stability: float
    n_samples: int
    n_states: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RegimeEvaluator:
    """Evaluate predicted regimes against optional ground truth and self-consistency."""

    def evaluate(
        self,
        *,
        predicted: np.ndarray,
        probabilities: np.ndarray,
        transition_matrix: np.ndarray,
        true_states: np.ndarray | None = None,
    ) -> EvaluationReport:
        pred = np.asarray(predicted, dtype=np.int64)
        proba = np.asarray(probabilities, dtype=np.float64)
        tm = np.asarray(transition_matrix, dtype=np.float64)
        n = len(pred)
        k = proba.shape[1] if proba.ndim == 2 else len(proba)

        if true_states is not None:
            truth = np.asarray(true_states, dtype=np.int64)
            m = min(len(truth), n)
            prediction_accuracy = float(np.mean(pred[:m] == truth[:m])) if m else float("nan")
            # Transition accuracy on true vs predicted change events
            if m > 1:
                true_chg = truth[1:m] != truth[: m - 1]
                pred_chg = pred[1:m] != pred[: m - 1]
                transition_accuracy = float(np.mean(true_chg == pred_chg))
            else:
                transition_accuracy = float("nan")
            ll = _log_likelihood(proba[:m], truth[:m])
            ce = _cross_entropy(proba[:m], truth[:m])
        else:
            # Unsupervised proxies
            prediction_accuracy = float(np.mean(np.max(proba, axis=1))) if n else float("nan")
            transition_accuracy = _self_transition_consistency(pred, tm)
            ll = _log_likelihood(proba, pred)
            ce = _cross_entropy(proba, pred)

        # Stability: average run length / n
        durations = PersistenceEngine.run_lengths(pred)
        all_durs = [d for v in durations.values() for d in v]
        state_stability = float(np.mean(all_durs) / max(n, 1)) if all_durs else 0.0
        persist_scores = list(PersistenceEngine.persistence_score(tm).values())
        persistence_stability = float(np.mean(persist_scores)) if persist_scores else 0.0

        return EvaluationReport(
            prediction_accuracy=prediction_accuracy,
            transition_accuracy=transition_accuracy,
            log_likelihood=ll,
            cross_entropy=ce,
            state_stability=state_stability,
            persistence_stability=persistence_stability,
            n_samples=n,
            n_states=k,
        )


def _log_likelihood(proba: np.ndarray, states: np.ndarray) -> float:
    if len(states) == 0:
        return float("nan")
    p = np.asarray(proba, dtype=np.float64)
    s = np.asarray(states, dtype=np.int64)
    if p.ndim == 1:
        return float(np.log(np.clip(p[s], 1e-12, 1.0)).mean())
    rows = np.arange(len(s))
    return float(np.log(np.clip(p[rows, s], 1e-12, 1.0)).mean())


def _cross_entropy(proba: np.ndarray, states: np.ndarray) -> float:
    return float(-_log_likelihood(proba, states))


def _self_transition_consistency(pred: np.ndarray, tm: np.ndarray) -> float:
    if len(pred) < 2:
        return float("nan")
    scores = []
    for i in range(1, len(pred)):
        a, b = int(pred[i - 1]), int(pred[i])
        scores.append(float(tm[a, b]))
    return float(np.mean(scores)) if scores else float("nan")

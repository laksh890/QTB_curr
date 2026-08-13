"""Training orchestration for Markov chain models."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from iqrp.app.regimes.markov.config import MarkovSettings
from iqrp.app.regimes.markov.diagnostics import MarkovDiagnostics
from iqrp.app.regimes.markov.evaluator import MarkovEvaluator


class MarkovTrainer:
    """Fit / partial-fit helper with training statistics."""

    def __init__(self, settings: MarkovSettings | None = None) -> None:
        self.settings = settings or MarkovSettings.default()
        self.history: list[dict[str, Any]] = []

    def train(
        self,
        model: Any,
        states: np.ndarray | pl.DataFrame,
        *,
        weights: np.ndarray | None = None,
        true_states: np.ndarray | None = None,
    ) -> dict[str, Any]:
        model.fit(states, weights=weights)
        stats = self._stats(model, states, true_states=true_states)
        self.history.append(stats)
        return stats

    def partial_train(
        self,
        model: Any,
        states: np.ndarray | pl.DataFrame,
        *,
        weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        model.partial_fit(states, weights=weights)
        stats = {
            "n_transitions": int(model.estimator.matrix.n_transitions),
            "log_likelihood": float(model.log_likelihood(states)),
            "mode": "partial_fit",
        }
        self.history.append(stats)
        return stats

    def _stats(
        self,
        model: Any,
        states: np.ndarray | pl.DataFrame,
        *,
        true_states: np.ndarray | None,
    ) -> dict[str, Any]:
        s = model._extract_states(states)
        proba = model.predict_proba(states)
        pred = model.predict(states)
        truth = true_states if true_states is not None else s
        ll = float(model.log_likelihood(states))
        report = MarkovEvaluator().evaluate(
            true_states=truth,
            predicted_states=pred,
            probabilities=proba,
            transition=model.transition_matrix(),
            log_likelihood=ll,
            n_params=model.n_params,
        )
        diag = MarkovDiagnostics().generate(
            states=s,
            transition=model.transition_matrix(),
            counts=model.estimator.matrix.count_matrix(),
            min_count_warning=self.settings.estimation.min_count_warning,
            state_names=model.state_names,
        )
        return {
            "mode": "fit",
            "n_samples": int(s.size),
            "n_transitions": int(model.estimator.matrix.n_transitions),
            "evaluation": report,
            "diagnostics": diag,
            "version": model.meta.version if hasattr(model, "meta") else "1.0.0",
        }

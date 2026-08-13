"""Transition probability estimators (MLE, Bayesian, frequency, weighted, online)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from iqrp.app.math.matrices.matrix import normalize_rows
from iqrp.app.regimes.markov.transition import TransitionMatrix

EstimationMethod = Literal["mle", "bayesian", "frequency", "weighted"]


class TransitionEstimator:
    """Estimate row-stochastic transition matrices from discrete state sequences."""

    def __init__(
        self,
        n_states: int,
        *,
        method: EstimationMethod = "bayesian",
        laplace_alpha: float = 1.0,
        dirichlet_alpha: float = 1.0,
        forgetting_factor: float = 1.0,
    ) -> None:
        self.n_states = int(n_states)
        self.method = method
        self.laplace_alpha = float(laplace_alpha)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.forgetting_factor = float(forgetting_factor)
        self.matrix = TransitionMatrix(self.n_states, laplace_alpha=self.laplace_alpha)

    def fit(
        self,
        states: Any,
        *,
        weights: Any | None = None,
    ) -> np.ndarray:
        self.matrix.reset()
        self.matrix.update_sequence(
            states,
            weights=weights if self.method == "weighted" else None,
            forgetting_factor=1.0,
        )
        return self.probability_matrix()

    def partial_fit(
        self,
        states: Any,
        *,
        weights: Any | None = None,
    ) -> np.ndarray:
        self.matrix.update_sequence(
            states,
            weights=weights if self.method == "weighted" else None,
            forgetting_factor=self.forgetting_factor,
        )
        return self.probability_matrix()

    def probability_matrix(self) -> np.ndarray:
        if self.method in ("mle", "frequency"):
            return self.matrix.probability_matrix(alpha=0.0)
        if self.method == "bayesian":
            return self._bayesian_posterior_mean()
        # weighted uses stored (possibly weighted) counts + Laplace
        return self.matrix.probability_matrix(alpha=self.laplace_alpha)

    def _bayesian_posterior_mean(self) -> np.ndarray:
        """Dirichlet(alpha) prior -> posterior mean (n_ij + alpha) / (n_i. + K*alpha)."""
        alpha = self.dirichlet_alpha
        counts = self.matrix.counts + alpha
        return normalize_rows(counts)

    def log_likelihood(self, states: Any) -> float:
        s = np.asarray(states, dtype=np.int64).reshape(-1)
        p = self.probability_matrix()
        ll = 0.0
        for t in range(s.size - 1):
            i, j = int(s[t]), int(s[t + 1])
            if 0 <= i < self.n_states and 0 <= j < self.n_states:
                ll += float(np.log(max(p[i, j], 1e-300)))
        return ll

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_states": self.n_states,
            "method": self.method,
            "laplace_alpha": self.laplace_alpha,
            "dirichlet_alpha": self.dirichlet_alpha,
            "forgetting_factor": self.forgetting_factor,
            "matrix": self.matrix.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransitionEstimator:
        obj = cls(
            int(data["n_states"]),
            method=data.get("method", "bayesian"),
            laplace_alpha=float(data.get("laplace_alpha", 1.0)),
            dirichlet_alpha=float(data.get("dirichlet_alpha", 1.0)),
            forgetting_factor=float(data.get("forgetting_factor", 1.0)),
        )
        obj.matrix = TransitionMatrix.from_dict(data["matrix"])
        return obj

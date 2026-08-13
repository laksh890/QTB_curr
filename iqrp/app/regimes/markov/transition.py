"""Transition count and probability matrices for discrete Markov chains."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse  # type: ignore[import-untyped]

from iqrp.app.math.matrices.matrix import normalize_rows
from iqrp.app.math.stochastic.markov_utils import is_stochastic


class TransitionMatrix:
    """Mutable transition count / probability container with online updates."""

    def __init__(
        self,
        n_states: int,
        *,
        laplace_alpha: float = 0.0,
        sparse_threshold: float = 0.0,
    ) -> None:
        if n_states < 1:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError("n_states must be >= 1", code="MARKOV_N_STATES")
        self.n_states = int(n_states)
        self.laplace_alpha = float(laplace_alpha)
        self.sparse_threshold = float(sparse_threshold)
        self.counts = np.zeros((self.n_states, self.n_states), dtype=np.float64)
        self._n_transitions = 0

    @property
    def n_transitions(self) -> int:
        return int(self._n_transitions)

    def reset(self) -> None:
        self.counts[:] = 0.0
        self._n_transitions = 0

    def update_pair(self, from_state: int, to_state: int, *, weight: float = 1.0) -> None:
        i, j = int(from_state), int(to_state)
        if 0 <= i < self.n_states and 0 <= j < self.n_states:
            self.counts[i, j] += float(weight)
            self._n_transitions += 1

    def update_sequence(
        self,
        states: Any,
        *,
        weights: Any | None = None,
        forgetting_factor: float = 1.0,
    ) -> None:
        s = np.asarray(states, dtype=np.int64).reshape(-1)
        if s.size < 2:
            return
        if forgetting_factor < 1.0:
            self.counts *= float(forgetting_factor)
        if weights is None:
            _accumulate_counts(self.counts, s)
            self._n_transitions += int(s.size - 1)
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.size == s.size:
                w = w[1:]
            for t in range(s.size - 1):
                i, j = int(s[t]), int(s[t + 1])
                if 0 <= i < self.n_states and 0 <= j < self.n_states:
                    wt = float(w[t]) if t < w.size else 1.0
                    self.counts[i, j] += wt
                    self._n_transitions += 1

    def count_matrix(self) -> np.ndarray:
        return self.counts.copy()

    def probability_matrix(self, *, alpha: float | None = None) -> np.ndarray:
        a = self.laplace_alpha if alpha is None else float(alpha)
        smoothed = self.counts + a
        return normalize_rows(smoothed)

    def sparse_probability_matrix(self, *, alpha: float | None = None) -> sparse.csr_matrix:
        p = self.probability_matrix(alpha=alpha)
        if self.sparse_threshold > 0:
            p = np.where(p >= self.sparse_threshold, p, 0.0)
            p = normalize_rows(p)
        return sparse.csr_matrix(p)

    def validate(self, *, tol: float = 1e-8) -> bool:
        return is_stochastic(self.probability_matrix(), tol=tol)

    def apply_window(
        self, states: Any, window_size: int, *, alpha: float | None = None
    ) -> np.ndarray:
        """Re-estimate counts from the trailing window of ``states``."""
        s = np.asarray(states, dtype=np.int64).reshape(-1)
        if window_size > 0 and s.size > window_size:
            s = s[-window_size:]
        tmp = TransitionMatrix(self.n_states, laplace_alpha=self.laplace_alpha)
        tmp.update_sequence(s)
        self.counts = tmp.counts
        self._n_transitions = tmp._n_transitions
        return self.probability_matrix(alpha=alpha)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_states": self.n_states,
            "laplace_alpha": self.laplace_alpha,
            "sparse_threshold": self.sparse_threshold,
            "counts": self.counts.tolist(),
            "n_transitions": self._n_transitions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransitionMatrix:
        obj = cls(
            int(data["n_states"]),
            laplace_alpha=float(data.get("laplace_alpha", 0.0)),
            sparse_threshold=float(data.get("sparse_threshold", 0.0)),
        )
        obj.counts = np.asarray(data["counts"], dtype=np.float64)
        obj._n_transitions = int(data.get("n_transitions", 0))
        return obj


def _accumulate_counts(counts: np.ndarray, states: np.ndarray) -> None:
    k = counts.shape[0]
    for t in range(states.size - 1):
        i, j = int(states[t]), int(states[t + 1])
        if 0 <= i < k and 0 <= j < k:
            counts[i, j] += 1.0

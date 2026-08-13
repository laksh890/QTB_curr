"""Abstract transition model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from iqrp.app.math.matrices.matrix import normalize_rows
from iqrp.app.math.stochastic.markov_utils import is_stochastic, n_step_transition


class TransitionModel(ABC):
    """Maps latent state at ``t`` to distribution over states at ``t+1``."""

    @abstractmethod
    def transition_matrix(self) -> np.ndarray:
        """Row-stochastic ``K x K`` transition matrix ``P(i -> j)``."""

    @abstractmethod
    def sample_next_state(
        self,
        state: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Draw ``s_{t+1} | s_t = state``."""

    @abstractmethod
    def transition_probability(self, from_state: int, to_state: int) -> float:
        """``P(s_{t+1}=to | s_t=from)``."""

    def n_step_matrix(self, n: int) -> np.ndarray:
        """``P^n`` via the math-engine matrix power utility."""
        return n_step_transition(self.transition_matrix(), n)

    def validate(self, *, tol: float = 1e-8) -> bool:
        return is_stochastic(self.transition_matrix(), tol=tol)

    def expected_durations(self) -> dict[int, float]:
        """Geometric expected sojourn times ``1 / (1 - P_ii)``."""
        p = self.transition_matrix()
        out: dict[int, float] = {}
        for i in range(p.shape[0]):
            stay = float(p[i, i])
            out[i] = float(1.0 / max(1.0 - stay, 1e-12))
        return out


class MatrixTransitionModel(TransitionModel):
    """Concrete transition model parameterized by a row-stochastic matrix."""

    def __init__(self, matrix: Any) -> None:
        tm = normalize_rows(np.asarray(matrix, dtype=np.float64))
        if tm.ndim != 2 or tm.shape[0] != tm.shape[1]:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(
                "Transition matrix must be square",
                code="SS_TRANSITION_SHAPE",
            )
        self._matrix = tm

    def transition_matrix(self) -> np.ndarray:
        return self._matrix.copy()

    def sample_next_state(
        self,
        state: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> int:
        rng = rng or np.random.default_rng()
        row = self._matrix[int(state)]
        return int(rng.choice(len(row), p=row))

    def transition_probability(self, from_state: int, to_state: int) -> float:
        return float(self._matrix[int(from_state), int(to_state)])

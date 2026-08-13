"""HMM transition and initial-state parameters."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math.matrices.matrix import normalize_rows
from iqrp.app.math.stochastic.markov_utils import is_stochastic


class HMMTransitions:
    """Row-stochastic transition matrix ``A`` and initial distribution ``pi``."""

    def __init__(
        self,
        n_states: int,
        *,
        transition: np.ndarray | None = None,
        initial: np.ndarray | None = None,
        dirichlet_alpha: float = 1.0,
    ) -> None:
        self.n_states = int(n_states)
        self.dirichlet_alpha = float(dirichlet_alpha)
        if transition is None:
            self.transition = np.full((self.n_states, self.n_states), 1.0 / self.n_states)
        else:
            self.transition = normalize_rows(np.asarray(transition, dtype=np.float64))
        if initial is None:
            self.initial = np.full(self.n_states, 1.0 / self.n_states)
        else:
            pi = np.asarray(initial, dtype=np.float64).reshape(-1)
            self.initial = pi / max(float(pi.sum()), 1e-300)

    def validate(self, *, tol: float = 1e-8) -> bool:
        pi_ok = bool(np.allclose(self.initial.sum(), 1.0, atol=tol))
        return pi_ok and is_stochastic(self.transition, tol=tol)

    def m_step(
        self,
        xi: np.ndarray,
        gamma: np.ndarray,
    ) -> None:
        """Update from expected transitions ``xi (T-1,K,K)`` and occupancy ``gamma``."""
        alpha = self.dirichlet_alpha
        counts = xi.sum(axis=0) + alpha
        self.transition = normalize_rows(counts)
        self.initial = gamma[0] / max(float(gamma[0].sum()), 1e-300)

    def expected_durations(self) -> dict[int, float]:
        out: dict[int, float] = {}
        for i in range(self.n_states):
            out[i] = float(1.0 / max(1.0 - float(self.transition[i, i]), 1e-12))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_states": self.n_states,
            "dirichlet_alpha": self.dirichlet_alpha,
            "transition": self.transition.tolist(),
            "initial": self.initial.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HMMTransitions:
        return cls(
            int(data["n_states"]),
            transition=np.asarray(data["transition"], dtype=np.float64),
            initial=np.asarray(data["initial"], dtype=np.float64),
            dirichlet_alpha=float(data.get("dirichlet_alpha", 1.0)),
        )

"""Bayesian transition and initial-state parameterizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from iqrp.app.math.matrices.matrix import normalize_rows
from iqrp.app.regimes.bayesian.priors import ModelPriors, sample_dirichlet_rows


@dataclass
class BayesianTransitions:
    """Transition matrix and initial distribution with Dirichlet priors."""

    n_states: int
    transition: np.ndarray
    initial: np.ndarray
    prior_alpha: np.ndarray
    prior_initial: np.ndarray

    @classmethod
    def from_priors(cls, priors: ModelPriors, *, rng: np.random.Generator) -> BayesianTransitions:
        k = int(priors.transition_alpha.shape[0])
        tm = sample_dirichlet_rows(priors.transition_alpha, rng)
        pi = rng.dirichlet(np.clip(priors.initial_alpha, 1e-6, None))
        return cls(
            n_states=k,
            transition=tm,
            initial=pi,
            prior_alpha=priors.transition_alpha.copy(),
            prior_initial=priors.initial_alpha.copy(),
        )

    def sample_posterior(
        self,
        states: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> BayesianTransitions:
        """Dirichlet-Multinomial conjugate update given a latent path."""
        s = np.asarray(states, dtype=np.int64).reshape(-1)
        k = self.n_states
        counts = self.prior_alpha.copy()
        for t in range(len(s) - 1):
            a, b = int(s[t]), int(s[t + 1])
            if 0 <= a < k and 0 <= b < k:
                counts[a, b] += 1.0
        tm = sample_dirichlet_rows(counts, rng)
        init_counts = self.prior_initial.copy()
        if s.size:
            init_counts[int(np.clip(s[0], 0, k - 1))] += 1.0
        pi = rng.dirichlet(np.clip(init_counts, 1e-6, None))
        return BayesianTransitions(
            n_states=k,
            transition=tm,
            initial=pi,
            prior_alpha=self.prior_alpha,
            prior_initial=self.prior_initial,
        )

    def expected_durations(self) -> dict[int, float]:
        p = np.clip(np.diag(self.transition), 1e-12, 1 - 1e-12)
        return {i: float(1.0 / (1.0 - p[i])) for i in range(self.n_states)}

    def persistence(self) -> np.ndarray:
        return np.diag(self.transition).copy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_states": self.n_states,
            "transition": self.transition.tolist(),
            "initial": self.initial.tolist(),
            "prior_alpha": self.prior_alpha.tolist(),
            "prior_initial": self.prior_initial.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BayesianTransitions:
        return cls(
            n_states=int(data["n_states"]),
            transition=normalize_rows(np.asarray(data["transition"], dtype=np.float64)),
            initial=np.asarray(data["initial"], dtype=np.float64),
            prior_alpha=np.asarray(data["prior_alpha"], dtype=np.float64),
            prior_initial=np.asarray(data["prior_initial"], dtype=np.float64),
        )

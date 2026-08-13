"""Backward (beta) filter / message pass."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.state_space.base.filter_result import FilterResult
from iqrp.app.state_space.base.probabilities import backward_probabilities, forward_probabilities
from iqrp.app.state_space.filtering.base_filter import BaseFilter


class BackwardFilter(BaseFilter):
    """Backward recursion; also exposes a combined filter result with β messages."""

    def run(
        self,
        log_emissions: Any,
        transition: Any,
        *,
        initial: Any | None = None,
    ) -> FilterResult:
        eps = float(self.settings.filtering.numerical_eps)
        log_b = np.asarray(log_emissions, dtype=np.float64)
        alpha, scales, ll = forward_probabilities(log_b, transition, initial=initial, eps=eps)
        beta = backward_probabilities(log_b, transition, scales=scales, eps=eps)
        states = self.hard_states(alpha)
        return FilterResult(
            filtered_states=states,
            filtered_probabilities=alpha,
            log_likelihood=ll,
            normalization_constants=scales,
            log_messages=np.log(np.clip(beta, eps, None)),
            metadata={"algorithm": "backward", "beta_shape": list(beta.shape)},
        )

    def backward_messages(
        self,
        log_emissions: Any,
        transition: Any,
        *,
        scales: Any | None = None,
    ) -> np.ndarray:
        eps = float(self.settings.filtering.numerical_eps)
        return backward_probabilities(log_emissions, transition, scales=scales, eps=eps)

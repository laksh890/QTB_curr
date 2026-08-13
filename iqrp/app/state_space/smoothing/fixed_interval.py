"""Fixed-interval (forward-backward) smoother."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.state_space.base.filter_result import FilterResult
from iqrp.app.state_space.base.probabilities import (
    backward_probabilities,
    forward_probabilities,
    state_occupancy_probabilities,
)
from iqrp.app.state_space.base.smoother_result import SmootherResult
from iqrp.app.state_space.smoothing.base_smoother import BaseSmoother


class FixedIntervalSmoother(BaseSmoother):
    """Classic fixed-interval smoother ``gamma_t proportional to alpha_t * beta_t``."""

    def run(
        self,
        log_emissions: Any,
        transition: Any,
        *,
        initial: Any | None = None,
        filter_result: FilterResult | None = None,
        lag: int | None = None,
    ) -> SmootherResult:
        del lag  # unused for fixed-interval
        eps = float(self.settings.filtering.numerical_eps)
        log_b = np.asarray(log_emissions, dtype=np.float64)
        if filter_result is None:
            alpha, scales, ll = forward_probabilities(log_b, transition, initial=initial, eps=eps)
        else:
            alpha = filter_result.filtered_probabilities
            scales = filter_result.normalization_constants
            ll = filter_result.log_likelihood
        beta = backward_probabilities(log_b, transition, scales=scales, eps=eps)
        gamma = state_occupancy_probabilities(alpha, beta)
        states = self.hard_states(gamma)
        return SmootherResult(
            smoothed_states=states,
            smoothed_probabilities=gamma,
            backward_messages=beta,
            log_likelihood=ll,
            metadata={"algorithm": "fixed_interval"},
        )

"""Fixed-lag smoother for near-online inference."""

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


class FixedLagSmoother(BaseSmoother):
    """Approximate fixed-lag smoother over sliding windows of length ``lag``."""

    def run(
        self,
        log_emissions: Any,
        transition: Any,
        *,
        initial: Any | None = None,
        filter_result: FilterResult | None = None,
        lag: int | None = None,
    ) -> SmootherResult:
        eps = float(self.settings.filtering.numerical_eps)
        log_b = np.asarray(log_emissions, dtype=np.float64)
        t_steps, k = log_b.shape
        lag_n = int(lag if lag is not None else self.settings.smoothing.fixed_lag)
        lag_n = max(1, min(lag_n, t_steps))

        if filter_result is None:
            alpha, _scales, ll = forward_probabilities(log_b, transition, initial=initial, eps=eps)
        else:
            alpha = filter_result.filtered_probabilities
            ll = filter_result.log_likelihood

        gamma = np.zeros((t_steps, k), dtype=np.float64)
        beta_store = np.zeros((t_steps, k), dtype=np.float64)

        for t in range(t_steps):
            end = min(t + lag_n, t_steps)
            start = max(0, end - lag_n)
            # Local fixed-interval smooth on [start, end)
            local_log_b = log_b[start:end]
            local_init = alpha[start] if start > 0 else initial
            local_alpha, local_scales, _ = forward_probabilities(
                local_log_b, transition, initial=local_init, eps=eps
            )
            local_beta = backward_probabilities(
                local_log_b, transition, scales=local_scales, eps=eps
            )
            local_gamma = state_occupancy_probabilities(local_alpha, local_beta)
            idx = t - start
            gamma[t] = local_gamma[idx]
            beta_store[t] = local_beta[idx]

        # Fallback: where numerical issues leave zeros, use filtered alpha
        row_sums = gamma.sum(axis=1, keepdims=True)
        bad = row_sums.ravel() <= 0
        if np.any(bad):
            gamma[bad] = alpha[bad]
        states = self.hard_states(gamma)
        return SmootherResult(
            smoothed_states=states,
            smoothed_probabilities=gamma,
            backward_messages=beta_store,
            log_likelihood=ll,
            metadata={"algorithm": "fixed_lag", "lag": lag_n},
        )

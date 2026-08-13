"""Forward (alpha) filter using math-engine stable recursions."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.state_space.base.filter_result import FilterResult
from iqrp.app.state_space.base.probabilities import forward_probabilities
from iqrp.app.state_space.filtering.base_filter import BaseFilter


class ForwardFilter(BaseFilter):
    """Memory-efficient forward filter with optional chunked processing."""

    def run(
        self,
        log_emissions: Any,
        transition: Any,
        *,
        initial: Any | None = None,
    ) -> FilterResult:
        log_b = np.asarray(log_emissions, dtype=np.float64)
        chunk = max(1, int(self.settings.filtering.chunk_size))
        eps = float(self.settings.filtering.numerical_eps)

        if log_b.shape[0] <= chunk:
            alpha, scales, ll = forward_probabilities(log_b, transition, initial=initial, eps=eps)
            states = self.hard_states(alpha)
            return FilterResult(
                filtered_states=states,
                filtered_probabilities=alpha,
                log_likelihood=ll,
                normalization_constants=scales,
                log_messages=np.log(np.clip(alpha, eps, None)),
                metadata={"algorithm": "forward", "chunked": False},
            )

        # Chunked pass for long series: stitch scaled forward messages
        pieces_alpha: list[np.ndarray] = []
        pieces_scale: list[np.ndarray] = []
        log_lik = 0.0
        cur_initial = initial
        for start in range(0, log_b.shape[0], chunk):
            end = min(start + chunk, log_b.shape[0])
            alpha, scales, ll = forward_probabilities(
                log_b[start:end],
                transition,
                initial=cur_initial,
                eps=eps,
            )
            pieces_alpha.append(alpha)
            pieces_scale.append(scales)
            log_lik += ll
            cur_initial = alpha[-1]
        alpha_all = np.vstack(pieces_alpha)
        scales_all = np.concatenate(pieces_scale)
        states = self.hard_states(alpha_all)
        return FilterResult(
            filtered_states=states,
            filtered_probabilities=alpha_all,
            log_likelihood=float(log_lik),
            normalization_constants=scales_all,
            log_messages=np.log(np.clip(alpha_all, eps, None)),
            metadata={"algorithm": "forward", "chunked": True, "chunk_size": chunk},
        )

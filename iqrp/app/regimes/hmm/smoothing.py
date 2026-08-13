"""HMM smoothing wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.regimes.hmm.forward_backward import ForwardBackwardResult, forward_backward
from iqrp.app.state_space.base.smoother_result import SmootherResult


def smooth(
    log_emissions: Any,
    transition: Any,
    *,
    initial: Any | None = None,
) -> tuple[SmootherResult, ForwardBackwardResult]:
    fb = forward_backward(log_emissions, transition, initial=initial)
    states = np.argmax(fb.gamma, axis=1).astype(np.int64)
    result = SmootherResult(
        smoothed_states=states,
        smoothed_probabilities=fb.gamma,
        backward_messages=fb.beta,
        log_likelihood=fb.log_likelihood,
        metadata={"algorithm": "forward_backward"},
    )
    return result, fb

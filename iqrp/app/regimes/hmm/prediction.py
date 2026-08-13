"""HMM prediction and multi-step forecasting."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math.stochastic.markov_utils import n_step_transition
from iqrp.app.regimes.hmm.transitions import HMMTransitions
from iqrp.app.state_space.base.forecast_result import ForecastResult
from iqrp.app.state_space.forecasting.uncertainty import forecast_uncertainty


def current_state_distribution(gamma: np.ndarray) -> np.ndarray:
    """Posterior over states at the last time index."""
    g = np.asarray(gamma, dtype=np.float64)
    if g.ndim == 1:
        return np.asarray(g / max(float(g.sum()), 1e-300), dtype=np.float64)
    row = g[-1]
    return np.asarray(row / max(float(row.sum()), 1e-300), dtype=np.float64)


def forecast_states(
    current: Any,
    transitions: HMMTransitions,
    *,
    horizon: int = 1,
    state_names: tuple[str, ...] = (),
    confidence_level: float = 0.95,
) -> ForecastResult:
    h = max(1, int(horizon))
    pi = np.asarray(current, dtype=np.float64).reshape(-1)
    pi = pi / max(float(pi.sum()), 1e-300)
    p = transitions.transition
    steps = np.empty((h, len(pi)), dtype=np.float64)
    for step in range(1, h + 1):
        steps[step - 1] = pi @ n_step_transition(p, step)
    result = ForecastResult.from_probabilities(
        steps[-1],
        horizon=h,
        expected_duration=transitions.expected_durations(),
        step_distributions=steps,
        state_names=state_names,
        confidence_level=confidence_level,
    )
    unc = forecast_uncertainty(steps, confidence_level=confidence_level)
    return ForecastResult(
        horizon=result.horizon,
        expected_state=result.expected_state,
        probability_distribution=result.probability_distribution,
        confidence_interval=result.confidence_interval,
        expected_duration=result.expected_duration,
        step_distributions=result.step_distributions,
        state_names=result.state_names,
        metadata={**result.metadata, "uncertainty": unc},
    )

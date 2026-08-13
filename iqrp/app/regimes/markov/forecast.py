"""Multi-step Markov forecasting via matrix exponentiation."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math.stochastic.markov_utils import n_step_transition
from iqrp.app.regimes.markov.persistence import expected_duration
from iqrp.app.state_space.base.forecast_result import ForecastResult
from iqrp.app.state_space.forecasting.uncertainty import forecast_uncertainty


class MarkovForecaster:
    """1-step and N-step forecasts using math-engine ``P^h``."""

    def forecast(
        self,
        current: Any,
        transition: Any,
        *,
        horizon: int = 1,
        state_names: tuple[str, ...] = (),
        confidence_level: float = 0.95,
    ) -> ForecastResult:
        h = max(1, int(horizon))
        pi = _as_distribution(current)
        p = np.asarray(transition, dtype=np.float64)
        steps = np.empty((h, len(pi)), dtype=np.float64)
        for step in range(1, h + 1):
            steps[step - 1] = pi @ n_step_transition(p, step)
        final = steps[-1]
        result = ForecastResult.from_probabilities(
            final,
            horizon=h,
            expected_duration=expected_duration(p),
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
            metadata={
                **result.metadata,
                "uncertainty": unc,
                "one_step": steps[0].tolist(),
                "most_likely_future_state": int(result.expected_state),
            },
        )

    def one_step(self, current: Any, transition: Any) -> np.ndarray:
        pi = _as_distribution(current)
        p = np.asarray(transition, dtype=np.float64)
        return np.asarray(pi @ p, dtype=np.float64)

    def n_step(self, current: Any, transition: Any, n: int) -> np.ndarray:
        pi = _as_distribution(current)
        return np.asarray(pi @ n_step_transition(transition, n), dtype=np.float64)


def _as_distribution(current: Any, *, n_states: int | None = None) -> np.ndarray:
    x = np.asarray(current, dtype=np.float64).reshape(-1)
    if x.size == 1 and n_states is not None:
        out = np.zeros(int(n_states), dtype=np.float64)
        idx = int(x[0])
        if 0 <= idx < n_states:
            out[idx] = 1.0
        else:
            out[:] = 1.0 / n_states
        return out
    s = float(x.sum())
    return x / s if s > 0 else x

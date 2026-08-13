"""Multi-step latent-state forecasting via matrix powers."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math.stochastic.markov_utils import n_step_transition
from iqrp.app.state_space.base.forecast_result import ForecastResult
from iqrp.app.state_space.base.probabilities import forecast_distribution
from iqrp.app.state_space.base.transition_model import TransitionModel
from iqrp.app.state_space.config import StateSpaceSettings
from iqrp.app.state_space.forecasting.uncertainty import forecast_uncertainty


class MultiStepForecaster:
    """Arbitrary-horizon forecasts using math-engine ``P^h``."""

    def __init__(self, settings: StateSpaceSettings | None = None) -> None:
        self.settings = settings or StateSpaceSettings.default()

    def forecast(
        self,
        current_distribution: Any,
        transition: Any | TransitionModel,
        *,
        horizon: int | None = None,
        state_names: tuple[str, ...] = (),
    ) -> ForecastResult:
        h = int(horizon if horizon is not None else self.settings.forecasting.default_horizon)
        h = max(1, h)
        if isinstance(transition, TransitionModel):
            tm = transition.transition_matrix()
            durations = transition.expected_durations()
        else:
            tm = np.asarray(transition, dtype=np.float64)
            durations = {
                i: float(1.0 / max(1.0 - float(tm[i, i]), 1e-12)) for i in range(tm.shape[0])
            }

        pi = np.asarray(current_distribution, dtype=np.float64).reshape(-1)
        pi = pi / max(float(pi.sum()), 1e-300)

        steps = np.empty((h, len(pi)), dtype=np.float64)
        for step in range(1, h + 1):
            steps[step - 1] = forecast_distribution(pi, tm, step)

        final = steps[-1]
        level = float(self.settings.forecasting.confidence_level)
        result = ForecastResult.from_probabilities(
            final,
            horizon=h,
            expected_duration=durations,
            step_distributions=steps,
            state_names=state_names,
            confidence_level=level,
        )
        unc = forecast_uncertainty(steps, confidence_level=level)
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
                "p_h": n_step_transition(tm, h).tolist(),
            },
        )

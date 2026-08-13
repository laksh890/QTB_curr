"""Posterior predictive forecasting for Bayesian regime models."""

from __future__ import annotations

import numpy as np

from iqrp.app.math.stochastic.markov_utils import n_step_transition
from iqrp.app.regimes.bayesian.posterior import Posterior
from iqrp.app.state_space.base.forecast_result import ForecastResult
from iqrp.app.state_space.forecasting.uncertainty import forecast_uncertainty


def current_state_distribution(proba: np.ndarray) -> np.ndarray:
    p = np.asarray(proba, dtype=np.float64)
    if p.ndim == 1:
        return np.asarray(p / max(float(p.sum()), 1e-300), dtype=np.float64)
    row = p[-1]
    return np.asarray(row / max(float(row.sum()), 1e-300), dtype=np.float64)


def expected_regime_duration(transition: np.ndarray) -> dict[int, float]:
    p = np.clip(np.diag(np.asarray(transition, dtype=np.float64)), 1e-12, 1 - 1e-12)
    return {i: float(1.0 / (1.0 - p[i])) for i in range(p.size)}


def forecast_from_posterior(
    posterior: Posterior,
    current: np.ndarray | None = None,
    *,
    horizon: int = 5,
    state_names: tuple[str, ...] = (),
    confidence_level: float = 0.95,
    n_draws: int | None = None,
) -> ForecastResult:
    """Mixture forecast over posterior transition draws."""
    h = max(1, int(horizon))
    draws = posterior.draws
    if n_draws is not None:
        draws = draws[: max(1, int(n_draws))]
    if not draws:
        k = 1
        steps = np.full((h, k), 1.0)
        return ForecastResult.from_probabilities(
            steps[-1], horizon=h, state_names=state_names, step_distributions=steps
        )

    k = draws[0].transition.shape[0]
    if current is None:
        # use mean occupancy of last latent state across draws
        last = []
        for d in draws:
            if d.states is not None and d.states.size:
                one = np.zeros(k)
                one[int(d.states[-1])] = 1.0
                last.append(one)
            else:
                last.append(d.initial)
        pi0 = np.mean(last, axis=0)
    else:
        pi0 = np.asarray(current, dtype=np.float64).reshape(-1)
    pi0 = pi0 / max(float(pi0.sum()), 1e-300)

    step_stack = []
    duration_stack = []
    for d in draws:
        steps = np.empty((h, k), dtype=np.float64)
        for step in range(1, h + 1):
            steps[step - 1] = pi0 @ n_step_transition(d.transition, step)
        step_stack.append(steps)
        duration_stack.append(expected_regime_duration(d.transition))

    mean_steps = np.mean(step_stack, axis=0)
    # credible band on final-step probabilities
    final = np.stack([s[-1] for s in step_stack], axis=0)
    alpha = 1.0 - float(confidence_level)
    low = np.percentile(final, 100 * alpha / 2, axis=0)
    high = np.percentile(final, 100 * (1 - alpha / 2), axis=0)
    unc = forecast_uncertainty(mean_steps[-1], confidence_level=confidence_level)
    dur_mean = {i: float(np.mean([d.get(i, 0.0) for d in duration_stack])) for i in range(k)}
    result = ForecastResult.from_probabilities(
        mean_steps[-1],
        horizon=h,
        state_names=state_names,
        expected_duration=dur_mean,
        step_distributions=mean_steps,
        confidence_level=confidence_level,
    )
    # attach posterior predictive uncertainty metadata
    meta = dict(result.metadata)
    meta["credible_low"] = low.tolist()
    meta["credible_high"] = high.tolist()
    meta["forecast_uncertainty"] = unc
    meta["n_posterior_draws"] = len(draws)
    return ForecastResult(
        horizon=result.horizon,
        expected_state=result.expected_state,
        probability_distribution=result.probability_distribution,
        confidence_interval=result.confidence_interval,
        expected_duration=result.expected_duration,
        step_distributions=result.step_distributions,
        state_names=result.state_names,
        metadata=meta,
    )


def posterior_predictive_state_proba(
    posterior: Posterior,
    *,
    n_steps: int | None = None,
) -> np.ndarray:
    return posterior.posterior_state_probabilities(n_steps)

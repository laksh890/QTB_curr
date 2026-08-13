"""Hidden-regime emission process for synthetic HMM-style datasets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from iqrp.app.simulation.regimes.regime_switching import RegimePath, RegimeSwitchingSimulator


@dataclass(frozen=True, slots=True)
class HiddenRegimeObservation:
    """Latent regimes plus noisy observations (returns)."""

    latent: RegimePath
    observations: np.ndarray
    emission_means: np.ndarray
    emission_stds: np.ndarray


class HiddenRegimeSimulator:
    """Generate latent Markov regimes with Gaussian emission returns."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()
        self._switcher = RegimeSwitchingSimulator(self.rng)

    def simulate(
        self,
        n_steps: int,
        *,
        transition_matrix: np.ndarray,
        state_names: tuple[str, ...] | list[str],
        emission_means: tuple[float, ...] | list[float] | np.ndarray,
        emission_stds: tuple[float, ...] | list[float] | np.ndarray,
        initial_state: int | None = None,
    ) -> HiddenRegimeObservation:
        means = np.asarray(emission_means, dtype=np.float64)
        stds = np.asarray(emission_stds, dtype=np.float64)
        latent = self._switcher.simulate(
            n_steps,
            transition_matrix=transition_matrix,
            state_names=state_names,
            drifts=means,
            volatilities=stds,
            initial_state=initial_state,
        )
        obs = self.rng.normal(means[latent.state_ids], stds[latent.state_ids])
        return HiddenRegimeObservation(
            latent=latent,
            observations=obs,
            emission_means=means,
            emission_stds=stds,
        )

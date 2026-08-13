"""Markov regime-switching path for drifts and volatilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REGIME_PRESETS: dict[str, dict[str, float]] = {
    "bull": {"drift": 0.12, "volatility": 0.18, "trend": 1.0},
    "bear": {"drift": -0.15, "volatility": 0.35, "trend": -1.0},
    "sideways": {"drift": 0.0, "volatility": 0.12, "trend": 0.0},
    "high_volatility": {"drift": 0.02, "volatility": 0.45, "trend": 0.0},
    "low_volatility": {"drift": 0.04, "volatility": 0.08, "trend": 0.2},
    "trending": {"drift": 0.18, "volatility": 0.2, "trend": 1.5},
    "mean_reverting": {"drift": 0.0, "volatility": 0.15, "trend": 0.0},
}


@dataclass(frozen=True, slots=True)
class RegimePath:
    state_ids: np.ndarray
    drifts: np.ndarray
    volatilities: np.ndarray
    trends: np.ndarray
    transition_matrix: np.ndarray
    state_names: tuple[str, ...]


class RegimeSwitchingSimulator:
    """Generate discrete-time Markov regime sequences with parameter overlays."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    def simulate(
        self,
        n_steps: int,
        *,
        transition_matrix: np.ndarray,
        state_names: tuple[str, ...] | list[str],
        drifts: tuple[float, ...] | list[float] | np.ndarray,
        volatilities: tuple[float, ...] | list[float] | np.ndarray,
        initial_state: int | None = None,
    ) -> RegimePath:
        tm = np.asarray(transition_matrix, dtype=np.float64)
        row_sums = tm.sum(axis=1, keepdims=True)
        tm = tm / np.where(row_sums <= 0, 1.0, row_sums)
        k = tm.shape[0]
        names = tuple(state_names)[:k]
        if len(names) < k:
            names = names + tuple(f"state_{i}" for i in range(len(names), k))
        drift_arr = np.asarray(drifts, dtype=np.float64)
        vol_arr = np.asarray(volatilities, dtype=np.float64)
        if drift_arr.size < k:
            drift_arr = np.pad(drift_arr, (0, k - drift_arr.size), constant_values=0.0)
        if vol_arr.size < k:
            vol_arr = np.pad(vol_arr, (0, k - vol_arr.size), constant_values=0.2)

        states = np.zeros(n_steps, dtype=np.int64)
        states[0] = int(initial_state if initial_state is not None else self.rng.integers(0, k))
        for t in range(1, n_steps):
            states[t] = int(self.rng.choice(k, p=tm[states[t - 1]]))

        trends = np.zeros(n_steps, dtype=np.float64)
        for i, name in enumerate(names):
            preset = REGIME_PRESETS.get(name, {})
            trends[states == i] = float(preset.get("trend", np.sign(drift_arr[i])))

        return RegimePath(
            state_ids=states,
            drifts=drift_arr[states],
            volatilities=vol_arr[states],
            trends=trends,
            transition_matrix=tm,
            state_names=names,
        )

    @staticmethod
    def mixed_transition(n_states: int = 3, persistence: float = 0.95) -> np.ndarray:
        p = float(np.clip(persistence, 0.0, 0.999))
        off = (1.0 - p) / max(n_states - 1, 1)
        tm = np.full((n_states, n_states), off, dtype=np.float64)
        np.fill_diagonal(tm, p)
        return tm

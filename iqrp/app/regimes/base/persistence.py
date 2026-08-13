"""Regime persistence / duration analytics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class PersistenceReport:
    state_durations: dict[int, list[int]]
    expected_duration: dict[int, float]
    persistence_score: dict[int, float]
    average_duration: dict[int, float]
    rolling_persistence: np.ndarray  # length T

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_durations": {str(k): v for k, v in self.state_durations.items()},
            "expected_duration": {str(k): v for k, v in self.expected_duration.items()},
            "persistence_score": {str(k): v for k, v in self.persistence_score.items()},
            "average_duration": {str(k): v for k, v in self.average_duration.items()},
            "rolling_persistence": np.asarray(self.rolling_persistence).tolist(),
        }


class PersistenceEngine:
    """Compute duration statistics from a hard state sequence."""

    @staticmethod
    def run_lengths(states: np.ndarray) -> dict[int, list[int]]:
        seq = np.asarray(states)
        durations: dict[int, list[int]] = {}
        if seq.size == 0:
            return durations
        current = int(seq[0])
        length = 1
        for s in seq[1:]:
            sid = int(s)
            if sid == current:
                length += 1
            else:
                durations.setdefault(current, []).append(length)
                current = sid
                length = 1
        durations.setdefault(current, []).append(length)
        return durations

    @staticmethod
    def expected_duration_from_transition(transition_matrix: np.ndarray) -> dict[int, float]:
        """E[duration | i] = 1 / (1 - P_ii) for Markov self-transition."""
        tm = np.asarray(transition_matrix, dtype=np.float64)
        out: dict[int, float] = {}
        for i in range(tm.shape[0]):
            p_stay = float(tm[i, i])
            out[i] = float(1.0 / max(1e-12, 1.0 - min(p_stay, 1.0 - 1e-12)))
        return out

    @staticmethod
    def persistence_score(transition_matrix: np.ndarray) -> dict[int, float]:
        """Self-transition probability as persistence score in [0, 1]."""
        tm = np.asarray(transition_matrix, dtype=np.float64)
        return {i: float(tm[i, i]) for i in range(tm.shape[0])}

    @staticmethod
    def average_duration(durations: dict[int, list[int]]) -> dict[int, float]:
        return {k: float(np.mean(v)) if v else float("nan") for k, v in durations.items()}

    @staticmethod
    def rolling_persistence(states: np.ndarray, window: int = 20) -> np.ndarray:
        """Fraction of bars in the window equal to the current state."""
        seq = np.asarray(states)
        n = len(seq)
        out = np.full(n, np.nan, dtype=np.float64)
        w = max(1, window)
        for i in range(n):
            start = max(0, i - w + 1)
            window_states = seq[start : i + 1]
            out[i] = float(np.mean(window_states == seq[i]))
        return out

    @classmethod
    def analyze(
        cls,
        states: np.ndarray,
        transition_matrix: np.ndarray,
        *,
        rolling_window: int = 20,
    ) -> PersistenceReport:
        durations = cls.run_lengths(states)
        return PersistenceReport(
            state_durations=durations,
            expected_duration=cls.expected_duration_from_transition(transition_matrix),
            persistence_score=cls.persistence_score(transition_matrix),
            average_duration=cls.average_duration(durations),
            rolling_persistence=cls.rolling_persistence(states, rolling_window),
        )

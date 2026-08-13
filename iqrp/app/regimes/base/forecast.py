"""Regime forecast objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class RegimeForecast:
    """Multi-horizon regime forecast distribution."""

    steps: int
    probabilities: np.ndarray  # shape (steps, n_states) or (n_states,) for 1-step
    state_names: tuple[str, ...] = ()
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    expected_duration: dict[int, float] = field(default_factory=dict)
    most_likely_path: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        probs = np.asarray(self.probabilities)
        return {
            "steps": self.steps,
            "probabilities": probs.tolist(),
            "state_names": list(self.state_names),
            "confidence_intervals": {k: list(v) for k, v in self.confidence_intervals.items()},
            "expected_duration": {str(k): v for k, v in self.expected_duration.items()},
            "most_likely_path": list(self.most_likely_path),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_probabilities(
        cls,
        probabilities: np.ndarray,
        *,
        state_names: tuple[str, ...] = (),
        expected_duration: dict[int, float] | None = None,
        alpha: float = 0.1,
    ) -> RegimeForecast:
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
            steps = 1
        else:
            steps = int(probs.shape[0])
        # Simple CI on max-state probability across horizons (Wilson-like clip)
        ci: dict[str, tuple[float, float]] = {}
        for h in range(steps):
            p = float(np.max(probs[h]))
            half = max(alpha * p, 0.01)
            ci[f"step_{h + 1}"] = (max(0.0, p - half), min(1.0, p + half))
        path = tuple(int(np.argmax(probs[h])) for h in range(steps))
        return cls(
            steps=steps,
            probabilities=probs,
            state_names=state_names,
            confidence_intervals=ci,
            expected_duration=expected_duration or {},
            most_likely_path=path,
        )

    def one_step(self) -> np.ndarray:
        probs = np.asarray(self.probabilities)
        return probs[0] if probs.ndim == 2 else probs

    def n_step(self, n: int) -> np.ndarray:
        probs = np.asarray(self.probabilities)
        if probs.ndim == 1:
            if n != 1:
                raise ValueError("Only 1-step available")
            return probs
        if n < 1 or n > probs.shape[0]:
            raise ValueError(f"n must be in 1..{probs.shape[0]}")
        return np.asarray(probs[n - 1], dtype=np.float64)

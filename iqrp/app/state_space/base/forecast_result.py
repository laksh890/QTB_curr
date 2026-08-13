"""Forecast result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ForecastResult:
    """Multi-step latent-state forecast."""

    horizon: int
    expected_state: int
    probability_distribution: np.ndarray
    confidence_interval: tuple[float, float]
    expected_duration: dict[int, float]
    step_distributions: np.ndarray | None = None
    state_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "probability_distribution",
            np.asarray(self.probability_distribution, dtype=np.float64).reshape(-1),
        )
        if self.step_distributions is not None:
            object.__setattr__(
                self,
                "step_distributions",
                np.asarray(self.step_distributions, dtype=np.float64),
            )

    @property
    def most_likely_path(self) -> list[int]:
        if self.step_distributions is None:
            return [int(self.expected_state)] * int(self.horizon)
        return [int(np.argmax(row)) for row in self.step_distributions]

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon": int(self.horizon),
            "expected_state": int(self.expected_state),
            "probability_distribution": self.probability_distribution.tolist(),
            "confidence_interval": [
                float(self.confidence_interval[0]),
                float(self.confidence_interval[1]),
            ],
            "expected_duration": {str(k): float(v) for k, v in self.expected_duration.items()},
            "step_distributions": (
                None if self.step_distributions is None else self.step_distributions.tolist()
            ),
            "state_names": list(self.state_names),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ForecastResult:
        ci = data["confidence_interval"]
        dur_raw = data.get("expected_duration") or {}
        expected_duration = {int(k): float(v) for k, v in dur_raw.items()}
        return cls(
            horizon=int(data["horizon"]),
            expected_state=int(data["expected_state"]),
            probability_distribution=np.asarray(data["probability_distribution"], dtype=np.float64),
            confidence_interval=(float(ci[0]), float(ci[1])),
            expected_duration=expected_duration,
            step_distributions=(
                None
                if data.get("step_distributions") is None
                else np.asarray(data["step_distributions"], dtype=np.float64)
            ),
            state_names=tuple(data.get("state_names") or ()),
            metadata=dict(data.get("metadata") or {}),
        )

    @classmethod
    def from_probabilities(
        cls,
        distribution: np.ndarray,
        *,
        horizon: int,
        expected_duration: dict[int, float] | None = None,
        step_distributions: np.ndarray | None = None,
        state_names: tuple[str, ...] = (),
        confidence_level: float = 0.95,
    ) -> ForecastResult:
        p = np.asarray(distribution, dtype=np.float64).reshape(-1)
        p = p / max(float(p.sum()), 1e-300)
        expected = int(np.argmax(p))
        # Highest-probability mass interval over sorted states for CI width on max mass
        sorted_idx = np.argsort(p)[::-1]
        cum = 0.0
        included: list[int] = []
        for i in sorted_idx:
            included.append(int(i))
            cum += float(p[i])
            if cum >= confidence_level:
                break
        lo = float(min(p[i] for i in included)) if included else 0.0
        hi = float(max(p[i] for i in included)) if included else 0.0
        return cls(
            horizon=int(horizon),
            expected_state=expected,
            probability_distribution=p,
            confidence_interval=(lo, hi),
            expected_duration=dict(expected_duration or {}),
            step_distributions=step_distributions,
            state_names=state_names,
        )

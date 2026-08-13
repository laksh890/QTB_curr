"""Smoother result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class SmootherResult:
    """Output of a fixed-interval or fixed-lag smoother."""

    smoothed_states: np.ndarray
    smoothed_probabilities: np.ndarray
    backward_messages: np.ndarray
    log_likelihood: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "smoothed_states", np.asarray(self.smoothed_states, dtype=np.int64).reshape(-1)
        )
        object.__setattr__(
            self,
            "smoothed_probabilities",
            np.asarray(self.smoothed_probabilities, dtype=np.float64),
        )
        object.__setattr__(
            self, "backward_messages", np.asarray(self.backward_messages, dtype=np.float64)
        )

    @property
    def n_steps(self) -> int:
        return int(self.smoothed_probabilities.shape[0])

    @property
    def n_states(self) -> int:
        return int(self.smoothed_probabilities.shape[1])

    def to_frame(
        self, *, timestamp_column: str = "open_time", timestamps: list[Any] | None = None
    ) -> pl.DataFrame:
        data: dict[str, Any] = {"state_id": self.smoothed_states.tolist()}
        if timestamps is not None:
            data[timestamp_column] = timestamps
        for j in range(self.n_states):
            data[f"proba_{j}"] = self.smoothed_probabilities[:, j].tolist()
        return pl.DataFrame(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "smoothed_states": self.smoothed_states.tolist(),
            "smoothed_probabilities": self.smoothed_probabilities.tolist(),
            "backward_messages": self.backward_messages.tolist(),
            "log_likelihood": (None if self.log_likelihood is None else float(self.log_likelihood)),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SmootherResult:
        return cls(
            smoothed_states=np.asarray(data["smoothed_states"], dtype=np.int64),
            smoothed_probabilities=np.asarray(data["smoothed_probabilities"], dtype=np.float64),
            backward_messages=np.asarray(data["backward_messages"], dtype=np.float64),
            log_likelihood=(
                None if data.get("log_likelihood") is None else float(data["log_likelihood"])
            ),
            metadata=dict(data.get("metadata") or {}),
        )

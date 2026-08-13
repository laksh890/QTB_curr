"""Filter result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Output of a forward (or backward) filter pass."""

    filtered_states: np.ndarray
    filtered_probabilities: np.ndarray
    log_likelihood: float
    normalization_constants: np.ndarray
    log_messages: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "filtered_states", np.asarray(self.filtered_states, dtype=np.int64).reshape(-1)
        )
        object.__setattr__(
            self,
            "filtered_probabilities",
            np.asarray(self.filtered_probabilities, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "normalization_constants",
            np.asarray(self.normalization_constants, dtype=np.float64).reshape(-1),
        )
        if self.log_messages is not None:
            object.__setattr__(
                self, "log_messages", np.asarray(self.log_messages, dtype=np.float64)
            )

    @property
    def n_steps(self) -> int:
        return int(self.filtered_probabilities.shape[0])

    @property
    def n_states(self) -> int:
        if self.filtered_probabilities.ndim < 2:
            return int(self.filtered_probabilities.size)
        return int(self.filtered_probabilities.shape[1])

    def to_frame(
        self, *, timestamp_column: str = "open_time", timestamps: list[Any] | None = None
    ) -> pl.DataFrame:
        data: dict[str, Any] = {"state_id": self.filtered_states.tolist()}
        if timestamps is not None:
            data[timestamp_column] = timestamps
        for j in range(self.n_states):
            data[f"proba_{j}"] = self.filtered_probabilities[:, j].tolist()
        data["scale"] = self.normalization_constants.tolist()
        return pl.DataFrame(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filtered_states": self.filtered_states.tolist(),
            "filtered_probabilities": self.filtered_probabilities.tolist(),
            "log_likelihood": float(self.log_likelihood),
            "normalization_constants": self.normalization_constants.tolist(),
            "log_messages": None if self.log_messages is None else self.log_messages.tolist(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FilterResult:
        return cls(
            filtered_states=np.asarray(data["filtered_states"], dtype=np.int64),
            filtered_probabilities=np.asarray(data["filtered_probabilities"], dtype=np.float64),
            log_likelihood=float(data["log_likelihood"]),
            normalization_constants=np.asarray(data["normalization_constants"], dtype=np.float64),
            log_messages=(
                None
                if data.get("log_messages") is None
                else np.asarray(data["log_messages"], dtype=np.float64)
            ),
            metadata=dict(data.get("metadata") or {}),
        )

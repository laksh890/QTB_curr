"""Training window bounds for walk-forward folds.

Look-ahead prevention
---------------------
``end`` is exclusive and must always be strictly less than the prediction /
test start (after purge). Training never includes future observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TrainingWindow:
    """Half-open index interval ``[start, end)`` used for model fitting."""

    start: int
    end: int  # exclusive

    def __post_init__(self) -> None:
        if int(self.start) < 0:
            raise ValueError(f"TrainingWindow.start must be >= 0, got {self.start}")
        if int(self.end) < int(self.start):
            raise ValueError(
                f"TrainingWindow.end ({self.end}) must be >= start ({self.start})"
            )

    @property
    def size(self) -> int:
        return int(self.end) - int(self.start)

    def indices(self) -> np.ndarray:
        """Return contiguous train indices."""
        return np.arange(int(self.start), int(self.end), dtype=int)

    def contains(self, idx: int) -> bool:
        return int(self.start) <= int(idx) < int(self.end)

    def with_bounds(self, start: int | None = None, end: int | None = None) -> TrainingWindow:
        return TrainingWindow(
            start=int(self.start if start is None else start),
            end=int(self.end if end is None else end),
        )

    def assert_before(self, prediction_time: int) -> None:
        """Raise if training end is not at/before ``prediction_time`` (exclusive end)."""
        # Half-open [start, end): end == prediction_time means last train idx is prediction_time-1.
        if int(self.end) > int(prediction_time):
            raise ValueError(
                f"NO FUTURE TRAINING: train end {self.end} must be <= "
                f"prediction timestamp {prediction_time}"
            )

    def __repr__(self) -> str:
        return f"TrainingWindow([{self.start}, {self.end}), size={self.size})"

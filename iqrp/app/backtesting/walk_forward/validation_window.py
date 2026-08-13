"""Validation (inner) window for nested walk-forward selection.

Used optionally between train and test for hyperparameter selection.
Always causal: validation sits after train and before test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ValidationWindow:
    """Half-open index interval ``[start, end)`` for inner validation."""

    start: int
    end: int  # exclusive

    def __post_init__(self) -> None:
        if int(self.start) < 0:
            raise ValueError(f"ValidationWindow.start must be >= 0, got {self.start}")
        if int(self.end) < int(self.start):
            raise ValueError(
                f"ValidationWindow.end ({self.end}) must be >= start ({self.start})"
            )

    @property
    def size(self) -> int:
        return int(self.end) - int(self.start)

    def indices(self) -> np.ndarray:
        return np.arange(int(self.start), int(self.end), dtype=int)

    def contains(self, idx: int) -> bool:
        return int(self.start) <= int(idx) < int(self.end)

    def assert_after_train(self, train_end: int) -> None:
        if int(self.start) < int(train_end):
            raise ValueError(
                f"Validation must start at/after train end ({train_end}), "
                f"got start={self.start}"
            )

    def assert_before_test(self, test_start: int) -> None:
        if int(self.end) > int(test_start):
            raise ValueError(
                f"Validation end ({self.end}) must be <= test start ({test_start})"
            )

    def __repr__(self) -> str:
        return f"ValidationWindow([{self.start}, {self.end}), size={self.size})"

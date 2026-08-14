"""Out-of-sample test window for walk-forward folds.

Predictions are evaluated only on this interval. Training must end strictly
before ``start`` (after any purge gap).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TestWindow:
    """Half-open index interval ``[start, end)`` for OOS evaluation."""

    start: int
    end: int  # exclusive

    def __post_init__(self) -> None:
        if int(self.start) < 0:
            raise ValueError(f"TestWindow.start must be >= 0, got {self.start}")
        if int(self.end) < int(self.start):
            raise ValueError(f"TestWindow.end ({self.end}) must be >= start ({self.start})")

    @property
    def size(self) -> int:
        return int(self.end) - int(self.start)

    @property
    def prediction_timestamp(self) -> int:
        """Earliest prediction time in this fold (inclusive index)."""
        return int(self.start)

    def indices(self) -> np.ndarray:
        return np.arange(int(self.start), int(self.end), dtype=int)

    def contains(self, idx: int) -> bool:
        return int(self.start) <= int(idx) < int(self.end)

    def embargo_zone(self, embargo: int) -> tuple[int, int]:
        """Return half-open ``[end, end+embargo)`` embargo interval."""
        e = max(int(embargo), 0)
        return int(self.end), int(self.end) + e

    def __repr__(self) -> str:
        return f"TestWindow([{self.start}, {self.end}), size={self.size})"

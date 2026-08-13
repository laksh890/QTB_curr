"""State label mapping utilities for discrete Markov observations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import polars as pl

StateMapper = Callable[[Any], int]


class LabelStateMapper:
    """Map arbitrary labels (str/int) to contiguous state ids ``0..K-1``."""

    def __init__(
        self,
        *,
        n_states: int | None = None,
        label_to_id: Mapping[Any, int] | None = None,
        custom_mapper: StateMapper | None = None,
    ) -> None:
        self._fixed = dict(label_to_id) if label_to_id else {}
        self._custom = custom_mapper
        self._learned: dict[Any, int] = {}
        self._n_states = n_states

    @property
    def n_states(self) -> int:
        if self._n_states is not None:
            return int(self._n_states)
        ids = list(self._fixed.values()) + list(self._learned.values())
        return (max(ids) + 1) if ids else 0

    def fit(self, labels: Any) -> LabelStateMapper:
        arr = _to_list(labels)
        if self._custom is not None:
            return self
        if self._fixed:
            return self
        unique: list[Any] = []
        for lab in arr:
            if lab not in unique:
                unique.append(lab)
        # Identity mapping when labels are already contiguous integer ids
        if unique and all(isinstance(u, (int, np.integer)) for u in unique):
            ints = sorted(int(u) for u in unique)
            if ints == list(range(len(ints))) or (
                self._n_states is not None and all(0 <= i < self._n_states for i in ints)
            ):
                self._learned = {i: i for i in ints}
                if self._n_states is None:
                    self._n_states = max(ints) + 1 if ints else 0
                return self
        self._learned = {lab: i for i, lab in enumerate(unique)}
        if self._n_states is None:
            self._n_states = len(unique)
        return self

    def transform(self, labels: Any) -> np.ndarray:
        arr = _to_list(labels)
        out = np.empty(len(arr), dtype=np.int64)
        for i, lab in enumerate(arr):
            out[i] = self.map_one(lab)
        return out

    def map_one(self, label: Any) -> int:
        if self._custom is not None:
            return int(self._custom(label))
        if label in self._fixed:
            return int(self._fixed[label])
        if label in self._learned:
            return int(self._learned[label])
        # numeric already
        if isinstance(label, (int, np.integer)):
            return int(label)
        try:
            return int(label)
        except (TypeError, ValueError) as exc:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(
                f"Unknown state label: {label!r}",
                code="MARKOV_UNKNOWN_LABEL",
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_states": self._n_states,
            "fixed": {str(k): int(v) for k, v in self._fixed.items()},
            "learned": {str(k): int(v) for k, v in self._learned.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabelStateMapper:
        obj = cls(n_states=data.get("n_states"), label_to_id=data.get("fixed") or {})
        obj._learned = {k: int(v) for k, v in (data.get("learned") or {}).items()}
        return obj


def _to_list(labels: Any) -> list[Any]:
    if isinstance(labels, pl.Series):
        return list(labels.to_list())
    if isinstance(labels, pl.DataFrame):
        return list(labels.to_series(0).to_list())
    return list(np.asarray(labels).reshape(-1).tolist())

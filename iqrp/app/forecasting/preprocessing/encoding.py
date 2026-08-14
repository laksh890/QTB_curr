"""Categorical encoding for forecasting features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl


@dataclass
class LabelEncoder:
    """Maps categorical string / int labels to dense integers."""

    classes_: list[Any] = field(default_factory=list)
    fitted: bool = False

    def fit(self, values: list[Any] | np.ndarray | pl.Series) -> LabelEncoder:
        if isinstance(values, pl.Series):
            vals = values.to_list()
        else:
            vals = list(np.asarray(values).reshape(-1))
        seen: list[Any] = []
        for v in vals:
            if v not in seen:
                seen.append(v)
        self.classes_ = seen
        self.fitted = True
        return self

    def transform(self, values: list[Any] | np.ndarray | pl.Series) -> np.ndarray:
        if isinstance(values, pl.Series):
            vals = values.to_list()
        else:
            vals = list(np.asarray(values).reshape(-1))
        index = {c: i for i, c in enumerate(self.classes_)}
        out = np.empty(len(vals), dtype=np.int64)
        unk = len(self.classes_)
        for i, v in enumerate(vals):
            out[i] = index.get(v, unk)
        return out

    def fit_transform(self, values: list[Any] | np.ndarray | pl.Series) -> np.ndarray:
        return self.fit(values).transform(values)

    def inverse_transform(self, codes: np.ndarray) -> list[Any]:
        out: list[Any] = []
        for c in np.asarray(codes).reshape(-1):
            ci = int(c)
            out.append(self.classes_[ci] if 0 <= ci < len(self.classes_) else None)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"classes_": list(self.classes_), "fitted": self.fitted}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabelEncoder:
        return cls(
            classes_=list(data.get("classes_") or []), fitted=bool(data.get("fitted", False))
        )


@dataclass
class OneHotEncoder:
    """Simple one-hot encoder for a single categorical column."""

    classes_: list[Any] = field(default_factory=list)
    fitted: bool = False

    def fit(self, values: list[Any] | np.ndarray | pl.Series) -> OneHotEncoder:
        enc = LabelEncoder().fit(values)
        self.classes_ = enc.classes_
        self.fitted = True
        return self

    def transform(self, values: list[Any] | np.ndarray | pl.Series) -> np.ndarray:
        codes = LabelEncoder(classes_=self.classes_, fitted=True).transform(values)
        k = max(len(self.classes_), 1)
        out = np.zeros((codes.size, k), dtype=np.float64)
        for i, c in enumerate(codes):
            if 0 <= int(c) < k:
                out[i, int(c)] = 1.0
        return out

    def fit_transform(self, values: list[Any] | np.ndarray | pl.Series) -> np.ndarray:
        return self.fit(values).transform(values)

    def to_dict(self) -> dict[str, Any]:
        return {"classes_": list(self.classes_), "fitted": self.fitted}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OneHotEncoder:
        return cls(
            classes_=list(data.get("classes_") or []), fitted=bool(data.get("fitted", False))
        )


def encode_frame_categoricals(
    frame: pl.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pl.DataFrame, dict[str, LabelEncoder]]:
    """Integer-encode non-numeric columns; return transformed frame + encoders."""
    cols = columns or [c for c in frame.columns if not frame[c].dtype.is_numeric()]
    encoders: dict[str, LabelEncoder] = {}
    out = frame
    for c in cols:
        if c not in out.columns:
            continue
        enc = LabelEncoder().fit(out[c])
        encoders[c] = enc
        codes = enc.transform(out[c])
        out = out.with_columns(pl.Series(name=c, values=codes))
    return out, encoders

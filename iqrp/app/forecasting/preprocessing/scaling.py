"""Feature scaling transforms for forecasting inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

ScalerKind = Literal["none", "standard", "minmax", "robust"]


@dataclass
class Scaler:
    kind: ScalerKind = "standard"
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    min_: np.ndarray | None = None
    max_: np.ndarray | None = None
    median_: np.ndarray | None = None
    iqr_: np.ndarray | None = None
    fitted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, x: np.ndarray) -> Scaler:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if self.kind == "none":
            self.fitted = True
            return self
        if self.kind == "standard":
            self.mean_ = np.mean(arr, axis=0)
            self.scale_ = np.std(arr, axis=0)
            self.scale_ = np.where(self.scale_ < 1e-12, 1.0, self.scale_)
        elif self.kind == "minmax":
            self.min_ = np.min(arr, axis=0)
            self.max_ = np.max(arr, axis=0)
            span = self.max_ - self.min_
            self.scale_ = np.where(span < 1e-12, 1.0, span)
        elif self.kind == "robust":
            self.median_ = np.median(arr, axis=0)
            q75 = np.percentile(arr, 75, axis=0)
            q25 = np.percentile(arr, 25, axis=0)
            self.iqr_ = q75 - q25
            self.iqr_ = np.where(self.iqr_ < 1e-12, 1.0, self.iqr_)
        self.fitted = True
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr.reshape(-1, 1)
        if not self.fitted or self.kind == "none":
            return arr.reshape(-1) if squeeze else arr
        if self.kind == "standard":
            out = (arr - self.mean_) / self.scale_
        elif self.kind == "minmax":
            out = (arr - self.min_) / self.scale_
        else:
            out = (arr - self.median_) / self.iqr_
        return out.reshape(-1) if squeeze else out

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        squeeze = arr.ndim == 1
        if squeeze:
            arr = arr.reshape(-1, 1)
        if not self.fitted or self.kind == "none":
            return arr.reshape(-1) if squeeze else arr
        if self.kind == "standard":
            out = arr * self.scale_ + self.mean_
        elif self.kind == "minmax":
            out = arr * self.scale_ + self.min_
        else:
            out = arr * self.iqr_ + self.median_
        return out.reshape(-1) if squeeze else out

    def to_dict(self) -> dict[str, Any]:
        def _ser(a: np.ndarray | None) -> list[float] | None:
            return None if a is None else np.asarray(a).tolist()

        return {
            "kind": self.kind,
            "mean_": _ser(self.mean_),
            "scale_": _ser(self.scale_),
            "min_": _ser(self.min_),
            "max_": _ser(self.max_),
            "median_": _ser(self.median_),
            "iqr_": _ser(self.iqr_),
            "fitted": self.fitted,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scaler:
        def _arr(key: str) -> np.ndarray | None:
            v = data.get(key)
            return None if v is None else np.asarray(v, dtype=np.float64)

        return cls(
            kind=data.get("kind", "standard"),  # type: ignore[arg-type]
            mean_=_arr("mean_"),
            scale_=_arr("scale_"),
            min_=_arr("min_"),
            max_=_arr("max_"),
            median_=_arr("median_"),
            iqr_=_arr("iqr_"),
            fitted=bool(data.get("fitted", False)),
            metadata=dict(data.get("metadata") or {}),
        )

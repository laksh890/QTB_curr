"""Shared result containers and temporal-safety metadata for time-series analytics.

Analytical discoveries are measurements and evidence — never trading signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class TemporalMode(str, Enum):
    """Declares whether an operation can leak future information."""

    POINT_IN_TIME = "point_in_time"
    ROLLING = "rolling"
    EXPANDING = "expanding"
    TRAINING_ONLY = "training_only"
    CAUSAL = "causal"
    FULL_SAMPLE = "full_sample"  # research-only; may use future info


@dataclass(slots=True)
class AnalysisResult:
    """Canonical structured output for every analytical method."""

    method: str
    value: Any
    timestamps: tuple[Any, ...] | None = None
    window: int | tuple[int, int] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    statistic: float | None = None
    pvalue: float | None = None
    critical_values: dict[str, float] | None = None
    confidence: float | None = None
    confidence_interval: tuple[float, float] | None = None
    null_hypothesis: str | None = None
    alternative_hypothesis: str | None = None
    temporal_mode: TemporalMode = TemporalMode.FULL_SAMPLE
    significant: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    data_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        val = self.value
        if isinstance(val, np.ndarray):
            val = val.tolist()
        return {
            "method": self.method,
            "value": val,
            "timestamps": list(self.timestamps) if self.timestamps is not None else None,
            "window": self.window if not isinstance(self.window, tuple) else list(self.window),
            "parameters": dict(self.parameters),
            "statistic": self.statistic,
            "pvalue": self.pvalue,
            "critical_values": dict(self.critical_values) if self.critical_values else None,
            "confidence": self.confidence,
            "confidence_interval": list(self.confidence_interval) if self.confidence_interval else None,
            "null_hypothesis": self.null_hypothesis,
            "alternative_hypothesis": self.alternative_hypothesis,
            "temporal_mode": self.temporal_mode.value,
            "significant": self.significant,
            "metadata": dict(self.metadata),
            "data_version": self.data_version,
        }


@dataclass(slots=True)
class DecompositionResult:
    method: str
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    observed: np.ndarray
    model: str = "additive"
    parameters: dict[str, Any] = field(default_factory=dict)
    temporal_mode: TemporalMode = TemporalMode.FULL_SAMPLE
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "trend": self.trend.tolist(),
            "seasonal": self.seasonal.tolist(),
            "residual": self.residual.tolist(),
            "observed": self.observed.tolist(),
            "model": self.model,
            "parameters": dict(self.parameters),
            "temporal_mode": self.temporal_mode.value,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ChangePointResult:
    method: str
    indices: list[int]
    scores: np.ndarray | None = None
    kind: str = "mean"
    parameters: dict[str, Any] = field(default_factory=dict)
    temporal_mode: TemporalMode = TemporalMode.FULL_SAMPLE
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "indices": list(self.indices),
            "scores": None if self.scores is None else self.scores.tolist(),
            "kind": self.kind,
            "parameters": dict(self.parameters),
            "temporal_mode": self.temporal_mode.value,
            "metadata": dict(self.metadata),
        }


def as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def finite_mask(x: np.ndarray) -> np.ndarray:
    return np.isfinite(x)

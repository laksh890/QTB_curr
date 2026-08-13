"""Aggregate regime detection results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.regimes.base.forecast import RegimeForecast
from iqrp.app.regimes.base.persistence import PersistenceReport
from iqrp.app.regimes.base.probabilities import ProbabilityBundle
from iqrp.app.regimes.base.state import RegimeState
from iqrp.app.regimes.base.transition import RegimeTransition


@dataclass
class RegimeResult:
    """Full output of a regime detection run."""

    model_name: str
    model_version: str
    states: list[RegimeState]
    state_ids: np.ndarray
    state_probabilities: np.ndarray
    transition_matrix: np.ndarray
    transitions: list[RegimeTransition] = field(default_factory=list)
    probabilities: ProbabilityBundle | None = None
    persistence: PersistenceReport | None = None
    forecast: RegimeForecast | None = None
    timestamps: list[datetime | None] = field(default_factory=list)
    feature_columns: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_frame(self, timestamp_column: str = "open_time") -> pl.DataFrame:
        data: dict[str, Any] = {
            "state_id": self.state_ids.tolist(),
            "state_name": [s.state_name for s in self.states],
            "probability": [s.probability for s in self.states],
            "confidence": [s.confidence for s in self.states],
            "persistence": [s.persistence for s in self.states],
        }
        if self.timestamps and any(t is not None for t in self.timestamps):
            data[timestamp_column] = self.timestamps
        k = self.state_probabilities.shape[1] if self.state_probabilities.ndim == 2 else 0
        if k:
            for j in range(k):
                data[f"proba_{j}"] = self.state_probabilities[:, j].tolist()
        return pl.DataFrame(data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "states": [s.to_dict() for s in self.states],
            "state_ids": np.asarray(self.state_ids).tolist(),
            "state_probabilities": np.asarray(self.state_probabilities).tolist(),
            "transition_matrix": np.asarray(self.transition_matrix).tolist(),
            "transitions": [t.to_dict() for t in self.transitions],
            "probabilities": None if self.probabilities is None else self.probabilities.to_dict(),
            "persistence": None if self.persistence is None else self.persistence.to_dict(),
            "forecast": None if self.forecast is None else self.forecast.to_dict(),
            "timestamps": [
                t.isoformat() if isinstance(t, datetime) else t for t in self.timestamps
            ],
            "feature_columns": list(self.feature_columns),
            "metadata": dict(self.metadata),
        }

"""Label contract and metadata."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import polars as pl


@dataclass(frozen=True, slots=True)
class LabelMeta:
    """Immutable descriptor for a registered label."""

    name: str
    version: str
    description: str
    category: str
    prediction_horizon: int
    required_inputs: tuple[str, ...] = ()
    output_columns: tuple[str, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)
    dependencies: tuple[str, ...] = ()
    source: str = "iqrp.labels"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "category": self.category,
            "prediction_horizon": self.prediction_horizon,
            "required_inputs": list(self.required_inputs),
            "output_columns": list(self.output_columns),
            "parameters": dict(self.parameters),
            "dependencies": list(self.dependencies),
            "source": self.source,
            "created_at": self.created_at.isoformat(),
        }


class Label(ABC):
    """Base class for all label generators."""

    meta: LabelMeta

    @abstractmethod
    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Compute label columns from ``frame``."""

    def validate_inputs(self, frame: pl.DataFrame) -> None:
        missing = [c for c in self.meta.required_inputs if c not in frame.columns]
        if missing:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(
                f"Label '{self.meta.name}' missing inputs: {missing}",
                code="LABEL_MISSING_INPUTS",
                details={"label": self.meta.name, "missing": missing},
            )

    def run(self, frame: pl.DataFrame) -> pl.DataFrame:
        self.validate_inputs(frame)
        result = self.compute(frame)
        missing_out = [c for c in self.meta.output_columns if c not in result.columns]
        if missing_out:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(
                f"Label '{self.meta.name}' did not produce: {missing_out}",
                code="LABEL_MISSING_OUTPUT",
                details={"label": self.meta.name, "missing": missing_out},
            )
        return result

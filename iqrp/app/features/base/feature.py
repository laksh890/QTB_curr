"""Feature contract and metadata-backed abstract feature definition."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import polars as pl


@dataclass(frozen=True, slots=True)
class FeatureMeta:
    """Immutable descriptor for a registered feature."""

    name: str
    version: str
    description: str
    category: str
    dependencies: tuple[str, ...] = ()
    required_columns: tuple[str, ...] = ()
    output_columns: tuple[str, ...] = ()
    window: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    source: str = "iqrp.features"
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "category": self.category,
            "dependencies": list(self.dependencies),
            "required_columns": list(self.required_columns),
            "output_columns": list(self.output_columns),
            "window": self.window,
            "parameters": dict(self.parameters),
            "source": self.source,
            "created_at": self.created_at.isoformat(),
        }


class Feature(ABC):
    """Base class for all feature generators.

    Implementations must set :attr:`meta` and implement :meth:`compute`.
    ``compute`` receives a Polars frame and returns a frame containing at least
    the declared ``output_columns`` (and typically the join key ``open_time``).
    """

    meta: FeatureMeta

    @abstractmethod
    def compute(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Compute feature columns from ``frame``."""

    def validate_inputs(self, frame: pl.DataFrame) -> None:
        missing = [c for c in self.meta.required_columns if c not in frame.columns]
        if missing:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(
                f"Feature '{self.meta.name}' missing columns: {missing}",
                code="FEATURE_MISSING_COLUMNS",
                details={"feature": self.meta.name, "missing": missing},
            )

    def run(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Validate inputs then compute."""
        self.validate_inputs(frame)
        result = self.compute(frame)
        missing_out = [c for c in self.meta.output_columns if c not in result.columns]
        if missing_out:
            from iqrp.app.core.exceptions import ValidationError

            raise ValidationError(
                f"Feature '{self.meta.name}' did not produce: {missing_out}",
                code="FEATURE_MISSING_OUTPUT",
                details={"feature": self.meta.name, "missing": missing_out},
            )
        return result

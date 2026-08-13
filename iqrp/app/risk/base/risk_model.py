"""Abstract risk model contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from iqrp.app.risk.base.risk_measure import RiskMeasure, RiskReport


class RiskModel(ABC):
    """Base class for risk models — measurements only, never alpha."""

    name: str = "risk_model"
    version: str = "1.0.0"

    @abstractmethod
    def calculate(self, *args: Any, **kwargs: Any) -> RiskMeasure | RiskReport | dict[str, Any]:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version}

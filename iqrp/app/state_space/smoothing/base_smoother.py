"""Base smoother interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from iqrp.app.state_space.base.filter_result import FilterResult
from iqrp.app.state_space.base.smoother_result import SmootherResult
from iqrp.app.state_space.config import StateSpaceSettings


class BaseSmoother(ABC):
    def __init__(self, settings: StateSpaceSettings | None = None) -> None:
        self.settings = settings or StateSpaceSettings.default()

    @abstractmethod
    def run(
        self,
        log_emissions: Any,
        transition: Any,
        *,
        initial: Any | None = None,
        filter_result: FilterResult | None = None,
        lag: int | None = None,
    ) -> SmootherResult:
        """Produce smoothed state probabilities."""

    def hard_states(self, probabilities: np.ndarray) -> np.ndarray:
        return np.argmax(np.asarray(probabilities, dtype=np.float64), axis=1).astype(np.int64)

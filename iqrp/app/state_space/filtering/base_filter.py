"""Base filter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from iqrp.app.state_space.base.filter_result import FilterResult
from iqrp.app.state_space.config import StateSpaceSettings


class BaseFilter(ABC):
    """Abstract filtering algorithm over discrete latent states."""

    def __init__(self, settings: StateSpaceSettings | None = None) -> None:
        self.settings = settings or StateSpaceSettings.default()

    @abstractmethod
    def run(
        self,
        log_emissions: Any,
        transition: Any,
        *,
        initial: Any | None = None,
    ) -> FilterResult:
        """Execute the filter on ``(T, K)`` log-emission scores."""

    def hard_states(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.asarray(probabilities, dtype=np.float64)
        if p.ndim == 1:
            return np.asarray([int(np.argmax(p))], dtype=np.int64)
        return np.argmax(p, axis=1).astype(np.int64)

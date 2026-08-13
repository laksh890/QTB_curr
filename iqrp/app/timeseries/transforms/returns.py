"""Return transforms (thin wrappers over package transforms)."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode
from iqrp.app.timeseries.transforms import log_returns as _log_returns
from iqrp.app.timeseries.transforms import simple_returns as _simple_returns


def log_returns(x: np.ndarray | list[float]) -> AnalysisResult:
    """Causal log-return transform r_t = log(x_t) - log(x_{t-1})."""
    out = _log_returns(x)
    return AnalysisResult(
        method="transform.log_returns",
        value=out,
        temporal_mode=TemporalMode.CAUSAL,
        parameters={},
        metadata={"leakage_safe": True},
    )


def simple_returns(x: np.ndarray | list[float]) -> AnalysisResult:
    """Causal simple-return transform r_t = (x_t - x_{t-1}) / x_{t-1}."""
    out = _simple_returns(x)
    return AnalysisResult(
        method="transform.simple_returns",
        value=out,
        temporal_mode=TemporalMode.CAUSAL,
        parameters={},
        metadata={"leakage_safe": True},
    )


# re-exports of array-level helpers for callers that want ndarray only
returns_log = _log_returns
returns_simple = _simple_returns

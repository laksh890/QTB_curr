"""Unified diagnostics entrypoint."""

from iqrp.app.timeseries.diagnostics.structural_breaks import (
    distribution_shift,
    full_diagnostics,
    heteroskedasticity,
    seasonality_diagnostics,
    structural_breaks,
)

__all__ = [
    "structural_breaks",
    "distribution_shift",
    "heteroskedasticity",
    "seasonality_diagnostics",
    "full_diagnostics",
]

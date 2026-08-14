"""Unified diagnostics entrypoint."""

from iqrp.app.timeseries.diagnostics.structural_breaks import (
    distribution_shift,
    full_diagnostics,
    heteroskedasticity,
    seasonality_diagnostics,
    structural_breaks,
)

__all__ = [
    "distribution_shift",
    "full_diagnostics",
    "heteroskedasticity",
    "seasonality_diagnostics",
    "structural_breaks",
]

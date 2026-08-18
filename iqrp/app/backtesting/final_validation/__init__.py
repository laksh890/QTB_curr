"""Prompt 42 — Institutional-grade final trading validation."""

from iqrp.app.backtesting.final_validation.protocol import FinalValidationConfig, DISCLAIMER
from iqrp.app.backtesting.final_validation.runner import run_final_validation

__all__ = ["FinalValidationConfig", "DISCLAIMER", "run_final_validation"]

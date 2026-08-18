"""Independent OOS validation of frozen Prompt-42 candidates."""

from iqrp.app.backtesting.independent_validation.protocol import (
    DISCLAIMER,
    IndependentValidationConfig,
)
from iqrp.app.backtesting.independent_validation.runner import run_independent_validation

__all__ = ["DISCLAIMER", "IndependentValidationConfig", "run_independent_validation"]

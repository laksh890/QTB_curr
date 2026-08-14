"""Candidate consolidation package (Prompt 40)."""

from iqrp.app.backtesting.alpha_research.consolidation.protocol import (
    DISCLAIMER,
    ConsolidationConfig,
)
from iqrp.app.backtesting.alpha_research.consolidation.runner import run_consolidation

__all__ = ["DISCLAIMER", "ConsolidationConfig", "run_consolidation"]

"""Execution slippage engine: estimation, impact, and realized analytics."""

from __future__ import annotations

from iqrp.app.execution.slippage.estimator import estimate_slippage
from iqrp.app.execution.slippage.historical import (
    HistoricalSlippageModel,
    HistoricalSlippageRecord,
    historical_slippage_bps,
)
from iqrp.app.execution.slippage.liquidity import liquidity_slippage
from iqrp.app.execution.slippage.market_impact import market_impact, path_impact
from iqrp.app.execution.slippage.model import (
    ExecutionSlippageModel,
    SlippageBreakdown,
    SlippageModel,
)
from iqrp.app.execution.slippage.nonlinear_impact import impact_curve, nonlinear_impact
from iqrp.app.execution.slippage.realized import compare_expected_realized, realized_slippage
from iqrp.app.execution.slippage.spread import effective_spread_bps, spread_slippage
from iqrp.app.execution.slippage.volatility import volatility_slippage

__all__ = [
    "ExecutionSlippageModel",
    "HistoricalSlippageModel",
    "HistoricalSlippageRecord",
    "SlippageBreakdown",
    "SlippageModel",
    "compare_expected_realized",
    "effective_spread_bps",
    "estimate_slippage",
    "historical_slippage_bps",
    "impact_curve",
    "liquidity_slippage",
    "market_impact",
    "nonlinear_impact",
    "path_impact",
    "realized_slippage",
    "spread_slippage",
    "volatility_slippage",
]

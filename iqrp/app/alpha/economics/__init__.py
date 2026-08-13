"""Alpha economics: costs, turnover, capacity, and scalability."""

from __future__ import annotations

from iqrp.app.alpha.economics.capacity import capacity_decay, estimate_capacity
from iqrp.app.alpha.economics.market_impact import market_impact_bps
from iqrp.app.alpha.economics.scalability import scalability_curve, scalability_report
from iqrp.app.alpha.economics.slippage import slippage_bps
from iqrp.app.alpha.economics.transaction_costs import estimate_transaction_cost
from iqrp.app.alpha.economics.turnover import average_turnover, turnover_series

__all__ = [
    "average_turnover",
    "capacity_decay",
    "estimate_capacity",
    "estimate_transaction_cost",
    "market_impact_bps",
    "scalability_curve",
    "scalability_report",
    "slippage_bps",
    "turnover_series",
]

"""Smart order routing: multi-venue selection, scoring, allocation, fallback.

Never routes when venue/instrument/trading/order-type/price/qty/kill-switch/risk
checks fail. Execution never generates alpha or overrides hard risk limits.
"""

from iqrp.app.execution.smart_routing.allocation import (
    AllocationMode,
    AllocationPlan,
    VenueAllocation,
    allocate_quantity,
)
from iqrp.app.execution.smart_routing.cost_model import VenueCostEstimate, estimate_venue_cost
from iqrp.app.execution.smart_routing.fallback import (
    FallbackChain,
    FallbackStep,
    build_fallback_chain,
    select_fallback,
)
from iqrp.app.execution.smart_routing.liquidity import (
    LiquiditySnapshot,
    aggregate_fillable,
    assess_liquidity,
)
from iqrp.app.execution.smart_routing.router import (
    RejectionReason,
    RoutingDecision,
    RoutingOrder,
    SmartRouter,
    normalize_order,
)
from iqrp.app.execution.smart_routing.scoring import (
    DEFAULT_WEIGHTS,
    ScoreWeights,
    VenueScore,
    rank_venues,
    score_venue,
)
from iqrp.app.execution.smart_routing.venue import (
    SimulatedVenue,
    Venue,
    VenueInterface,
    VenueOrderRequest,
    VenueResponse,
    VenueResponseStatus,
    as_venue,
)
from iqrp.app.execution.smart_routing.venue_state import VenueState

__all__ = [
    "SmartRouter",
    "RoutingDecision",
    "RoutingOrder",
    "RejectionReason",
    "normalize_order",
    "Venue",
    "VenueState",
    "VenueInterface",
    "VenueOrderRequest",
    "VenueResponse",
    "VenueResponseStatus",
    "SimulatedVenue",
    "as_venue",
    "LiquiditySnapshot",
    "assess_liquidity",
    "aggregate_fillable",
    "VenueCostEstimate",
    "estimate_venue_cost",
    "ScoreWeights",
    "DEFAULT_WEIGHTS",
    "VenueScore",
    "score_venue",
    "rank_venues",
    "AllocationMode",
    "AllocationPlan",
    "VenueAllocation",
    "allocate_quantity",
    "FallbackChain",
    "FallbackStep",
    "build_fallback_chain",
    "select_fallback",
]

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
    "DEFAULT_WEIGHTS",
    "AllocationMode",
    "AllocationPlan",
    "FallbackChain",
    "FallbackStep",
    "LiquiditySnapshot",
    "RejectionReason",
    "RoutingDecision",
    "RoutingOrder",
    "ScoreWeights",
    "SimulatedVenue",
    "SmartRouter",
    "Venue",
    "VenueAllocation",
    "VenueCostEstimate",
    "VenueInterface",
    "VenueOrderRequest",
    "VenueResponse",
    "VenueResponseStatus",
    "VenueScore",
    "VenueState",
    "aggregate_fillable",
    "allocate_quantity",
    "as_venue",
    "assess_liquidity",
    "build_fallback_chain",
    "estimate_venue_cost",
    "normalize_order",
    "rank_venues",
    "score_venue",
    "select_fallback",
]

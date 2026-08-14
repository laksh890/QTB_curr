"""Trading-horizon & short-horizon research engine (research capability).

Separates data timeframe, signal timeframe, and holding period. Does not
fabricate finer-than-native bars. Does not claim live profitability.
"""

from iqrp.app.backtesting.horizon.availability import (
    check_horizon_availability,
    detect_native_frequency,
    filter_available_timeframes,
)
from iqrp.app.backtesting.horizon.config import HorizonResearchConfig
from iqrp.app.backtesting.horizon.engine import HorizonResearchEngine
from iqrp.app.backtesting.horizon.parse import (
    availability_reason,
    can_derive,
    grid_specs,
    parse_holding,
    parse_timeframe,
)
from iqrp.app.backtesting.horizon.ranking import (
    classify_horizon,
    compute_horizon_research_score,
    select_best_robust_horizon,
)
from iqrp.app.backtesting.horizon.types import (
    DEFAULT_CAPITAL_LEVELS,
    DEFAULT_DATA_TIMEFRAMES,
    DEFAULT_HOLDING_BARS,
    HoldingPeriod,
    HorizonResult,
    HorizonSpec,
    HorizonStatus,
    SignalSide,
    Timeframe,
)

__all__ = [
    "DEFAULT_CAPITAL_LEVELS",
    "DEFAULT_DATA_TIMEFRAMES",
    "DEFAULT_HOLDING_BARS",
    "HoldingPeriod",
    "HorizonResearchConfig",
    "HorizonResearchEngine",
    "HorizonResult",
    "HorizonSpec",
    "HorizonStatus",
    "SignalSide",
    "Timeframe",
    "availability_reason",
    "can_derive",
    "check_horizon_availability",
    "classify_horizon",
    "compute_horizon_research_score",
    "detect_native_frequency",
    "filter_available_timeframes",
    "grid_specs",
    "parse_holding",
    "parse_timeframe",
    "select_best_robust_horizon",
]

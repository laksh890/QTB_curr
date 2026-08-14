"""Alpha research types, classifications, and defaults.

Research platform only — not a profitability claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class AlphaClassification(str, Enum):
    ROBUST_ALPHA = "ROBUST_ALPHA"
    PROMISING_ALPHA = "PROMISING_ALPHA"
    FRAGILE_ALPHA = "FRAGILE_ALPHA"
    COST_INEFFICIENT = "COST_INEFFICIENT"
    OOS_FAILURE = "OOS_FAILURE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    SAMPLE_TOO_SHORT = "SAMPLE_TOO_SHORT"


class SignalKind(str, Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"
    CATEGORICAL = "categorical"


DEFAULT_FORWARD_HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10, 20)

# Equity NSE session bars/day (legacy default used by NIFTY research)
EQUITY_BARS_PER_DAY: dict[str, float] = {
    "1m": 375.0,
    "5m": 75.0,
    "15m": 25.0,
    "30m": 13.0,
    "1h": 7.0,
}

# Crypto 24×7 UTC bars/day
CRYPTO_BARS_PER_DAY: dict[str, float] = {
    "1m": 1440.0,
    "5m": 288.0,
    "15m": 96.0,
    "30m": 48.0,
    "1h": 24.0,
}

# Standardized cost scenarios reusing commission/spread/slippage fields
COST_SCENARIOS: dict[str, dict[str, float]] = {
    "BASE": {"commission_bps": 1.0, "spread_bps": 2.0, "slippage_bps": 2.0},
    "MODERATE": {"commission_bps": 2.0, "spread_bps": 4.0, "slippage_bps": 4.0},
    "ADVERSE": {"commission_bps": 4.0, "spread_bps": 8.0, "slippage_bps": 8.0},
}


def bars_per_day(timeframe: str, *, market_type: str = "equity") -> float:
    tf = str(timeframe).lower()
    table = CRYPTO_BARS_PER_DAY if str(market_type).lower() in {"crypto", "cryptocurrency"} else EQUITY_BARS_PER_DAY
    return float(table.get(tf, EQUITY_BARS_PER_DAY.get(tf, 75.0)))


def holding_clock_minutes(timeframe: str, holding_bars: int) -> float:
    minutes = {"1m": 1.0, "5m": 5.0, "15m": 15.0, "30m": 30.0, "1h": 60.0}.get(str(timeframe).lower(), 5.0)
    return float(minutes * int(holding_bars))


class ResearchStatus(str, Enum):
    """Campaign-level research statuses (not production readiness)."""

    REJECT = "REJECT"
    COST_INEFFICIENT = "COST_INEFFICIENT"
    IN_SAMPLE_ONLY = "IN_SAMPLE_ONLY"
    OOS_FAILED = "OOS_FAILED"
    UNSTABLE = "UNSTABLE"
    SAMPLE_INSUFFICIENT = "SAMPLE_INSUFFICIENT"
    CONDITIONAL = "CONDITIONAL"
    CANDIDATE = "CANDIDATE"


def map_alpha_to_research_status(classification: str, metrics: Mapping[str, Any] | None = None) -> str:
    """Map engine AlphaClassification → campaign ResearchStatus."""
    m = dict(metrics or {})
    c = str(classification)
    if c == AlphaClassification.COST_INEFFICIENT.value:
        return ResearchStatus.COST_INEFFICIENT.value
    if c in {AlphaClassification.SAMPLE_TOO_SHORT.value, AlphaClassification.INSUFFICIENT_DATA.value}:
        return ResearchStatus.SAMPLE_INSUFFICIENT.value
    if c == AlphaClassification.OOS_FAILURE.value:
        return ResearchStatus.OOS_FAILED.value
    if c == AlphaClassification.FRAGILE_ALPHA.value:
        return ResearchStatus.UNSTABLE.value
    if c == AlphaClassification.ROBUST_ALPHA.value:
        return ResearchStatus.CANDIDATE.value
    if c == AlphaClassification.PROMISING_ALPHA.value:
        if m.get("oos_evaluated") and float(m.get("oos_sharpe", 0) or 0) <= 0:
            return ResearchStatus.IN_SAMPLE_ONLY.value
        return ResearchStatus.CONDITIONAL.value
    return ResearchStatus.REJECT.value


DEFAULT_ALPHA_SCORE_WEIGHTS: dict[str, float] = {
    "net_sharpe": 0.18,
    "net_expectancy": 0.10,
    "oos": 0.18,
    "ic": 0.12,
    "ic_stability": 0.08,
    "drawdown": 0.10,
    "turnover": 0.06,
    "cost_sensitivity": 0.08,
    "parameter_stability": 0.05,
    "regime_stability": 0.05,
}

DEFAULT_ALPHA_GATES: dict[str, Any] = {
    "min_trades": 10,
    "min_oos_sharpe": 0.0,
    "min_net_expectancy": 0.0,
    "max_drawdown": 0.40,
    "min_neighborhood_stability": 0.35,
    "min_sessions_for_significance": 60,
    "require_positive_net_alpha": True,
}

SAMPLE_TOO_SHORT_DISCLAIMER = (
    "SAMPLE TOO SHORT — development window is too small for statistical "
    "significance claims. Results are pipeline / research validation only, "
    "not evidence of live or robust alpha. Not a profitability claim."
)


@dataclass
class TimeframeContext:
    """Explicit multi-timeframe roles."""

    feature_timeframe: str
    signal_timeframe: str
    execution_timeframe: str

    def to_dict(self) -> dict[str, str]:
        return {
            "feature_timeframe": self.feature_timeframe,
            "signal_timeframe": self.signal_timeframe,
            "execution_timeframe": self.execution_timeframe,
        }


__all__ = [
    "AlphaClassification",
    "COST_SCENARIOS",
    "CRYPTO_BARS_PER_DAY",
    "DEFAULT_ALPHA_GATES",
    "DEFAULT_ALPHA_SCORE_WEIGHTS",
    "DEFAULT_FORWARD_HORIZONS",
    "EQUITY_BARS_PER_DAY",
    "ResearchStatus",
    "SAMPLE_TOO_SHORT_DISCLAIMER",
    "SignalKind",
    "TimeframeContext",
    "bars_per_day",
    "holding_clock_minutes",
    "map_alpha_to_research_status",
]

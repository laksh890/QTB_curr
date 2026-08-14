"""Horizon research configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from iqrp.app.backtesting.horizon.ranking import DEFAULT_ROBUST_GATES, DEFAULT_SCORE_WEIGHTS
from iqrp.app.backtesting.horizon.types import (
    DEFAULT_CAPITAL_LEVELS,
    DEFAULT_DATA_TIMEFRAMES,
    DEFAULT_HOLDING_BARS,
)


@dataclass
class HorizonResearchConfig:
    """Configurable sweep / scoring / cost / capacity settings."""

    data_timeframes: list[str] = field(
        default_factory=lambda: list(DEFAULT_DATA_TIMEFRAMES)
    )
    signal_timeframes: list[str] | None = None
    holding_bars: list[int] = field(default_factory=lambda: list(DEFAULT_HOLDING_BARS))
    capital_levels: list[float] = field(default_factory=lambda: list(DEFAULT_CAPITAL_LEVELS))

    commission_bps: float = 1.0
    spread_bps: float = 2.0
    slippage_bps: float = 2.0
    financing_bps_per_period: float = 0.0
    impact_bps_per_period: float = 0.0

    score_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCORE_WEIGHTS))
    robust_gates: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_ROBUST_GATES))

    train_frac: float = 0.6
    validation_frac: float = 0.2
    train_end: str | None = None
    validation_end: str | None = None

    neighborhood_max_ratio: float = 2.0
    allow_short: bool = True
    signal_params: dict[str, Any] = field(default_factory=lambda: {"lookback": 1})

    capacity_adv: float = 1e8
    capacity_impact_coef: float = 0.1
    capacity_impact_exp: float = 0.5

    periods_per_year: float = 252.0
    strategy_id: str = "horizon_research_momentum"
    instrument: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_timeframes": list(self.data_timeframes),
            "signal_timeframes": list(self.signal_timeframes)
            if self.signal_timeframes is not None
            else None,
            "holding_bars": list(self.holding_bars),
            "capital_levels": list(self.capital_levels),
            "commission_bps": self.commission_bps,
            "spread_bps": self.spread_bps,
            "slippage_bps": self.slippage_bps,
            "score_weights": dict(self.score_weights),
            "robust_gates": dict(self.robust_gates),
            "train_frac": self.train_frac,
            "validation_frac": self.validation_frac,
            "train_end": self.train_end,
            "validation_end": self.validation_end,
            "allow_short": self.allow_short,
            "strategy_id": self.strategy_id,
            "instrument": self.instrument,
            "disclaimer": (
                "Research configuration only. Results are simulated / modelled, "
                "not live performance claims."
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> HorizonResearchConfig:
        d = dict(data or {})
        return cls(
            data_timeframes=list(d.get("data_timeframes") or DEFAULT_DATA_TIMEFRAMES),
            signal_timeframes=d.get("signal_timeframes"),
            holding_bars=list(d.get("holding_bars") or DEFAULT_HOLDING_BARS),
            capital_levels=list(d.get("capital_levels") or DEFAULT_CAPITAL_LEVELS),
            commission_bps=float(d.get("commission_bps", 1.0)),
            spread_bps=float(d.get("spread_bps", 2.0)),
            slippage_bps=float(d.get("slippage_bps", 2.0)),
            financing_bps_per_period=float(d.get("financing_bps_per_period", 0.0)),
            impact_bps_per_period=float(d.get("impact_bps_per_period", 0.0)),
            score_weights=dict(d.get("score_weights") or DEFAULT_SCORE_WEIGHTS),
            robust_gates=dict(d.get("robust_gates") or DEFAULT_ROBUST_GATES),
            train_frac=float(d.get("train_frac", 0.6)),
            validation_frac=float(d.get("validation_frac", 0.2)),
            train_end=d.get("train_end"),
            validation_end=d.get("validation_end"),
            neighborhood_max_ratio=float(d.get("neighborhood_max_ratio", 2.0)),
            allow_short=bool(d.get("allow_short", True)),
            signal_params=dict(d.get("signal_params") or {"lookback": 1}),
            capacity_adv=float(d.get("capacity_adv", 1e8)),
            capacity_impact_coef=float(d.get("capacity_impact_coef", 0.1)),
            capacity_impact_exp=float(d.get("capacity_impact_exp", 0.5)),
            periods_per_year=float(d.get("periods_per_year", 252.0)),
            strategy_id=str(d.get("strategy_id") or "horizon_research_momentum"),
            instrument=str(d.get("instrument") or ""),
        )


__all__ = ["HorizonResearchConfig"]

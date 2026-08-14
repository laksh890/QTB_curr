"""Model→Alpha adapter types (thin integration layer).

Does not implement forecasting models. Research evidence is not a profitability guarantee.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class OutputMappingKind(str, Enum):
    RETURN_THRESHOLD = "return_threshold"
    PROBABILITY_UP = "probability_up"
    VOLATILITY_EXPANSION = "volatility_expansion"
    VOLATILITY_CONTRACTION = "volatility_contraction"
    REGIME_LABEL_MAP = "regime_label_map"
    CONTINUOUS_PASSTHROUGH = "continuous_passthrough"


@dataclass(slots=True)
class SignalMappingConfig:
    """Explicit, configurable forecast→LONG/SHORT/FLAT mapping."""

    kind: OutputMappingKind = OutputMappingKind.RETURN_THRESHOLD
    long_threshold: float = 0.0
    short_threshold: float = 0.0
    # For probability_up: P(up) >= long_prob → LONG; P(up) <= short_prob → SHORT
    long_prob: float = 0.55
    short_prob: float = 0.45
    # For volatility expansion: z-score of forecast vol vs rolling median
    vol_z_threshold: float = 0.5
    vol_lookback: int = 20
    # Regime label → position
    regime_map: dict[str, float] = field(default_factory=dict)
    flat_on_unknown_regime: bool = True
    allow_short: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["kind"] = self.kind.value
        return d


@dataclass(slots=True)
class ModelAdapterSpec:
    """Describes how an EXISTING model is exposed to Alpha Research."""

    adapter_id: str
    model_id: str
    model_family: str
    model_version: str = "1.0.0"
    input_schema: tuple[str, ...] = ("returns",)
    output_type: str = "forecast"  # forecast | regime | probability
    timeframe: str = "1h"
    horizon: int = 1
    causal: bool = True
    signal_mapping: SignalMappingConfig = field(default_factory=SignalMappingConfig)
    availability: str = "available"  # available | unavailable | partial
    status: str = "registered"
    factory_path: str = ""  # documentation only
    notes: str = ""
    holding_bars: int = 1

    @property
    def signal_id(self) -> str:
        # e.g. garch_volatility_v1_1h
        safe = self.adapter_id.replace(".", "_").replace("-", "_")
        return safe

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "model_id": self.model_id,
            "model_family": self.model_family,
            "model_version": self.model_version,
            "input_schema": list(self.input_schema),
            "output_type": self.output_type,
            "timeframe": self.timeframe,
            "horizon": self.horizon,
            "causal": self.causal,
            "signal_mapping": self.signal_mapping.to_dict(),
            "availability": self.availability,
            "status": self.status,
            "factory_path": self.factory_path,
            "notes": self.notes,
            "holding_bars": self.holding_bars,
            "signal_id": self.signal_id,
            "disclaimer": "MODEL→SIGNAL ADAPTER — wiring validation only, not a profitability claim.",
        }


__all__ = [
    "ModelAdapterSpec",
    "OutputMappingKind",
    "SignalMappingConfig",
]

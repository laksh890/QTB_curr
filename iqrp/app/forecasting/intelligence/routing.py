"""Model routing by asset, regime, volatility, liquidity, timeframe, confidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from iqrp.app.forecasting.intelligence.config import RoutingConfig


@dataclass
class RoutingTable:
    by_regime: dict[str, str] = field(default_factory=dict)
    by_asset: dict[str, str] = field(default_factory=dict)
    by_timeframe: dict[str, str] = field(default_factory=dict)
    default_model: str = "mock"
    high_vol_model: str | None = None
    low_confidence_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "by_regime": dict(self.by_regime),
            "by_asset": dict(self.by_asset),
            "by_timeframe": dict(self.by_timeframe),
            "default_model": self.default_model,
            "high_vol_model": self.high_vol_model,
            "low_confidence_model": self.low_confidence_model,
        }


def route_model(
    frame: pl.DataFrame,
    table: RoutingTable,
    *,
    config: RoutingConfig,
    confidence: float | None = None,
) -> str:
    if not config.enabled:
        return table.default_model
    # confidence routing
    if config.by_confidence and confidence is not None and confidence < 0.4:
        if table.low_confidence_model:
            return table.low_confidence_model
    # volatility routing
    if config.by_volatility and config.vol_column in frame.columns:
        vol = float(frame[config.vol_column].to_numpy()[-1])
        med = float(np.median(frame[config.vol_column].to_numpy()))
        if vol > med * 1.5 and table.high_vol_model:
            return table.high_vol_model
    # regime routing
    if config.by_regime and config.regime_column in frame.columns:
        reg = str(frame[config.regime_column].to_list()[-1])
        if reg in table.by_regime:
            return table.by_regime[reg]
    # asset routing
    if "asset_id" in frame.columns:
        asset = str(frame["asset_id"].to_list()[-1])
        if asset in table.by_asset:
            return table.by_asset[asset]
    # liquidity proxy
    if "spread" in frame.columns:
        spread = float(frame["spread"].to_numpy()[-1])
        if spread > float(np.median(frame["spread"].to_numpy())) * 2 and table.low_confidence_model:
            return table.low_confidence_model
    return table.default_model


def build_routing_table(
    default_model: str,
    *,
    regime_models: dict[str, str] | None = None,
    asset_models: dict[str, str] | None = None,
    high_vol_model: str | None = None,
) -> RoutingTable:
    return RoutingTable(
        default_model=default_model,
        by_regime=dict(regime_models or {}),
        by_asset=dict(asset_models or {}),
        high_vol_model=high_vol_model or default_model,
        low_confidence_model=default_model,
    )

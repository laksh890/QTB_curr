"""Hydra-backed settings for the Institutional Backtesting Platform.

CRITICAL RULES
--------------
- No event handler may access data after the event timestamp.
- Look-ahead / leakage / invalid universe → invalidate the backtest.
- Every run must record data / feature / model / code versions and seed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class ClockConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    frequency: Literal["tick", "second", "minute", "hourly", "daily", "custom"] = "daily"
    timezone: str = "UTC"
    # Used only when frequency == "custom" (seconds).
    custom_step_seconds: float = 86400.0


class EventEngineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    advance_empty_ticks: bool = False
    max_events: int | None = None
    enforce_pit: bool = True
    invalidate_on_lookahead: bool = True


class PITConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enforce_no_lookahead: bool = True
    enforce_universe_asof: bool = True
    detect_leakage: bool = True
    max_label_horizon: int | None = None


class CorporateActionsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    adjust_splits: bool = True
    adjust_dividends: bool = True
    handle_mergers: bool = True
    handle_delistings: bool = True
    handle_symbol_changes: bool = True
    dividend_method: Literal["subtract", "factor"] = "subtract"


class CostsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    commission_bps: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    financing_bps: float = 0.0
    borrow_bps: float = 0.0


class LatencyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    signal_ms: float = 0.0
    decision_ms: float = 0.0
    order_ms: float = 0.0
    execution_ms: float = 0.0
    market_data_ms: float = 0.0


class WalkForwardConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: Literal["rolling", "expanding", "anchored"] = "rolling"
    train_periods: int = 252
    validation_periods: int = 63
    test_periods: int = 21
    purge_periods: int = 0
    embargo_periods: int = 0


class ReproducibilityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int = 42
    data_version: str = "1.0.0"
    feature_version: str = "1.0.0"
    label_version: str = "1.0.0"
    model_version: str = "1.0.0"
    risk_version: str = "1.0.0"
    portfolio_version: str = "1.0.0"
    execution_version: str = "1.0.0"
    code_version: str = "1.0.0"


class ReportingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    write_markdown: bool = True
    write_json: bool = True
    write_scorecard: bool = True


class BacktestSettings(BaseModel):
    """Top-level backtesting configuration."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    name: str = "default_backtest"
    initial_cash: float = 1_000_000.0
    clock: ClockConfig = Field(default_factory=ClockConfig)
    event_engine: EventEngineConfig = Field(default_factory=EventEngineConfig)
    pit: PITConfig = Field(default_factory=PITConfig)
    corporate_actions: CorporateActionsConfig = Field(default_factory=CorporateActionsConfig)
    costs: CostsConfig = Field(default_factory=CostsConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @classmethod
    def from_mapping(cls, data: Any) -> BacktestSettings:
        try:
            if hasattr(data, "items") and not isinstance(data, dict):
                data = OmegaConf.to_container(data, resolve=True)
            return cls.model_validate(dict(data or {}))
        except Exception as exc:
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid backtesting settings: {exc}",
                code="BACKTEST_CONFIG_INVALID",
            ) from exc

    @classmethod
    def from_hydra(
        cls,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
    ) -> BacktestSettings:
        path = Path(config_path) if config_path else _default_config_path()
        cfg: Any = OmegaConf.create({})
        if path.is_file():
            cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cls.from_mapping(OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    def default(cls) -> BacktestSettings:
        path = _default_config_path()
        if path.is_file():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "backtesting" / "default.yaml"


__all__ = [
    "BacktestSettings",
    "ClockConfig",
    "CorporateActionsConfig",
    "CostsConfig",
    "EventEngineConfig",
    "LatencyConfig",
    "PITConfig",
    "ReportingConfig",
    "ReproducibilityConfig",
    "WalkForwardConfig",
]

"""Hydra-backed settings for Institutional Execution Order Manager.

CRITICAL RULES
--------------
- Execution never generates alpha.
- Never override hard risk limits.
- Urgency influences aggressiveness but NEVER overrides hard risk.
- Idempotent fills/events; no future information.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class TickLotConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_tick_size: float = 0.01
    default_lot_size: float = 1.0
    min_qty: float = 1.0
    max_qty: float = 1_000_000.0


class PriceBandConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    band_pct: float = 0.10  # reject limit prices more than 10% from reference
    require_reference: bool = False


class CapitalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    check_enabled: bool = True
    max_notional: float = 10_000_000.0
    max_order_notional: float = 5_000_000.0


class RiskConfig(BaseModel):
    """Hard risk gates — urgency/alpha NEVER override these."""

    model_config = ConfigDict(frozen=True)

    enforce_hard_limits: bool = True
    require_risk_callback: bool = False
    max_participation: float = 0.10


class KillSwitchConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    check_on_submit: bool = True
    check_global: bool = True
    check_account: bool = True
    check_venue: bool = True
    check_strategy: bool = True


class ReconciliationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    qty_tolerance: float = 0.0
    notional_tolerance: float = 0.01
    alert_on_diff: bool = True


class FillConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    idempotent: bool = True
    allow_overfill: bool = False


class ExecutionSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int = 42
    data_version: str = "1.0.0"
    model_version: str = "1.0.0"
    default_time_in_force: Literal["DAY", "GTC", "IOC", "FOK", "GTD", "OPG", "CLS"] = "DAY"
    default_urgency: Literal["LOW", "NORMAL", "HIGH", "CRITICAL"] = "NORMAL"
    default_venue: str = "SIM"
    tick_lot: TickLotConfig = Field(default_factory=TickLotConfig)
    price_bands: PriceBandConfig = Field(default_factory=PriceBandConfig)
    capital: CapitalConfig = Field(default_factory=CapitalConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    kill_switch: KillSwitchConfig = Field(default_factory=KillSwitchConfig)
    reconciliation: ReconciliationConfig = Field(default_factory=ReconciliationConfig)
    fills: FillConfig = Field(default_factory=FillConfig)

    @classmethod
    def from_mapping(cls, data: Any) -> ExecutionSettings:
        try:
            if hasattr(data, "items") and not isinstance(data, dict):
                data = OmegaConf.to_container(data, resolve=True)
            return cls.model_validate(dict(data or {}))
        except Exception as exc:
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid execution settings: {exc}",
                code="EXECUTION_CONFIG_INVALID",
            ) from exc

    @classmethod
    def from_hydra(
        cls,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
    ) -> ExecutionSettings:
        path = Path(config_path) if config_path else _default_config_path()
        cfg: Any = OmegaConf.create({})
        if path.is_file():
            cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cls.from_mapping(OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    def default(cls) -> ExecutionSettings:
        path = _default_config_path()
        if path.is_file():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "execution" / "default.yaml"

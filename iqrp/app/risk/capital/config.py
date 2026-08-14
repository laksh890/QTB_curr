"""Hydra-backed settings for Institutional Capital Allocation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class DrawdownThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    caution: float = 0.05
    reduced_risk: float = 0.10
    capital_preservation: float = 0.15
    trading_halt: float = 0.20


class CapitalSettings(BaseModel):
    """Frozen capital allocation settings; hard limits are never softened by confidence."""

    model_config = ConfigDict(frozen=True)

    seed: int = 42
    data_version: str = "1.0.0"
    model_version: str = "1.0.0"
    method: str = "risk_parity"

    max_weight: float = 0.40
    min_weight: float = 0.0
    max_gross_exposure: float = 1.5
    max_net_exposure: float = 1.0
    max_leverage: float = 2.0
    max_concentration: float = 0.40
    max_turnover: float = 0.50
    max_participation: float = 0.10
    min_adv_coverage: float = 0.01

    missing_capacity_scale: float = 0.50
    missing_liquidity_scale: float = 0.50
    default_adv: float = 1.0e6
    default_spread: float = 0.002
    impact_coeff: float = 0.10
    capacity_ttl_days: float = 5.0

    correlation_crowding_threshold: float = 0.60
    correlation_scale_floor: float = 0.25

    drawdown: DrawdownThresholds = Field(default_factory=DrawdownThresholds)

    risk_state_scales: dict[str, float] = Field(
        default_factory=lambda: {
            "NORMAL": 1.0,
            "CAUTION": 0.8,
            "REDUCED_RISK": 0.5,
            "CAPITAL_PRESERVATION": 0.25,
            "TRADING_HALT": 0.0,
        }
    )
    regime_scales: dict[str, float] = Field(
        default_factory=lambda: {
            "normal": 1.0,
            "low_vol": 1.0,
            "high_vol": 0.55,
            "stress": 0.35,
            "crisis": 0.25,
            "transition": 0.60,
        }
    )

    target_volatility: float = 0.10
    vol_floor: float = 1.0e-4

    hrp_linkage: str = "single"
    risk_parity_max_iter: int = 500
    risk_parity_tol: float = 1.0e-8

    rebalance_turnover_cap: float = 0.50
    rebalance_participation_cap: float = 0.10

    @classmethod
    def from_mapping(cls, data: Any) -> CapitalSettings:
        try:
            if hasattr(data, "items") and not isinstance(data, dict):
                data = OmegaConf.to_container(data, resolve=True)
            return cls.model_validate(dict(data or {}))
        except Exception as exc:
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid capital settings: {exc}", code="CAPITAL_CONFIG_INVALID"
            ) from exc

    @classmethod
    def from_hydra(
        cls,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
    ) -> CapitalSettings:
        path = Path(config_path) if config_path else _default_config_path()
        cfg: Any = OmegaConf.create({})
        if path.is_file():
            cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cls.from_mapping(OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    def default(cls) -> CapitalSettings:
        path = _default_config_path()
        if path.is_file():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "risk" / "capital" / "default.yaml"

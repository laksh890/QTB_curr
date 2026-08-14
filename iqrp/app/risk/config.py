"""Hydra-backed settings for Institutional Risk Intelligence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class VaRConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["historical", "parametric", "monte_carlo", "fhs"] = "historical"
    confidence: float = 0.95
    horizon: int = 1
    n_simulations: int = 5000


class ESConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["historical", "parametric", "monte_carlo"] = "historical"
    confidence: float = 0.95


class SizingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal[
        "fixed_fractional",
        "volatility_target",
        "risk_parity",
        "kelly",
        "fractional_kelly",
        "drawdown_adjusted",
    ] = "volatility_target"
    target_volatility: float = 0.10
    kelly_fraction: float = 0.25
    max_kelly: float = 0.5
    max_leverage: float = 2.0
    risk_per_trade: float = 0.01


class DrawdownConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    caution: float = 0.05
    reduced_risk: float = 0.10
    capital_preservation: float = 0.15
    trading_halt: float = 0.20


class LimitConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_position: float = 0.10
    max_gross_exposure: float = 1.5
    max_net_exposure: float = 1.0
    max_concentration: float = 0.25
    max_daily_loss: float = 0.03
    max_leverage: float = 2.0
    max_participation: float = 0.10
    min_adv_coverage: float = 0.01


class LeverageConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_leverage: float = 1.0
    max_leverage: float = 2.0
    min_leverage: float = 0.0
    vol_scalar: float = 1.0
    confidence_cap: float = 1.25  # confidence cannot authorize unlimited leverage


class MonteCarloConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_simulations: int = 5000
    horizon: int = 1
    seed: int = 42
    block_size: int = 5


class RiskSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    var: VaRConfig = Field(default_factory=VaRConfig)
    es: ESConfig = Field(default_factory=ESConfig)
    sizing: SizingConfig = Field(default_factory=SizingConfig)
    drawdown: DrawdownConfig = Field(default_factory=DrawdownConfig)
    limits: LimitConfig = Field(default_factory=LimitConfig)
    leverage: LeverageConfig = Field(default_factory=LeverageConfig)
    monte_carlo: MonteCarloConfig = Field(default_factory=MonteCarloConfig)
    seed: int = 42
    data_version: str = "1.0.0"
    model_version: str = "1.0.0"

    @classmethod
    def from_mapping(cls, data: Any) -> RiskSettings:
        try:
            if hasattr(data, "items") and not isinstance(data, dict):
                data = OmegaConf.to_container(data, resolve=True)
            return cls.model_validate(dict(data or {}))
        except Exception as exc:
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid risk settings: {exc}", code="RISK_CONFIG_INVALID"
            ) from exc

    @classmethod
    def from_hydra(
        cls,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
    ) -> RiskSettings:
        path = Path(config_path) if config_path else _default_config_path()
        cfg: Any = OmegaConf.create({})
        if path.is_file():
            cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cls.from_mapping(OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    def default(cls) -> RiskSettings:
        path = _default_config_path()
        if path.is_file():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "risk" / "default.yaml"

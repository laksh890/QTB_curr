"""Hydra-backed settings for Institutional Portfolio Construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class CovarianceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["sample", "ewma", "shrinkage", "ledoit_wolf", "factor", "robust"] = "shrinkage"
    ewma_lambda: float = 0.94
    shrinkage_intensity: float | None = None


class ExpectedReturnsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["forecast", "historical", "shrinkage", "black_litterman"] = "forecast"
    confidence_shrink: bool = True
    bl_tau: float = 0.05
    bl_risk_aversion: float = 1.0


class ConstraintsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_net: float | None = None
    max_long: float | None = None
    max_short: float | None = None
    max_concentration: float | None = None
    dollar_neutral: bool = False


class ObjectiveConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    risk_free_rate: float = 0.0
    turnover_penalty: float = 0.0
    cvar_confidence: float = 0.95


class PortfolioSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal[
        "min_variance",
        "mean_variance",
        "max_sharpe",
        "max_diversification",
        "risk_parity",
        "erc",
        "hrp",
        "herc",
        "min_cvar",
        "cvar",
        "drawdown",
        "turnover_aware",
        "robust",
        "black_litterman",
        "entropy",
        "risk_budget",
        "multi_objective",
    ] = "mean_variance"
    long_only: bool = True
    max_weight: float = 0.4
    max_gross: float = 1.5
    max_leverage: float = 2.0
    max_turnover: float = 0.5
    risk_aversion: float = 1.0
    fallback: Literal["current", "min_variance", "cash"] = "current"
    require_risk_validation: bool = True
    seed: int = 42
    data_version: str = "1.0.0"
    model_version: str = "1.0.0"
    covariance: CovarianceConfig = Field(default_factory=CovarianceConfig)
    expected_returns: ExpectedReturnsConfig = Field(default_factory=ExpectedReturnsConfig)
    constraints: ConstraintsConfig = Field(default_factory=ConstraintsConfig)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)

    @classmethod
    def from_mapping(cls, data: Any) -> PortfolioSettings:
        try:
            if hasattr(data, "items") and not isinstance(data, dict):
                data = OmegaConf.to_container(data, resolve=True)
            return cls.model_validate(dict(data or {}))
        except Exception as exc:  # noqa: BLE001
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid portfolio settings: {exc}",
                code="PORTFOLIO_CONFIG_INVALID",
            ) from exc

    @classmethod
    def from_hydra(
        cls,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
    ) -> PortfolioSettings:
        path = Path(config_path) if config_path else _default_config_path()
        cfg: Any = OmegaConf.create({})
        if path.is_file():
            cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cls.from_mapping(OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    def default(cls) -> PortfolioSettings:
        path = _default_config_path()
        if path.is_file():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "portfolio" / "default.yaml"

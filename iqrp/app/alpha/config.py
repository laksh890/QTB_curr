"""Hydra-backed settings for Institutional Alpha Research.

CRITICAL RULES baked into defaults / docs:
- Statistical significance alone ≠ alpha.
- Historical Sharpe alone cannot approve.
- Must track economic_hypothesis on SignalDefinition.
- Point-in-time: no future leakage in signal computation helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class DiscoveryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    momentum_lookbacks: tuple[int, ...] = (10, 20, 60)
    mean_reversion_lookbacks: tuple[int, ...] = (5, 10, 20)
    trend_fast: int = 10
    trend_slow: int = 40
    volatility_lookback: int = 20
    volume_lookback: int = 20
    statistical_min_abs_ic: float = 0.02
    statistical_min_obs: int = 30
    auto_register: bool = True
    publication_lag_default: int = 1


class ResearchConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    horizons: tuple[int, ...] = (1, 2, 5, 10)
    stability_window: int = 60
    stability_step: int = 10
    stability_min_obs: int = 30
    seasonality_period: int = 5
    predictor_min_train: int = 60
    predictor_test_size: int = 20
    predictor_step: int = 20
    ridge_alpha: float = 1.0


class ScoringConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    weight_predictive: float = 0.40
    weight_stability: float = 0.25
    weight_persistence: float = 0.15
    weight_economic_hypothesis: float = 0.20
    require_economic_hypothesis: bool = True
    min_hypothesis_chars: int = 20
    allow_sharpe_only_approval: bool = False  # Historical Sharpe alone cannot approve


class GovernanceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    preserve_rejected: bool = True
    auditable_transitions: bool = True
    terminal_statuses: tuple[str, ...] = ("REJECTED", "RETIRED")


class AlphaSettings(BaseModel):
    """Top-level alpha research settings."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    seed: int = 42
    data_version: str = "1.0.0"
    model_version: str = "1.0.0"
    owner_default: str = "research"
    universe_default: str = "default"
    frequency_default: str = "1d"
    output_dir: str = "data/reports/alpha_research"
    cache_dir: str = "data/cache/alpha_research"
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)

    @classmethod
    def from_mapping(cls, data: Any) -> AlphaSettings:
        try:
            if hasattr(data, "items") and not isinstance(data, dict):
                data = OmegaConf.to_container(data, resolve=True)
            return cls.model_validate(dict(data or {}))
        except Exception as exc:  # noqa: BLE001
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid alpha settings: {exc}",
                code="ALPHA_CONFIG_INVALID",
            ) from exc

    @classmethod
    def from_hydra(
        cls,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
    ) -> AlphaSettings:
        path = Path(config_path) if config_path else _default_config_path()
        cfg: Any = OmegaConf.create({})
        if path.is_file():
            cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cls.from_mapping(OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    def default(cls) -> AlphaSettings:
        path = _default_config_path()
        if path.is_file():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "alpha" / "default.yaml"

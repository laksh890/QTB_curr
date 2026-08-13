"""Hydra-backed configuration for the Market Simulation Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class MarketConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str = "SYNTH"
    exchange: str = "sim"
    timeframe: str = "1h"
    market_hours: int = 24
    tick_size: float = 0.01
    lot_size: float = 0.001


class DynamicsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    drift: float = 0.05
    volatility: float = 0.2
    mean_reversion_speed: float = 1.0
    mean_reversion_level: float = 100.0
    jump_intensity: float = 5.0
    jump_mean: float = -0.02
    jump_std: float = 0.04
    heston_kappa: float = 2.0
    heston_theta: float = 0.04
    heston_xi: float = 0.3
    heston_rho: float = -0.7
    vg_theta: float = -0.1
    vg_sigma: float = 0.2
    vg_nu: float = 0.2
    cir_kappa: float = 1.5
    cir_theta: float = 0.04
    cir_sigma: float = 0.1


class NoiseConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    distribution: Literal["gaussian", "student_t", "laplace", "cauchy", "uniform", "mixture"] = (
        "gaussian"
    )
    df: float = 5.0
    mixture_weights: tuple[float, ...] = (0.9, 0.1)
    mixture_scales: tuple[float, ...] = (1.0, 3.0)


class RegimesConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    n_states: int = 3
    state_names: tuple[str, ...] = ("bear", "sideways", "bull")
    persistence: float = 0.97
    drifts: tuple[float, ...] = (-0.15, 0.0, 0.12)
    volatilities: tuple[float, ...] = (0.35, 0.15, 0.22)


class LiquidityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_spread_bps: float = 5.0
    min_spread_bps: float = 1.0
    depth_levels: int = 5
    base_depth: float = 10.0
    volume_scale: float = 100.0
    slippage_impact: float = 0.1


class CorrelationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    rho: float = 0.3


class EventsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    flash_crash_prob: float = 0.002
    news_shock_prob: float = 0.01
    gap_open_prob: float = 0.005
    liquidity_collapse_prob: float = 0.003
    outage_prob: float = 0.001
    vol_spike_prob: float = 0.008
    momentum_burst_prob: float = 0.01


class ValidationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    significance: float = 0.05
    acf_lags: int = 20


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_points: int = 800
    output_dir: str = "data/reports/simulation"


class SimulationSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    random_seed: int = 42
    n_steps: int = 1000
    n_assets: int = 1
    dt: float = 0.004
    initial_price: float = 100.0
    asset_class: Literal["stock", "crypto", "forex", "commodity", "index"] = "crypto"
    default_model: str = "gbm"
    market: MarketConfig = Field(default_factory=MarketConfig)
    dynamics: DynamicsConfig = Field(default_factory=DynamicsConfig)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    regimes: RegimesConfig = Field(default_factory=RegimesConfig)
    liquidity: LiquidityConfig = Field(default_factory=LiquidityConfig)
    correlation: CorrelationConfig = Field(default_factory=CorrelationConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | Any) -> SimulationSettings:
        from iqrp.app.core.exceptions import ConfigurationError

        if not isinstance(data, dict):
            if OmegaConf.is_config(data):
                container = OmegaConf.to_container(data, resolve=True)
            else:
                container = data
            if not isinstance(container, dict):
                raise ConfigurationError(
                    "Simulation config mapping invalid",
                    code="SIM_CONFIG_INVALID",
                )
            data = container
        return cls.model_validate(data)

    @classmethod
    def from_hydra(
        cls,
        config_path: Path | None = None,
        overrides: list[str] | None = None,
    ) -> SimulationSettings:
        path = config_path or _default_config_path()
        cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        container = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(container, dict):
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                "Simulation config root must be a mapping",
                code="SIM_CONFIG_INVALID",
            )
        return cls.from_mapping(container)

    @classmethod
    def default(cls) -> SimulationSettings:
        path = _default_config_path()
        if path.exists():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "simulation" / "default.yaml"

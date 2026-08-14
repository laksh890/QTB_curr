"""Hydra-backed configuration for the Institutional Statistical Forecasting Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class OrderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    p: int | None = None
    d: int | None = None
    q: int | None = None
    P: int | None = None
    D: int | None = None
    Q: int | None = None
    seasonal_period: int = 12
    max_p: int = 5
    max_d: int = 2
    max_q: int = 5
    max_P: int = 2
    max_D: int = 1
    max_Q: int = 2
    max_var_lags: int = 5


class IdentificationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    auto: bool = True
    criterion: Literal["aic", "aicc", "bic", "hqic"] = "aic"
    adf_alpha: float = 0.05
    kpss_alpha: float = 0.05
    seasonal_detect: bool = True
    trend: Literal["none", "c", "ct", "auto"] = "auto"


class OnlineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: Literal["expanding", "sliding", "rolling"] = "expanding"
    window: int = 252
    warm_start: bool = True
    retrain_every: int = 0


class RegimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    column: str = "regime"
    condition_forecasts: bool = True
    use_probabilities: bool = False
    probability_prefix: str = "regime_p"


class ForecastConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_horizon: int = 5
    strategy: Literal["recursive", "direct"] = "recursive"
    interval_level: float = 0.95
    parallel_selection: bool = True
    n_jobs: int = 4


class ColumnsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: str = "open_time"
    target: str = "target"
    feature_columns: tuple[str, ...] | None = None
    endogenous: tuple[str, ...] | None = None
    exogenous: tuple[str, ...] | None = None


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_points: int = 500


class StatisticalSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    order: OrderConfig = Field(default_factory=OrderConfig)
    identification: IdentificationConfig = Field(default_factory=IdentificationConfig)
    online: OnlineConfig = Field(default_factory=OnlineConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    discovery_modules: tuple[str, ...] = (
        "iqrp.app.forecasting.statistical.ar.ar",
        "iqrp.app.forecasting.statistical.ma.ma",
        "iqrp.app.forecasting.statistical.arma.arma",
        "iqrp.app.forecasting.statistical.arima.arima",
        "iqrp.app.forecasting.statistical.sarima.sarima",
        "iqrp.app.forecasting.statistical.var.var",
        "iqrp.app.forecasting.statistical.varmax.varmax",
        "iqrp.app.forecasting.statistical.vecm.vecm",
        "iqrp.app.forecasting.statistical.exponential.simple",
        "iqrp.app.forecasting.statistical.exponential.holt",
        "iqrp.app.forecasting.statistical.exponential.holt_winters",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> StatisticalSettings:
        try:
            if hasattr(data, "items") and not isinstance(data, dict):
                data = OmegaConf.to_container(data, resolve=True)
            return cls.model_validate(dict(data or {}))
        except Exception as exc:
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid statistical forecasting settings: {exc}",
                code="STAT_CONFIG_INVALID",
            ) from exc

    @classmethod
    def from_hydra(
        cls,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
    ) -> StatisticalSettings:
        path = Path(config_path) if config_path else _default_config_path()
        cfg: Any = OmegaConf.create({})
        if path.is_file():
            cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cls.from_mapping(OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    def default(cls) -> StatisticalSettings:
        path = _default_config_path()
        if path.is_file():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "forecasting"
        / "statistical"
        / "default.yaml"
    )

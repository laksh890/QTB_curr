"""Hydra-backed configuration for the Institutional Volatility Forecasting Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class OrderConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    p: int = 1
    q: int = 1
    o: int = 1  # asymmetry order (GJR/APARCH)
    max_p: int = 2
    max_q: int = 2
    ewma_lambda: float = 0.94
    rolling_window: int = 21
    annualization: float = 252.0


class DistributionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: Literal["gaussian", "student_t", "skew_t", "ged", "laplace", "custom"] = "gaussian"
    df: float = 8.0
    skew: float = 0.0
    ged_nu: float = 1.5


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["L-BFGS-B", "SLSQP", "Nelder-Mead", "robust"] = "L-BFGS-B"
    maxiter: int = 500
    tol: float = 1e-8
    n_restarts: int = 2
    robust: bool = True


class OnlineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: Literal["expanding", "sliding", "rolling"] = "expanding"
    window: int = 252
    warm_start: bool = True
    adaptive_rate: float = 0.05


class RegimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    column: str = "regime"
    condition: bool = True
    ensemble_weight: bool = False


class ForecastConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_horizon: int = 5
    interval_level: float = 0.95
    scenario_paths: int = 0


class ColumnsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    timestamp: str = "open_time"
    target: str = "returns"
    feature_columns: tuple[str, ...] | None = None
    assets: tuple[str, ...] | None = None


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    max_points: int = 500


class VolatilitySettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    order: OrderConfig = Field(default_factory=OrderConfig)
    distribution: DistributionConfig = Field(default_factory=DistributionConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    online: OnlineConfig = Field(default_factory=OnlineConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    columns: ColumnsConfig = Field(default_factory=ColumnsConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    selection_criterion: Literal["aic", "bic", "loglik", "qlike"] = "aic"
    discovery_modules: tuple[str, ...] = (
        "iqrp.app.forecasting.volatility.ewma.ewma",
        "iqrp.app.forecasting.volatility.arch.arch",
        "iqrp.app.forecasting.volatility.garch.garch",
        "iqrp.app.forecasting.volatility.egarch.egarch",
        "iqrp.app.forecasting.volatility.gjr.gjr_garch",
        "iqrp.app.forecasting.volatility.figarch.figarch",
        "iqrp.app.forecasting.volatility.aparch.aparch",
        "iqrp.app.forecasting.volatility.cgarch.component_garch",
        "iqrp.app.forecasting.volatility.historical.historical",
        "iqrp.app.forecasting.volatility.multivariate.dcc_garch",
        "iqrp.app.forecasting.volatility.multivariate.bekk",
    )

    @classmethod
    def from_mapping(cls, data: Any) -> VolatilitySettings:
        try:
            if hasattr(data, "items") and not isinstance(data, dict):
                data = OmegaConf.to_container(data, resolve=True)
            return cls.model_validate(dict(data or {}))
        except Exception as exc:
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid volatility settings: {exc}",
                code="VOL_CONFIG_INVALID",
            ) from exc

    @classmethod
    def from_hydra(
        cls,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
    ) -> VolatilitySettings:
        path = Path(config_path) if config_path else _default_config_path()
        cfg: Any = OmegaConf.create({})
        if path.is_file():
            cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cls.from_mapping(OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    def default(cls) -> VolatilitySettings:
        path = _default_config_path()
        if path.is_file():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "forecasting"
        / "volatility"
        / "default.yaml"
    )

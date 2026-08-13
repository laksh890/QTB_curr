"""Hydra-backed settings for Institutional Time-Series Analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class DecompositionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["classical", "stl", "mstl"] = "stl"
    period: int = 24
    model: Literal["additive", "multiplicative"] = "additive"
    robust: bool = False


class StationarityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    alpha: float = 0.05
    max_lag: int | None = None


class ChangePointConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["cusum", "binseg", "pelt", "bayesian", "online"] = "pelt"
    kind: Literal["mean", "variance", "trend"] = "mean"
    min_size: int = 10
    penalty: float = 3.0


class SpectralConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    nperseg: int = 64
    detrend: bool = True


class WaveletConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    wavelet: str = "haar"
    level: int | None = None
    threshold: float = 0.1


class AnomalyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["statistical", "robust", "isolation_forest", "matrix_profile"] = "robust"
    contamination: float = 0.05
    z_threshold: float = 3.0


class MotifConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    window: int = 32
    top_k: int = 3


class TransformConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["log_return", "simple_return", "diff", "zscore", "robust", "rank"] = "log_return"
    window: int = 64
    temporal_mode: Literal["rolling", "expanding", "training_only"] = "rolling"


class MultipleTestingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["bonferroni", "holm", "fdr_bh", "none"] = "fdr_bh"
    alpha: float = 0.05


class FeatureConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    window: int = 64
    include_entropy: bool = True
    include_hurst: bool = True
    include_spectral: bool = True


class TimeSeriesSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    decomposition: DecompositionConfig = Field(default_factory=DecompositionConfig)
    stationarity: StationarityConfig = Field(default_factory=StationarityConfig)
    change_points: ChangePointConfig = Field(default_factory=ChangePointConfig)
    spectral: SpectralConfig = Field(default_factory=SpectralConfig)
    wavelet: WaveletConfig = Field(default_factory=WaveletConfig)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    motif: MotifConfig = Field(default_factory=MotifConfig)
    transform: TransformConfig = Field(default_factory=TransformConfig)
    multiple_testing: MultipleTestingConfig = Field(default_factory=MultipleTestingConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    seed: int = 42
    data_version: str = "1.0.0"

    @classmethod
    def from_mapping(cls, data: Any) -> TimeSeriesSettings:
        try:
            if hasattr(data, "items") and not isinstance(data, dict):
                data = OmegaConf.to_container(data, resolve=True)
            return cls.model_validate(dict(data or {}))
        except Exception as exc:  # noqa: BLE001
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid timeseries settings: {exc}",
                code="TS_CONFIG_INVALID",
            ) from exc

    @classmethod
    def from_hydra(
        cls,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
    ) -> TimeSeriesSettings:
        path = Path(config_path) if config_path else _default_config_path()
        cfg: Any = OmegaConf.create({})
        if path.is_file():
            cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cls.from_mapping(OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    def default(cls) -> TimeSeriesSettings:
        path = _default_config_path()
        if path.is_file():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "timeseries" / "default.yaml"

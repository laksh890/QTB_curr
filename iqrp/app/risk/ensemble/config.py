"""Hydra-backed settings for the Risk Intelligence Ensemble."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class NormalizationRef(BaseModel):
    model_config = ConfigDict(frozen=True)

    zero: float = 0.0
    one: float = 1.0
    invert: bool = False


class DrawdownThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    caution: float = 0.05
    reduced_risk: float = 0.10
    capital_preservation: float = 0.15
    trading_halt: float = 0.20


class StateThresholds(BaseModel):
    model_config = ConfigDict(frozen=True)

    caution: float = 0.35
    reduced_risk: float = 0.55
    capital_preservation: float = 0.72
    trading_halt: float = 0.88


class HysteresisConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    escalation_confirmations: int = 1
    recovery_confirmations: int = 3
    dimension_confirmation_threshold: float = 0.75


class StateCap(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_exposure: float = 1.0
    recommended_leverage: float = 1.0
    position_reduction: float = 0.0


class EnsembleLimitConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_position: float = 0.10
    max_gross_exposure: float = 1.5
    max_net_exposure: float = 1.0
    max_concentration: float = 0.25
    max_leverage: float = 2.0
    max_daily_loss: float = 0.03


class EnsembleLeverageConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_leverage: float = 1.0
    max_leverage: float = 2.0
    min_leverage: float = 0.0
    confidence_cap: float = 1.25
    target_volatility: float = 0.10


class ConfidenceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base: float = 0.70
    disagreement_penalty: float = 0.50
    sample_size_floor: int = 30
    sample_size_full: int = 252
    missing_metric_penalty: float = 0.15
    min_confidence: float = 0.05
    max_confidence: float = 0.99


class DisagreementConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    relative_epsilon: float = 1.0e-8
    high_disagreement: float = 0.35
    pairs: list[list[str]] = Field(
        default_factory=lambda: [
            ["var_historical", "var_monte_carlo"],
            ["garch_vol", "realized_vol"],
            ["es_parametric", "es_historical"],
            ["corr_normal", "corr_stress"],
            ["liquidity_model", "liquidity_observed"],
        ]
    )


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    var_alpha: float = 0.05
    es_alpha: float = 0.05
    tolerance_band: float = 0.02


class BudgetConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    target_risk: float = 0.10
    max_budget_scale: float = 1.5
    min_budget_scale: float = 0.25


class EnsembleSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed: int = 42
    data_version: str = "1.0.0"
    model_version: str = "1.0.0"
    ensemble_version: str = "1.0.0"

    critical_metric_keys: list[str] = Field(
        default_factory=lambda: ["volatility", "var", "cvar", "drawdown"]
    )
    missing_metrics_fallback_state: Literal[
        "NORMAL",
        "CAUTION",
        "REDUCED_RISK",
        "CAPITAL_PRESERVATION",
        "TRADING_HALT",
    ] = "CAPITAL_PRESERVATION"
    missing_metrics_fallback_action: Literal[
        "APPROVE",
        "APPROVE_REDUCED",
        "REJECT",
        "HALT",
    ] = "REJECT"
    hard_halt_on_single: bool = False
    min_dimensions_for_halt: int = 2

    weighting_scheme: Literal[
        "static",
        "risk_budget",
        "regime",
        "dynamic",
        "calibration",
        "stress",
        "user_defined",
    ] = "static"
    static_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "market": 0.18,
            "tail": 0.20,
            "liquidity": 0.12,
            "concentration": 0.10,
            "correlation": 0.10,
            "drawdown": 0.18,
            "model": 0.07,
            "operational": 0.05,
        }
    )
    user_defined_weights: dict[str, float] = Field(default_factory=dict)

    normalization: dict[str, NormalizationRef] = Field(
        default_factory=lambda: {
            "volatility": NormalizationRef(zero=0.0, one=0.50),
            "var": NormalizationRef(zero=0.0, one=0.10),
            "cvar": NormalizationRef(zero=0.0, one=0.15),
            "expected_shortfall": NormalizationRef(zero=0.0, one=0.15),
            "drawdown": NormalizationRef(zero=0.0, one=0.20),
            "liquidity_score": NormalizationRef(zero=1.0, one=0.0, invert=True),
            "concentration": NormalizationRef(zero=0.0, one=0.50),
            "correlation": NormalizationRef(zero=0.0, one=1.0),
            "model_risk": NormalizationRef(zero=0.0, one=1.0),
            "operational": NormalizationRef(zero=0.0, one=1.0),
            "gap_risk": NormalizationRef(zero=0.0, one=0.10),
        }
    )

    drawdown: DrawdownThresholds = Field(default_factory=DrawdownThresholds)
    state_thresholds: StateThresholds = Field(default_factory=StateThresholds)
    recovery_thresholds: StateThresholds = Field(
        default_factory=lambda: StateThresholds(
            caution=0.28,
            reduced_risk=0.45,
            capital_preservation=0.62,
            trading_halt=0.78,
        )
    )
    hysteresis: HysteresisConfig = Field(default_factory=HysteresisConfig)
    state_caps: dict[str, StateCap] = Field(
        default_factory=lambda: {
            "NORMAL": StateCap(max_exposure=1.0, recommended_leverage=1.0, position_reduction=0.0),
            "CAUTION": StateCap(max_exposure=0.75, recommended_leverage=0.75, position_reduction=0.25),
            "REDUCED_RISK": StateCap(max_exposure=0.50, recommended_leverage=0.50, position_reduction=0.50),
            "CAPITAL_PRESERVATION": StateCap(
                max_exposure=0.25, recommended_leverage=0.25, position_reduction=0.75
            ),
            "TRADING_HALT": StateCap(max_exposure=0.0, recommended_leverage=0.0, position_reduction=1.0),
        }
    )
    limits: EnsembleLimitConfig = Field(default_factory=EnsembleLimitConfig)
    leverage: EnsembleLeverageConfig = Field(default_factory=EnsembleLeverageConfig)
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    disagreement: DisagreementConfig = Field(default_factory=DisagreementConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    regime_scales: dict[str, float] = Field(
        default_factory=lambda: {
            "normal": 1.0,
            "low_vol": 0.9,
            "high_vol": 1.2,
            "stress": 1.35,
            "crisis": 1.5,
            "transition": 1.1,
        }
    )
    budget: BudgetConfig = Field(default_factory=BudgetConfig)

    @classmethod
    def from_mapping(cls, data: Any) -> EnsembleSettings:
        try:
            if hasattr(data, "items") and not isinstance(data, dict):
                data = OmegaConf.to_container(data, resolve=True)
            raw = dict(data or {})
            if "normalization" in raw and isinstance(raw["normalization"], dict):
                raw["normalization"] = {
                    str(k): (v if isinstance(v, NormalizationRef) else NormalizationRef.model_validate(v))
                    for k, v in raw["normalization"].items()
                }
            if "state_caps" in raw and isinstance(raw["state_caps"], dict):
                raw["state_caps"] = {
                    str(k): (v if isinstance(v, StateCap) else StateCap.model_validate(v))
                    for k, v in raw["state_caps"].items()
                }
            return cls.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            from iqrp.app.core.exceptions import ConfigurationError

            raise ConfigurationError(
                f"Invalid ensemble settings: {exc}", code="ENSEMBLE_CONFIG_INVALID"
            ) from exc

    @classmethod
    def from_hydra(
        cls,
        config_path: str | Path | None = None,
        overrides: list[str] | None = None,
    ) -> EnsembleSettings:
        path = Path(config_path) if config_path else _default_config_path()
        cfg: Any = OmegaConf.create({})
        if path.is_file():
            cfg = OmegaConf.load(path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        return cls.from_mapping(OmegaConf.to_container(cfg, resolve=True))

    @classmethod
    def default(cls) -> EnsembleSettings:
        path = _default_config_path()
        if path.is_file():
            return cls.from_hydra(path)
        return cls()


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "risk" / "ensemble" / "default.yaml"

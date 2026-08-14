"""Thin registry describing how EXISTING models expose signals to Alpha Research."""

from __future__ import annotations

from typing import Any

from iqrp.app.backtesting.alpha_research.adapters.types import (
    ModelAdapterSpec,
    OutputMappingKind,
    SignalMappingConfig,
)

_REGISTRY: dict[str, ModelAdapterSpec] = {}


def register_adapter(spec: ModelAdapterSpec, *, overwrite: bool = False) -> ModelAdapterSpec:
    if spec.adapter_id in _REGISTRY and not overwrite:
        raise ValueError(f"adapter already registered: {spec.adapter_id}")
    _REGISTRY[spec.adapter_id] = spec
    return spec


def get_adapter(adapter_id: str) -> ModelAdapterSpec:
    if adapter_id not in _REGISTRY:
        raise KeyError(adapter_id)
    return _REGISTRY[adapter_id]


def list_adapters() -> list[ModelAdapterSpec]:
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]


def clear_adapters() -> None:
    _REGISTRY.clear()


def register_default_adapters(*, overwrite: bool = True) -> list[ModelAdapterSpec]:
    """Register thin adapters for models known to exist in-repo (no model code)."""
    specs = [
        ModelAdapterSpec(
            adapter_id="garch_volatility_v1_1h",
            model_id="garch",
            model_family="volatility",
            model_version="1.0.0",
            input_schema=("returns",),
            output_type="forecast",
            timeframe="1h",
            horizon=1,
            factory_path="iqrp.app.forecasting.volatility.create_volatility_model",
            signal_mapping=SignalMappingConfig(
                kind=OutputMappingKind.VOLATILITY_EXPANSION,
                vol_z_threshold=0.5,
                vol_lookback=20,
            ),
            notes="Existing GARCH → vol-expansion/contraction tilt (wiring only)",
        ),
        ModelAdapterSpec(
            adapter_id="arima_return_v1_1h",
            model_id="arima",
            model_family="statistical",
            model_version="1.0.0",
            input_schema=("target",),
            output_type="forecast",
            timeframe="1h",
            horizon=1,
            factory_path="iqrp.app.forecasting.statistical.create_statistical_model",
            signal_mapping=SignalMappingConfig(
                kind=OutputMappingKind.RETURN_THRESHOLD,
                long_threshold=0.0,
                short_threshold=0.0,
            ),
            notes="Existing ARIMA level/return forecast → threshold signal",
        ),
        ModelAdapterSpec(
            adapter_id="xgb_return_v1_1h",
            model_id="xgboost",
            model_family="tree_ml",
            model_version="1.0.0",
            input_schema=("features", "target"),
            output_type="forecast",
            timeframe="1h",
            horizon=1,
            factory_path="iqrp.app.forecasting.tree_models.create_tree_model",
            signal_mapping=SignalMappingConfig(
                kind=OutputMappingKind.RETURN_THRESHOLD,
                long_threshold=0.0,
                short_threshold=0.0,
            ),
            notes="Existing XGBoost forecast → threshold signal",
        ),
        ModelAdapterSpec(
            adapter_id="lstm_return_v1_1h",
            model_id="lstm",
            model_family="neural",
            model_version="1.0.0",
            input_schema=("features", "target"),
            output_type="forecast",
            timeframe="1h",
            horizon=1,
            factory_path="iqrp.app.forecasting.neural.create_neural_model",
            availability="partial",
            signal_mapping=SignalMappingConfig(kind=OutputMappingKind.RETURN_THRESHOLD),
            notes="Existing LSTM; may be UNAVAILABLE if training constraints fail in smoke",
        ),
        ModelAdapterSpec(
            adapter_id="transformer_return_v1_1h",
            model_id="tide",
            model_family="transformer",
            model_version="1.0.0",
            input_schema=("features", "target"),
            output_type="forecast",
            timeframe="1h",
            horizon=1,
            factory_path="iqrp.app.forecasting.transformers.create_transformer_model",
            availability="partial",
            signal_mapping=SignalMappingConfig(kind=OutputMappingKind.RETURN_THRESHOLD),
            notes="Existing transformer (TiDE); may be UNAVAILABLE in smoke",
        ),
        ModelAdapterSpec(
            adapter_id="hmm_regime_v1_1h",
            model_id="hmm",
            model_family="regime",
            model_version="1.0.0",
            input_schema=("features",),
            output_type="regime",
            timeframe="1h",
            horizon=1,
            factory_path="iqrp.app.regimes.hmm.model",
            signal_mapping=SignalMappingConfig(
                kind=OutputMappingKind.REGIME_LABEL_MAP,
                regime_map={"0": 0.0, "1": 1.0, "2": -1.0},
                flat_on_unknown_regime=True,
            ),
            notes="Existing HMM; explicit module load required (default registry is mock-only)",
        ),
        ModelAdapterSpec(
            adapter_id="mock_regime_v1_1h",
            model_id="mock_regime",
            model_family="regime",
            model_version="1.0.0",
            input_schema=("features",),
            output_type="regime",
            timeframe="1h",
            horizon=1,
            factory_path="iqrp.app.regimes.models.mock",
            signal_mapping=SignalMappingConfig(
                kind=OutputMappingKind.REGIME_LABEL_MAP,
                regime_map={"0": 0.0, "1": 1.0, "2": -1.0},
            ),
            notes="Default-loadable mock regime for wiring validation",
        ),
    ]
    for s in specs:
        register_adapter(s, overwrite=overwrite)
    return list_adapters()


def adapters_to_jsonable() -> list[dict[str, Any]]:
    return [a.to_dict() for a in list_adapters()]


__all__ = [
    "adapters_to_jsonable",
    "clear_adapters",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    "register_default_adapters",
]

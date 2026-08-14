"""Thin MODEL → ALPHA adapters (integration only; no model reimplementation)."""

from iqrp.app.backtesting.alpha_research.adapters.forecast_adapter import (
    forecast_to_signal_series,
    map_values_to_signal,
    metadata_bundle,
)
from iqrp.app.backtesting.alpha_research.adapters.model_registry import (
    adapters_to_jsonable,
    get_adapter,
    list_adapters,
    register_adapter,
    register_default_adapters,
)
from iqrp.app.backtesting.alpha_research.adapters.pipeline import (
    align_model_signal_mtf,
    run_adapter,
)
from iqrp.app.backtesting.alpha_research.adapters.regime_adapter import (
    regime_probabilities_to_confidence,
    regime_result_to_signal_series,
    regime_states_to_signal,
)
from iqrp.app.backtesting.alpha_research.adapters.signal_registration import (
    attach_precomputed_signal,
    clear_model_signal_cache,
    register_model_adapter_signals,
)
from iqrp.app.backtesting.alpha_research.adapters.types import (
    ModelAdapterSpec,
    OutputMappingKind,
    SignalMappingConfig,
)

__all__ = [
    "ModelAdapterSpec",
    "OutputMappingKind",
    "SignalMappingConfig",
    "adapters_to_jsonable",
    "align_model_signal_mtf",
    "attach_precomputed_signal",
    "clear_model_signal_cache",
    "forecast_to_signal_series",
    "get_adapter",
    "list_adapters",
    "map_values_to_signal",
    "metadata_bundle",
    "register_adapter",
    "register_default_adapters",
    "register_model_adapter_signals",
    "regime_probabilities_to_confidence",
    "regime_result_to_signal_series",
    "regime_states_to_signal",
    "run_adapter",
]

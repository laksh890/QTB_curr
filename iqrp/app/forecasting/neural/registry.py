"""Discovery helpers for neural forecasting models."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

from iqrp.app.forecasting.base.registry import get_registry
from iqrp.app.forecasting.neural.config import NeuralSettings

_NEURAL_NAMES = frozenset(
    {
        "mlp",
        "lstm",
        "stacked_lstm",
        "bidirectional_lstm",
        "gru",
        "stacked_gru",
        "tcn",
        "nbeats",
        "nhits",
        "deepar",
        "seq2seq",
        "attention_seq2seq",
    }
)


def ensure_neural_models_loaded(modules: Iterable[str] | None = None) -> list[str]:
    settings = NeuralSettings.default()
    loaded: list[str] = []
    for mod in modules or settings.discovery_modules:
        try:
            importlib.import_module(mod)
            loaded.append(mod)
        except Exception:  # noqa: BLE001  # pragma: no cover
            continue
    return loaded


def list_neural_models() -> list[str]:
    ensure_neural_models_loaded()
    return [n for n in get_registry().list_names() if n in _NEURAL_NAMES]


def create_neural_model(name: str, **kwargs: Any) -> Any:
    ensure_neural_models_loaded()
    return get_registry().create(name, **kwargs)

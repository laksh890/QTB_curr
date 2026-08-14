"""Discovery helpers for transformer forecasting models."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

from iqrp.app.forecasting.base.registry import get_registry
from iqrp.app.forecasting.transformers.config import TransformerSettings

_TRANSFORMER_NAMES = frozenset(
    {
        "tft",
        "informer",
        "autoformer",
        "fedformer",
        "patchtst",
        "crossformer",
        "timesnet",
        "itransformer",
        "timemixer",
        "tide",
        "moe_transformer",
    }
)


def ensure_transformer_models_loaded(modules: Iterable[str] | None = None) -> list[str]:
    settings = TransformerSettings.default()
    loaded: list[str] = []
    for mod in modules or settings.discovery_modules:
        try:
            importlib.import_module(mod)
            loaded.append(mod)
        except Exception:  # pragma: no cover
            continue
    return loaded


def list_transformer_models() -> list[str]:
    ensure_transformer_models_loaded()
    return [n for n in get_registry().list_names() if n in _TRANSFORMER_NAMES]


def create_transformer_model(name: str, **kwargs: Any) -> Any:
    ensure_transformer_models_loaded()
    return get_registry().create(name, **kwargs)

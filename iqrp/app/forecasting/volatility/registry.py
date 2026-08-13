"""Discovery helpers for volatility forecasting models."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

from iqrp.app.forecasting.base.registry import get_registry
from iqrp.app.forecasting.volatility.config import VolatilitySettings

_VOL_NAMES = frozenset(
    {
        "historical_volatility",
        "rolling_volatility",
        "ewma",
        "arch",
        "garch",
        "egarch",
        "gjr_garch",
        "figarch",
        "aparch",
        "component_garch",
        "dcc_garch",
        "bekk",
    }
)


def ensure_volatility_models_loaded(modules: Iterable[str] | None = None) -> list[str]:
    settings = VolatilitySettings.default()
    loaded: list[str] = []
    for mod in modules or settings.discovery_modules:
        try:
            importlib.import_module(mod)
            loaded.append(mod)
        except Exception:  # noqa: BLE001
            continue
    return loaded


def list_volatility_models() -> list[str]:
    ensure_volatility_models_loaded()
    return [n for n in get_registry().list_names() if n in _VOL_NAMES]


def create_volatility_model(name: str, **kwargs: Any) -> Any:
    ensure_volatility_models_loaded()
    return get_registry().create(name, **kwargs)

"""Discovery helpers for statistical forecasting models."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

from iqrp.app.forecasting.base.registry import get_registry
from iqrp.app.forecasting.statistical.config import StatisticalSettings

_STAT_NAMES = frozenset(
    {
        "ar",
        "ma",
        "arma",
        "arima",
        "sarima",
        "var",
        "varmax",
        "vecm",
        "ses",
        "holt",
        "holt_winters",
    }
)


def ensure_statistical_models_loaded(modules: Iterable[str] | None = None) -> list[str]:
    settings = StatisticalSettings.default()
    loaded: list[str] = []
    for mod in modules or settings.discovery_modules:
        try:
            importlib.import_module(mod)
            loaded.append(mod)
        except Exception:
            continue
    return loaded


def list_statistical_models() -> list[str]:
    ensure_statistical_models_loaded()
    return [n for n in get_registry().list_names() if n in _STAT_NAMES]


def create_statistical_model(name: str, **kwargs: Any) -> Any:
    ensure_statistical_models_loaded()
    return get_registry().create(name, **kwargs)

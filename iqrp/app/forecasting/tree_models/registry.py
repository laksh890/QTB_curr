"""Discovery helpers for tree-based forecasting models."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from typing import Any

from iqrp.app.forecasting.base.registry import get_registry
from iqrp.app.forecasting.tree_models.config import TreeSettings

_TREE_NAMES = frozenset(
    {
        "xgboost",
        "lightgbm",
        "catboost",
        "hist_gradient_boosting",
        "random_forest",
        "extra_trees",
    }
)


def ensure_tree_models_loaded(modules: Iterable[str] | None = None) -> list[str]:
    settings = TreeSettings.default()
    loaded: list[str] = []
    for mod in modules or settings.discovery_modules:
        try:
            importlib.import_module(mod)
            loaded.append(mod)
        except Exception:
            continue
    return loaded


def list_tree_models() -> list[str]:
    ensure_tree_models_loaded()
    return [n for n in get_registry().list_names() if n in _TREE_NAMES]


def create_tree_model(name: str, **kwargs: Any) -> Any:
    ensure_tree_models_loaded()
    return get_registry().create(name, **kwargs)

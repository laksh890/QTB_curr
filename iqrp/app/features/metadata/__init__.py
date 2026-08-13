"""Feature metadata catalog helpers."""

from __future__ import annotations

from typing import Any

from iqrp.app.features.base.registry import ensure_features_loaded, get_registry


def list_metadata(*, category: str | None = None) -> list[dict[str, Any]]:
    ensure_features_loaded()
    registry = get_registry()
    return [registry.describe(n).to_dict() for n in registry.list_names(category=category)]


def get_metadata(name: str) -> dict[str, Any]:
    ensure_features_loaded()
    return get_registry().describe(name).to_dict()

"""Dynamic discovery of every forecasting engine — no hard-coded model lists."""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any

from iqrp.app.forecasting.base.registry import get_registry
from iqrp.app.forecasting.intelligence.config import DiscoveryConfig, IntelligenceSettings


@dataclass(slots=True)
class DiscoveredModel:
    name: str
    family: str
    version: str
    supports_proba: bool
    supports_intervals: bool
    supports_online: bool
    module: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "version": self.version,
            "supports_proba": self.supports_proba,
            "supports_intervals": self.supports_intervals,
            "supports_online": self.supports_online,
            "module": self.module,
            "meta": dict(self.meta),
        }


def discover_engine_modules(root_package: str = "iqrp.app.forecasting") -> list[str]:
    """Walk forecasting package tree and return registry / discovery modules."""
    found: list[str] = []
    try:
        root = importlib.import_module(root_package)
    except Exception:  # pragma: no cover
        return found
    paths = getattr(root, "__path__", None)
    if paths is None:
        return found
    for info in pkgutil.walk_packages(paths, root_package + "."):
        name = info.name
        leaf = name.rsplit(".", 1)[-1]
        if (
            leaf in {"registry", "mock"}
            or name.endswith(".models.mock")
            or (leaf == "__init__" and ".models" in name)
        ):
            found.append(name)
    return sorted(set(found))


def load_discovered_engines(modules: list[str] | None = None) -> list[str]:
    """Import discovery modules and invoke any ``ensure_*_loaded`` hooks."""
    loaded: list[str] = []
    for mod_name in modules or discover_engine_modules():
        try:
            mod = importlib.import_module(mod_name)
        except Exception:  # pragma: no cover
            continue
        loaded.append(mod_name)
        for attr in dir(mod):
            if attr.startswith("ensure_") and attr.endswith("_loaded"):
                fn = getattr(mod, attr)
                if callable(fn):
                    try:
                        fn()
                    except Exception:  # pragma: no cover
                        continue
    # always ensure framework baseline models
    try:
        from iqrp.app.forecasting.base.registry import ensure_forecast_models_loaded

        ensure_forecast_models_loaded()
    except Exception:  # pragma: no cover
        pass
    return loaded


def list_discovered_models(
    settings: DiscoveryConfig | IntelligenceSettings | None = None,
) -> list[DiscoveredModel]:
    cfg = _as_discovery(settings)
    load_discovered_engines()
    reg = get_registry()
    out: list[DiscoveredModel] = []
    for name in reg.list_names():
        if name in cfg.exclude_names:
            continue
        meta = reg.describe(name)
        family = str(meta.algorithm_family)
        if cfg.include_families is not None and family not in cfg.include_families:
            continue
        if family in cfg.exclude_families:
            continue
        out.append(
            DiscoveredModel(
                name=name,
                family=family,
                version=str(meta.version),
                supports_proba=bool(meta.supports_proba),
                supports_intervals=bool(meta.supports_intervals),
                supports_online=bool(meta.supports_online),
                meta=meta.to_dict(),
            )
        )
    if cfg.max_candidates is not None:
        out = out[: max(int(cfg.max_candidates), 0)]
    return out


def create_model(name: str, **kwargs: Any) -> Any:
    load_discovered_engines()
    return get_registry().create(name, **kwargs)


def _as_discovery(settings: DiscoveryConfig | IntelligenceSettings | None) -> DiscoveryConfig:
    if settings is None:
        return DiscoveryConfig()
    if isinstance(settings, IntelligenceSettings):
        return settings.discovery
    return settings

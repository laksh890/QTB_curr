"""Explicit strategy registry — never silently pick an arbitrary strategy."""

from __future__ import annotations

from iqrp.app.backtesting.strategy.base import Strategy


class StrategyRegistryError(KeyError):
    """Raised when a strategy id/version is missing or ambiguous."""


class StrategyRegistry:
    """Register and resolve strategies by ``(strategy_id, version)``.

    Selection always requires an explicit ``strategy_id``. Version defaults
    only when exactly one registered version exists for that id.
    """

    _registry: dict[tuple[str, str], type[Strategy]] = {}

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()

    @classmethod
    def register(cls, strategy_cls: type[Strategy], *, overwrite: bool = False) -> type[Strategy]:
        sid = str(getattr(strategy_cls, "strategy_id", "") or "").strip()
        ver = str(getattr(strategy_cls, "strategy_version", "") or "").strip()
        if not sid:
            raise ValueError(
                f"cannot register {strategy_cls!r}: strategy_id must be a non-empty string"
            )
        if not ver:
            raise ValueError(
                f"cannot register {strategy_cls!r}: strategy_version must be a non-empty string"
            )
        key = (sid, ver)
        if key in cls._registry and not overwrite:
            raise StrategyRegistryError(f"strategy already registered: id={sid!r} version={ver!r}")
        cls._registry[key] = strategy_cls
        return strategy_cls

    @classmethod
    def registered(cls) -> list[tuple[str, str]]:
        return sorted(cls._registry.keys())

    @classmethod
    def get(cls, strategy_id: str, version: str | None = None) -> type[Strategy]:
        sid = str(strategy_id or "").strip()
        if not sid:
            raise StrategyRegistryError(
                "strategy_id is required; refusing silent / arbitrary selection"
            )
        if version is not None and str(version).strip():
            key = (sid, str(version).strip())
            if key not in cls._registry:
                raise StrategyRegistryError(
                    f"unknown strategy id={sid!r} version={version!r}; "
                    f"registered={cls.registered()}"
                )
            return cls._registry[key]

        matches = [(k, v) for k, v in cls._registry.items() if k[0] == sid]
        if not matches:
            raise StrategyRegistryError(
                f"unknown strategy id={sid!r}; registered={cls.registered()}"
            )
        if len(matches) > 1:
            versions = sorted(k[1] for k, _ in matches)
            raise StrategyRegistryError(
                f"strategy id={sid!r} has multiple versions {versions}; "
                "pass strategy_version explicitly"
            )
        return matches[0][1]

    @classmethod
    def create(
        cls,
        strategy_id: str,
        version: str | None = None,
        **kwargs,
    ) -> Strategy:
        strategy_cls = cls.get(strategy_id, version)
        return strategy_cls(**kwargs)


__all__ = ["StrategyRegistry", "StrategyRegistryError"]

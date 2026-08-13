"""Shared protocol interfaces used across domain modules."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Configurable(Protocol):
    """Component that can be configured from a plain mapping."""

    def configure(self, settings: dict[str, Any]) -> None:
        """Apply configuration. Must be idempotent for identical settings."""
        ...


@runtime_checkable
class AsyncLifecycle(Protocol):
    """Component with asynchronous start/stop lifecycle hooks."""

    async def start(self) -> None:
        """Acquire resources and enter the running state."""
        ...

    async def stop(self) -> None:
        """Release resources and leave the running state."""
        ...


@runtime_checkable
class HealthCheckable(Protocol):
    """Component that can report operational health."""

    async def health_check(self) -> dict[str, Any]:
        """Return a health payload; ``status`` should be ``ok`` or ``degraded``."""
        ...

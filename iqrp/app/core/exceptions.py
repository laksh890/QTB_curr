"""Hierarchical exception taxonomy for IQRP.

All platform errors inherit from :class:`IQRPError`. Domain modules raise
the most specific subclass so callers can handle failures precisely without
catching broad built-in exceptions.
"""

from __future__ import annotations

from typing import Any


class IQRPError(Exception):
    """Base exception for every IQRP failure.

    Attributes:
        message: Human-readable description of the failure.
        code: Stable machine-readable error code (e.g. ``DATA_INGEST_FAILED``).
        details: Optional structured context for logging and diagnostics.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "IQRP_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.code = code
        self.details: dict[str, Any] = details or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the error for structured logging or API responses."""
        return {
            "error": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }

    def __str__(self) -> str:
        if self.details:
            return f"[{self.code}] {self.message} | details={self.details}"
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, code={self.code!r}, details={self.details!r})"
        )


class DataError(IQRPError):
    """Raised when data ingestion, storage, or retrieval fails."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "DATA_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class ConfigurationError(IQRPError):
    """Raised when configuration is missing, invalid, or inconsistent."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "CONFIGURATION_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class ValidationError(IQRPError):
    """Raised when input, schema, or invariant validation fails."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "VALIDATION_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class ModelError(IQRPError):
    """Raised when statistical, ML, or regime model operations fail."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "MODEL_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class ExecutionError(IQRPError):
    """Raised when backtesting or live execution orchestration fails."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "EXECUTION_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)

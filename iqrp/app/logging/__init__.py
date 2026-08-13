"""Centralized logging for IQRP (Loguru + Rich)."""

from iqrp.app.logging.setup import get_logger, setup_logging

__all__ = ["get_logger", "setup_logging"]

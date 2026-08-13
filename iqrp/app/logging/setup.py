"""Loguru sink configuration: console, JSON, and rotating files."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO, cast

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

from iqrp.app.logging.formatters import json_formatter

if TYPE_CHECKING:
    from collections.abc import Callable

    from loguru import Record

    from iqrp.app.config.settings import LoggingSettings

_CONFIGURED = False


class _InterceptHandler(logging.Handler):
    """Route stdlib logging through Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = logging.currentframe()
        depth = 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(settings: LoggingSettings | None = None) -> None:
    """Configure process-wide logging sinks.

    Safe to call multiple times; subsequent calls replace sinks according to
    the provided settings.
    """
    global _CONFIGURED

    if settings is None:
        from iqrp.app.config.settings import LoggingSettings

        settings = LoggingSettings()

    logger.remove()

    format_fn: Callable[[Record], str] = cast(
        "Callable[[Record], str]",
        json_formatter,
    )

    if settings.console_logs:
        if settings.json_logs:
            logger.add(
                cast(TextIO, sys.stderr),
                level=settings.level,
                format=format_fn,
                enqueue=settings.enqueue,
                backtrace=settings.backtrace,
                diagnose=settings.diagnose,
                colorize=False,
            )
        else:
            rich_handler = RichHandler(
                console=Console(stderr=True),
                rich_tracebacks=settings.backtrace,
                markup=True,
                show_path=settings.diagnose,
                log_time_format="[%Y-%m-%d %H:%M:%S]",
            )
            logger.add(
                rich_handler,
                level=settings.level,
                format="{message}",
                enqueue=settings.enqueue,
                backtrace=settings.backtrace,
                diagnose=settings.diagnose,
            )

    if settings.file_logs:
        log_dir = Path(settings.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / settings.log_filename
        file_format: str | Callable[[Record], str] = (
            format_fn if settings.json_logs else _file_text_format()
        )
        logger.add(
            str(log_path),
            level=settings.level,
            format=file_format,
            rotation=settings.rotation,
            retention=settings.retention,
            compression=settings.compression,
            enqueue=settings.enqueue,
            backtrace=settings.backtrace,
            diagnose=settings.diagnose,
            encoding="utf-8",
        )

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in ("asyncio", "uvicorn", "httpx", "httpcore"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True

    _CONFIGURED = True
    logger.debug(
        "logging_configured level={} json={} console={} file={}",
        settings.level,
        settings.json_logs,
        settings.console_logs,
        settings.file_logs,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a Loguru logger bound with an optional component name."""
    if not _CONFIGURED:
        setup_logging()
    if name:
        return logger.bind(component=name)
    return logger


def _file_text_format() -> str:
    return "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | " "{name}:{function}:{line} | {message}"


def reset_logging_state() -> None:
    """Reset configuration flag (tests only)."""
    global _CONFIGURED
    _CONFIGURED = False
    logger.remove()

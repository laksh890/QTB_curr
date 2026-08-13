"""Unit tests for logging setup."""

from __future__ import annotations

from pathlib import Path

import pytest

from iqrp.app.config.settings import LoggingSettings
from iqrp.app.logging.setup import get_logger, setup_logging


@pytest.mark.unit
def test_setup_logging_console_only(tmp_path: Path) -> None:
    settings = LoggingSettings(
        level="INFO",
        json_logs=False,
        console_logs=True,
        file_logs=False,
        log_dir=tmp_path / "logs",
    )
    setup_logging(settings)
    log = get_logger("test")
    log.info("hello")


@pytest.mark.unit
def test_setup_logging_with_file(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    settings = LoggingSettings(
        level="DEBUG",
        json_logs=True,
        console_logs=False,
        file_logs=True,
        log_dir=log_dir,
        log_filename="test.log",
    )
    setup_logging(settings)
    log = get_logger("file-test")
    log.info("json_line")
    # Allow enqueue to flush for file sink
    import time

    time.sleep(0.2)
    assert (log_dir / "test.log").exists()

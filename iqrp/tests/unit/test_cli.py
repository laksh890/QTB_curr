"""Unit tests for the Typer CLI."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from iqrp.app.cli import app

runner = CliRunner()


@pytest.mark.unit
def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "iqrp" in result.stdout


@pytest.mark.unit
def test_info_command() -> None:
    result = runner.invoke(app, ["info", "--environment", "testing"])
    assert result.exit_code == 0
    assert "testing" in result.stdout.lower() or "IQRP" in result.stdout


@pytest.mark.unit
def test_doctor_command() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "ok" in result.stdout.lower() or "polars" in result.stdout.lower()

"""Unit tests for the exception hierarchy."""

from __future__ import annotations

import pytest

from iqrp.app.core.exceptions import (
    ConfigurationError,
    DataError,
    ExecutionError,
    IQRPError,
    ModelError,
    ValidationError,
)


@pytest.mark.unit
def test_base_exception_str_and_dict() -> None:
    err = IQRPError("boom", code="X", details={"a": 1})
    assert "boom" in str(err)
    assert err.to_dict() == {
        "error": "IQRPError",
        "code": "X",
        "message": "boom",
        "details": {"a": 1},
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cls", "code"),
    [
        (DataError, "DATA_ERROR"),
        (ConfigurationError, "CONFIGURATION_ERROR"),
        (ValidationError, "VALIDATION_ERROR"),
        (ModelError, "MODEL_ERROR"),
        (ExecutionError, "EXECUTION_ERROR"),
    ],
)
def test_domain_exception_defaults(cls: type[IQRPError], code: str) -> None:
    err = cls("failed")
    assert isinstance(err, IQRPError)
    assert err.code == code
    assert err.message == "failed"

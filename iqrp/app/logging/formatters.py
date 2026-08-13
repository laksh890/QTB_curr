"""Log record formatters."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any


def json_formatter(record: dict[str, Any]) -> str:
    """Serialize a Loguru record as a single-line JSON object.

    Loguru invokes this as a dynamic format string provider; the returned
    string must contain ``{message}`` so Loguru can inject the message body.
    We embed the full payload in the message field instead.
    """
    payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "level": record["level"].name,
        "message": record["message"],
        "logger": record["name"],
        "function": record["function"],
        "line": record["line"],
        "module": record["module"],
        "process": record["process"].id,
        "thread": record["thread"].id,
    }

    extra = record.get("extra") or {}
    if extra:
        payload["extra"] = dict(extra)
        if "component" in extra:
            payload["component"] = extra["component"]

    if record["exception"] is not None:
        payload["exception"] = str(record["exception"])

    serialized = json.dumps(payload, default=str, ensure_ascii=False)
    record["message"] = serialized
    return "{message}\n"

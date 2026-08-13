"""UUID generation helpers."""

from __future__ import annotations

import os
import time
import uuid
from typing import Literal

UUIDVersion = Literal[4, 7]


def _uuid7() -> uuid.UUID:
    """Generate a UUID version 7 (Unix epoch milliseconds + random).

    Implements RFC 9562. Used when :func:`uuid.uuid7` is unavailable.
    """
    # 48-bit unix timestamp in milliseconds
    timestamp_ms = int(time.time() * 1000) & ((1 << 48) - 1)
    rand_a = int.from_bytes(os.urandom(2), "big") & 0x0FFF  # 12 bits
    rand_b = int.from_bytes(os.urandom(8), "big") & ((1 << 62) - 1)  # 62 bits

    value = (timestamp_ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    return uuid.UUID(int=value)


def new_uuid(version: UUIDVersion = 4) -> uuid.UUID:
    """Return a new UUID.

    Version 4 is random. Version 7 is time-ordered (preferred for IDs that
    are stored or indexed chronologically).
    """
    if version == 4:
        return uuid.uuid4()
    if version == 7:
        uuid7_fn = getattr(uuid, "uuid7", None)
        if callable(uuid7_fn):
            result = uuid7_fn()
            if isinstance(result, uuid.UUID):
                return result
        return _uuid7()
    raise ValueError(f"Unsupported UUID version: {version}")


def new_id(prefix: str | None = None, *, version: UUIDVersion = 4) -> str:
    """Return a string identifier, optionally prefixed (e.g. ``run_...``)."""
    value = str(new_uuid(version))
    if prefix:
        return f"{prefix}_{value}"
    return value

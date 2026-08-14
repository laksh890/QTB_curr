"""Validation helpers for model→signal adapters (causality / schema)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class AdapterValidationError(ValueError):
    """Raised when adapter inputs violate causality or schema rules."""


def assert_no_future_columns(frame: pd.DataFrame) -> None:
    bad = [c for c in frame.columns if str(c).lower().startswith(("future_", "fwd_", "y_"))]
    if bad:
        raise AdapterValidationError(f"future/label columns rejected: {bad}")


def assert_timestamps_monotonic(timestamps: pd.Series | pd.DatetimeIndex) -> None:
    ts = pd.to_datetime(pd.Series(timestamps), utc=True)
    if not bool(ts.is_monotonic_increasing):
        raise AdapterValidationError("timestamps must be monotonically increasing")


def validate_signal_values(signal: pd.Series, *, allow_short: bool = True) -> dict[str, Any]:
    arr = signal.to_numpy(dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    uniq = sorted({float(x) for x in np.unique(finite)[:20]}) if finite.size else []
    if allow_short and any(abs(abs(u) - 1.0) > 1e-9 and abs(u) > 1e-12 for u in uniq if u != 0.0):
        # continuous allowed but categorical path should be in {-1,0,1}
        pass
    if not allow_short and (finite < -1e-12).any():
        raise AdapterValidationError("short signals not allowed but negatives present")
    return {"n": int(arr.size), "n_finite": int(finite.size), "unique_sample": uniq}


def train_val_oos_slices(n: int, *, train_frac: float = 0.5, validation_frac: float = 0.25) -> dict[str, slice]:
    if n <= 0:
        return {"train": slice(0, 0), "validation": slice(0, 0), "oos": slice(0, 0)}
    t = max(int(n * train_frac), 1)
    v = max(int(n * validation_frac), 0)
    o0 = min(t + v, n)
    return {"train": slice(0, t), "validation": slice(t, o0), "oos": slice(o0, n)}


__all__ = [
    "AdapterValidationError",
    "assert_no_future_columns",
    "assert_timestamps_monotonic",
    "train_val_oos_slices",
    "validate_signal_values",
]

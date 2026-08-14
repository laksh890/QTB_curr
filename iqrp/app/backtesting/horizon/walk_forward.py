"""Walk-forward / OOS evaluation helpers for horizon candidates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from iqrp.app.backtesting.horizon.metrics import horizon_performance_metrics
from iqrp.app.backtesting.performance.returns import as_returns

# Mapping used by apply_purge_embargo / evaluate_oos


def split_periods(
    n: int,
    *,
    train_frac: float = 0.6,
    validation_frac: float = 0.2,
) -> dict[str, slice]:
    """Configurable train / validation / OOS index slices."""
    if n <= 0:
        return {"train": slice(0, 0), "validation": slice(0, 0), "oos": slice(0, 0)}
    t = int(n * float(train_frac))
    v = int(n * float(validation_frac))
    t = max(t, 1)
    v = max(v, 0)
    o0 = min(t + v, n)
    return {
        "train": slice(0, t),
        "validation": slice(t, o0),
        "oos": slice(o0, n),
    }


def split_by_timestamps(
    timestamps: Sequence[Any],
    *,
    train_end: Any = None,
    validation_end: Any = None,
) -> dict[str, np.ndarray]:
    """Split by absolute dates when provided (else fractional)."""
    import pandas as pd

    ts = pd.to_datetime(list(timestamps), utc=True)
    idx = np.arange(len(ts))
    if train_end is None or validation_end is None:
        sl = split_periods(len(ts))
        return {
            "train": idx[sl["train"]],
            "validation": idx[sl["validation"]],
            "oos": idx[sl["oos"]],
        }
    te = pd.Timestamp(train_end)
    ve = pd.Timestamp(validation_end)
    if te.tzinfo is None:
        te = te.tz_localize("UTC")
    if ve.tzinfo is None:
        ve = ve.tz_localize("UTC")
    train = idx[ts <= te]
    validation = idx[(ts > te) & (ts <= ve)]
    oos = idx[ts > ve]
    return {"train": train, "validation": validation, "oos": oos}


def apply_purge_embargo(
    parts: Mapping[str, np.ndarray],
    *,
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> dict[str, np.ndarray]:
    """Drop overlapping bars between chronological splits for forward-return targets.

    Purge removes the last ``purge_bars`` of train that may overlap validation
    targets; embargo removes the first ``embargo_bars`` of validation/OOS after
    the prior window ends.
    """
    train = np.asarray(parts.get("train", np.array([], dtype=int)))
    validation = np.asarray(parts.get("validation", np.array([], dtype=int)))
    oos = np.asarray(parts.get("oos", np.array([], dtype=int)))
    p = max(int(purge_bars), 0)
    e = max(int(embargo_bars), 0)
    if p > 0 and train.size:
        train = train[:-p] if train.size > p else train[:0]
    if e > 0 and validation.size:
        validation = validation[e:] if validation.size > e else validation[:0]
    if e > 0 and oos.size:
        oos = oos[e:] if oos.size > e else oos[:0]
    return {"train": train, "validation": validation, "oos": oos}


def rolling_walk_forward_slices(
    n: int,
    *,
    n_windows: int = 3,
    train_frac: float = 0.5,
    validation_frac: float = 0.25,
) -> list[dict[str, slice]]:
    """Multiple chronological walk-forward windows over ``[0, n)``.

    Uses the existing fractional split within each expanding end index —
    not a second walk-forward engine.
    """
    if n <= 0 or n_windows <= 0:
        return []
    out: list[dict[str, slice]] = []
    for w in range(int(n_windows)):
        end = int(n * (w + 1) / n_windows)
        end = max(end, 3)
        out.append(split_periods(end, train_frac=train_frac, validation_frac=validation_frac))
    return out


def evaluate_oos(
    gross_returns: Any,
    net_returns: Any | None = None,
    *,
    timestamps: Sequence[Any] | None = None,
    train_end: Any = None,
    validation_end: Any = None,
    train_frac: float = 0.6,
    validation_frac: float = 0.2,
    periods_per_year: float = 252.0,
    purge_bars: int = 0,
    embargo_bars: int = 0,
    evaluate_fn: Callable[[np.ndarray, np.ndarray], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Report train / validation / OOS metrics for a horizon candidate."""
    g = as_returns(gross_returns)
    n = as_returns(net_returns) if net_returns is not None else g
    if timestamps is not None and len(timestamps) == g.size:
        parts = split_by_timestamps(
            timestamps, train_end=train_end, validation_end=validation_end
        )
    else:
        sl = split_periods(g.size, train_frac=train_frac, validation_frac=validation_frac)
        parts = {
            "train": np.arange(g.size)[sl["train"]],
            "validation": np.arange(g.size)[sl["validation"]],
            "oos": np.arange(g.size)[sl["oos"]],
        }
    if purge_bars or embargo_bars:
        parts = apply_purge_embargo(parts, purge_bars=purge_bars, embargo_bars=embargo_bars)

    def _eval(ix: np.ndarray) -> dict[str, Any]:
        if ix.size == 0:
            return {"n": 0, "net_sharpe": 0.0, "total_return": 0.0, "evaluated": False}
        if evaluate_fn is not None:
            return dict(evaluate_fn(g[ix], n[ix]))
        m = horizon_performance_metrics(g[ix], n[ix], periods_per_year=periods_per_year)
        return {
            "n": int(ix.size),
            "net_sharpe": m["net_sharpe"],
            "gross_sharpe": m["gross_sharpe"],
            "total_return": m["total_return_net"],
            "expectancy_per_trade": m.get("expectancy_per_trade"),
            "maximum_drawdown": m["maximum_drawdown"],
            "evaluated": True,
            "result_type": "out_of_sample_research" if ix is parts["oos"] else "in_sample_research",
        }

    train = _eval(parts["train"])
    validation = _eval(parts["validation"])
    oos = _eval(parts["oos"])
    oos["evaluated"] = bool(parts["oos"].size > 0)
    oos["result_type"] = "out_of_sample_research"
    return {
        "train": train,
        "validation": validation,
        "oos": oos,
        "purge_bars": int(purge_bars),
        "embargo_bars": int(embargo_bars),
        "note": (
            "Horizon selection must not rely on full-period in-sample Sharpe alone. "
            "OOS metrics are research/simulated results, not live performance. "
            "Purge/embargo reduce label leakage across chronological splits when "
            "forward returns are used as targets."
        ),
    }


__all__ = [
    "apply_purge_embargo",
    "evaluate_oos",
    "rolling_walk_forward_slices",
    "split_by_timestamps",
    "split_periods",
]

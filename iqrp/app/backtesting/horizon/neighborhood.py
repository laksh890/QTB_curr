"""Neighborhood robustness across nearby horizons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from iqrp.app.backtesting.horizon.parse import parse_timeframe


def neighborhood_for(
    center_seconds: float,
    candidates: Sequence[float],
    *,
    max_ratio: float = 2.0,
) -> list[float]:
    """Return candidate horizons within a multiplicative neighborhood of center."""
    c = float(center_seconds)
    out = []
    for x in candidates:
        x = float(x)
        if c <= 0 or x <= 0:
            continue
        ratio = max(x / c, c / x)
        if ratio <= float(max_ratio) + 1e-12:
            out.append(x)
    return sorted(out)


def neighborhood_robustness(
    results: Sequence[Mapping[str, Any]],
    *,
    metric_key: str = "net_sharpe",
    center_key: str = "data_timeframe",
    max_ratio: float = 2.0,
    fragile_relative_gap: float = 0.5,
) -> dict[str, dict[str, Any]]:
    """Measure stability of ``metric_key`` across neighboring horizons.

    If performance exists only at one highly specific configuration while
    neighbors are weak, flag ``FRAGILE HORIZON``.
    """
    # index by timeframe seconds
    indexed: dict[float, Mapping[str, Any]] = {}
    for r in results:
        spec = r.get("spec") if isinstance(r.get("spec"), Mapping) else r
        tf = spec.get(center_key) or spec.get("data_timeframe")
        if tf is None:
            continue
        sec = parse_timeframe(str(tf)).seconds
        metrics = r.get("metrics") if isinstance(r.get("metrics"), Mapping) else r
        indexed[sec] = {
            "seconds": sec,
            "label": str(tf),
            "value": float(metrics.get(metric_key, 0.0) or 0.0),
            "key": r.get("key") or spec.get("key"),
        }

    secs = sorted(indexed.keys())
    out: dict[str, dict[str, Any]] = {}
    for sec in secs:
        neighbors = neighborhood_for(sec, secs, max_ratio=max_ratio)
        vals = np.asarray([indexed[n]["value"] for n in neighbors], dtype=np.float64)
        center_v = float(indexed[sec]["value"])
        if vals.size <= 1:
            stability = 0.0
            fragile = True
            reason = "no neighboring horizons to compare"
        else:
            # stability: 1 - CV of neighbor metrics (clipped)
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            cv = abs(std / mean) if abs(mean) > 1e-12 else (1.0 if std > 0 else 0.0)
            stability = float(max(0.0, min(1.0, 1.0 - cv)))
            neighbor_mean_ex = float(np.mean([indexed[n]["value"] for n in neighbors if n != sec]))
            fragile = bool(
                center_v > 0
                and (neighbor_mean_ex <= 0 or (center_v - neighbor_mean_ex) > fragile_relative_gap * max(abs(center_v), 1e-9))
            )
            reason = (
                "FRAGILE HORIZON: spike isolated from neighbors"
                if fragile
                else "performance stable across neighboring horizons"
            )
        out[indexed[sec]["label"]] = {
            "center": indexed[sec]["label"],
            "neighbor_labels": [indexed[n]["label"] for n in neighbors],
            "neighbor_values": {indexed[n]["label"]: indexed[n]["value"] for n in neighbors},
            "stability_score": stability,
            "fragile": fragile,
            "reason": reason,
            "metric_key": metric_key,
        }
    return out


__all__ = ["neighborhood_for", "neighborhood_robustness"]

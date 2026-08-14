"""Visualization payloads for alpha research (no matplotlib dependency)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def ic_curve_payload(
    horizons: Sequence[int] | Sequence[float],
    ics: Sequence[float],
    *,
    title: str = "IC decay curve",
    half_life: float | None = None,
) -> dict[str, Any]:
    h = [int(x) for x in horizons]
    v = [float(x) if np.isfinite(x) else None for x in ics]
    return {
        "name": "ic_curve",
        "title": title,
        "type": "line",
        "x": h,
        "y": v,
        "x_label": "horizon",
        "y_label": "IC",
        "half_life": float(half_life) if half_life is not None and np.isfinite(half_life) else None,
    }


def ic_series_payload(
    indices: Sequence[int] | Sequence[float],
    ics: Sequence[float],
    *,
    title: str = "Rolling IC",
    baseline: float | None = None,
) -> dict[str, Any]:
    return {
        "name": "ic_series",
        "title": title,
        "type": "line",
        "x": [int(i) for i in indices],
        "y": [float(v) if np.isfinite(v) else None for v in ics],
        "baseline": float(baseline) if baseline is not None else None,
        "x_label": "time",
        "y_label": "IC",
    }


def decay_payload(
    decay_report: Mapping[str, Any],
    *,
    title: str = "Signal decay",
) -> dict[str, Any]:
    """Accepts ``ic_decay_curve`` output or a raw curve list."""
    if "curve" in decay_report:
        curve = decay_report["curve"]
        horizons = [c["horizon"] for c in curve]
        ics = [c["ic"] for c in curve]
        half_life = decay_report.get("half_life")
    else:
        horizons = list(decay_report.get("horizons", []))
        ics = list(decay_report.get("ics", decay_report.get("ic", [])))
        half_life = decay_report.get("half_life")
    payload = ic_curve_payload(horizons, ics, title=title, half_life=half_life)
    payload["decay_rate"] = decay_report.get("decay_rate")
    return payload


def regime_bars_payload(
    by_regime: Mapping[str, Any],
    *,
    metric: str = "ic",
    title: str = "Regime performance",
) -> dict[str, Any]:
    """Bar chart of a metric by regime label.

    ``by_regime`` values may be floats or dicts containing ``metric``.
    """
    labels: list[str] = []
    values: list[float | None] = []
    for k in sorted(by_regime.keys(), key=str):
        labels.append(str(k))
        v = by_regime[k]
        if isinstance(v, Mapping):
            raw = v.get(metric, v.get("ic", v.get("mean_return")))
        else:
            raw = v
        try:
            fv = float(raw)  # type: ignore[arg-type]
            values.append(fv if np.isfinite(fv) else None)
        except (TypeError, ValueError):
            values.append(None)
    return {
        "name": "regime_bars",
        "title": title,
        "type": "bar",
        "labels": labels,
        "values": values,
        "metric": metric,
    }


def correlation_heatmap_payload(
    corr: Mapping[str, Any] | np.ndarray,
    *,
    labels: Sequence[str] | None = None,
    title: str = "Signal correlation",
) -> dict[str, Any]:
    if isinstance(corr, Mapping) and "matrix" in corr:
        mat = np.asarray(corr["matrix"], dtype=np.float64)
        labs = list(corr.get("labels", labels or []))
    else:
        mat = np.asarray(corr, dtype=np.float64)
        labs = list(labels) if labels is not None else [f"s{i}" for i in range(mat.shape[0])]
    if len(labs) != mat.shape[0]:
        labs = [f"s{i}" for i in range(mat.shape[0])]
    return {
        "name": "correlation_heatmap",
        "title": title,
        "type": "heatmap",
        "labels": labs,
        "matrix": [[float(v) if np.isfinite(v) else None for v in row] for row in mat.tolist()],
    }


def weight_bars_payload(
    weights: Mapping[str, float],
    *,
    title: str = "Ensemble weights",
) -> dict[str, Any]:
    labels = list(weights.keys())
    values = [float(weights[k]) for k in labels]
    return {
        "name": "weight_bars",
        "title": title,
        "type": "bar",
        "labels": labels,
        "values": values,
    }


def retirement_status_payload(
    result: Mapping[str, Any],
    *,
    signal_name: str | None = None,
) -> dict[str, Any]:
    return {
        "name": "retirement_status",
        "title": f"Retirement: {signal_name}" if signal_name else "Retirement status",
        "type": "status",
        "status": result.get("status", result.get("recommend")),
        "reasons": list(result.get("reasons", [])),
        "metrics": result.get("metrics", {}),
    }


def alpha_viz_bundle(
    *,
    ic_curve: Mapping[str, Any] | None = None,
    rolling_ic: Mapping[str, Any] | None = None,
    regime: Mapping[str, Any] | None = None,
    correlation: Mapping[str, Any] | None = None,
    weights: Mapping[str, float] | None = None,
    retirement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bundle chart-ready payloads for dashboards."""
    out: dict[str, Any] = {"name": "alpha_viz_bundle"}
    if ic_curve is not None:
        out["decay"] = decay_payload(ic_curve)
    if rolling_ic is not None:
        out["rolling_ic"] = ic_series_payload(
            rolling_ic.get("indices", list(range(len(rolling_ic.get("ic", []))))),
            rolling_ic.get("ic", []),
            baseline=rolling_ic.get("mean"),
        )
    if regime is not None:
        by = regime.get("by_regime", regime)
        out["regime_bars"] = regime_bars_payload(by)
    if correlation is not None:
        out["correlation"] = correlation_heatmap_payload(correlation)
    if weights is not None:
        out["weights"] = weight_bars_payload(weights)
    if retirement is not None:
        out["retirement"] = retirement_status_payload(retirement)
    return out

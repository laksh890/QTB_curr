"""Normalize heterogeneous risk metrics onto [0, 1] while preserving originals."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.ensemble.config import EnsembleSettings, NormalizationRef
from iqrp.app.risk.ensemble.types import NormalizedMetric, utc_now_iso


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("value", "score", "risk", "normalized"):
            if key in value:
                return _as_float(value[key])
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def normalize_value(
    raw: float,
    *,
    zero: float,
    one: float,
    invert: bool = False,
) -> float:
    """Map raw observation to [0, 1] risk with optional invert (higher raw = lower risk)."""
    if invert:
        # liquidity_score: 1 good → 0 risk; 0 bad → 1 risk
        lo, hi = min(zero, one), max(zero, one)
        if abs(hi - lo) < 1e-12:
            return 0.0
        # When invert=True with zero=1, one=0: risk = (zero - raw) / (zero - one)
        span = zero - one
        if abs(span) < 1e-12:
            return float(np.clip(1.0 - raw, 0.0, 1.0))
        risk = (zero - float(raw)) / span
        return float(np.clip(risk, 0.0, 1.0))

    span = float(one) - float(zero)
    if abs(span) < 1e-12:
        return 0.0 if raw <= zero else 1.0
    return float(np.clip((float(raw) - float(zero)) / span, 0.0, 1.0))


def _ref_for(settings: EnsembleSettings, name: str) -> NormalizationRef:
    refs = settings.normalization
    if name in refs:
        return refs[name]
    # Aliases
    aliases = {
        "vol": "volatility",
        "realized_vol": "volatility",
        "garch_vol": "volatility",
        "es": "expected_shortfall",
        "expected_shortfall": "expected_shortfall",
        "dd": "drawdown",
        "current_drawdown": "drawdown",
        "max_drawdown": "drawdown",
        "liquidity": "liquidity_score",
        "corr": "correlation",
        "avg_correlation": "correlation",
        "hhi": "concentration",
        "herfindahl": "concentration",
    }
    key = aliases.get(name, name)
    if key in refs:
        return refs[key]
    return NormalizationRef(zero=0.0, one=1.0, invert=False)


def normalize_metric(
    name: str,
    value: Any,
    *,
    settings: EnsembleSettings,
    timestamp: str | None = None,
    model_version: str | None = None,
    method: str | None = None,
) -> NormalizedMetric | None:
    raw = _as_float(value)
    if raw is None:
        return None
    ref = _ref_for(settings, name)
    # Align drawdown high end with trading_halt threshold when using default drawdown key
    if name in {"drawdown", "current_drawdown", "max_drawdown", "dd"}:
        one = float(settings.drawdown.trading_halt)
        zero = 0.0
        invert = False
    else:
        zero, one, invert = float(ref.zero), float(ref.one), bool(ref.invert)

    norm = normalize_value(raw, zero=zero, one=one, invert=invert)
    return NormalizedMetric(
        name=name,
        original_value=float(raw),
        normalized_value=float(norm),
        method=method or ("inverted_linear" if invert else "linear_threshold"),
        reference={"zero": float(zero), "one": float(one), "invert": float(invert)},
        timestamp=timestamp or utc_now_iso(),
        model_version=model_version or settings.model_version,
        unit="",
        metadata={},
    )


def normalize_metrics(
    metrics: dict[str, Any],
    *,
    settings: EnsembleSettings,
    timestamp: str | None = None,
) -> dict[str, NormalizedMetric]:
    """Normalize all numeric (or value-bearing) metrics; skip non-numeric keys silently."""
    out: dict[str, NormalizedMetric] = {}
    ts = timestamp or utc_now_iso()
    for name, value in metrics.items():
        if name.startswith("_"):
            continue
        nm = normalize_metric(
            str(name),
            value,
            settings=settings,
            timestamp=ts,
            model_version=settings.model_version,
        )
        if nm is not None:
            out[str(name)] = nm
    return out


class MetricNormalizer:
    """Stateful wrapper around normalize_metrics for ensemble pipelines."""

    def __init__(self, settings: EnsembleSettings) -> None:
        self.settings = settings

    def normalize(self, metrics: dict[str, Any], *, timestamp: str | None = None) -> dict[str, NormalizedMetric]:
        return normalize_metrics(metrics, settings=self.settings, timestamp=timestamp)

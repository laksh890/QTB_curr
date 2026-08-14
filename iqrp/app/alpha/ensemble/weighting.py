"""Ensemble weighting schemes for alpha signals.

IMPORTANT: never overweight solely on historical Sharpe. Weights combine IC,
uncertainty, correlation penalty, capacity, decay, and stability with
configurable component weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

WeightMethod = Literal[
    "equal",
    "ic",
    "risk_adj",
    "corr_adj",
    "regime",
    "dynamic",
    "composite",
]

DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "ic": 0.30,
    "stability": 0.20,
    "capacity": 0.15,
    "decay": 0.15,  # higher decay metric = worse; inverted below
    "corr_penalty": 0.10,  # higher = worse; inverted
    "uncertainty": 0.05,  # higher = worse; inverted
    "sharpe": 0.05,  # capped contribution — never dominant
}


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def normalize_weights(
    weights: Mapping[str, float] | np.ndarray,
    names: Sequence[str] | None = None,
    *,
    min_weight: float = 0.0,
) -> dict[str, float]:
    if isinstance(weights, Mapping):
        names = list(weights.keys()) if names is None else list(names)
        arr = np.asarray([float(weights.get(n, 0.0)) for n in names], dtype=np.float64)
    else:
        arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if names is None:
            names = [f"s{i}" for i in range(arr.size)]
    arr = np.clip(arr, 0.0, None)
    if float(arr.sum()) <= 0:
        arr = np.ones(len(names), dtype=np.float64)
    if min_weight > 0:
        arr = np.maximum(arr, min_weight)
    arr = arr / arr.sum()
    return {n: float(w) for n, w in zip(names, arr)}


def equal_weights(names: Sequence[str]) -> dict[str, float]:
    n = max(len(names), 1)
    return dict.fromkeys(names, 1.0 / n)


def _metric(m: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    v = m.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def signal_quality_score(
    metrics: Mapping[str, Any],
    *,
    score_weights: Mapping[str, float] | None = None,
) -> float:
    """Composite quality score in roughly [0, 1]; Sharpe is a small optional term."""
    sw = dict(DEFAULT_SCORE_WEIGHTS)
    if score_weights:
        sw.update({k: float(v) for k, v in score_weights.items()})

    ic = abs(_metric(metrics, "ic"))
    # map |IC| ~0.05 → ~0.5, saturate near 0.15
    ic_score = _clip01(ic / 0.10)

    stability = _clip01(_metric(metrics, "stability", 0.5))
    capacity = _clip01(_metric(metrics, "capacity", 0.5))

    decay = _clip01(_metric(metrics, "decay", 0.5))
    decay_score = 1.0 - decay  # low decay preferred

    corr_pen = _clip01(_metric(metrics, "corr_penalty", 0.0))
    corr_score = 1.0 - corr_pen

    # uncertainty: prefer explicit key; else invert stability proxy
    if "uncertainty" in metrics:
        unc_score = 1.0 - _clip01(_metric(metrics, "uncertainty"))
    else:
        unc_score = stability

    # Sharpe contributes at most its configured weight; soft-saturate
    sharpe = _metric(metrics, "sharpe", 0.0)
    sharpe_score = _clip01(abs(sharpe) / 3.0)

    components = {
        "ic": ic_score,
        "stability": stability,
        "capacity": capacity,
        "decay": decay_score,
        "corr_penalty": corr_score,
        "uncertainty": unc_score,
        "sharpe": sharpe_score,
    }
    total_w = sum(max(sw.get(k, 0.0), 0.0) for k in components) or 1.0
    score = sum(components[k] * max(sw.get(k, 0.0), 0.0) for k in components) / total_w
    return float(np.clip(score, 0.0, 1.0))


def compute_ensemble_weights(
    metrics_by_signal: Mapping[str, Mapping[str, Any]],
    *,
    method: WeightMethod = "composite",
    score_weights: Mapping[str, float] | None = None,
    regime_scores: Mapping[str, float] | None = None,
    min_weight: float = 0.0,
) -> dict[str, float]:
    """Compute ensemble weights from per-signal research metrics.

    Parameters
    ----------
    metrics_by_signal :
        Mapping ``signal_name -> {ic, sharpe, decay, stability, capacity,
        corr_penalty, uncertainty?, ...}``.
    method :
        ``equal``, ``ic``, ``risk_adj``, ``corr_adj``, ``regime``, ``dynamic``,
        or ``composite`` (default — multi-factor score, not Sharpe-only).
    """
    names = list(metrics_by_signal.keys())
    if not names:
        return {}

    if method == "equal":
        return equal_weights(names)

    raw: dict[str, float] = {}
    for name in names:
        m = metrics_by_signal[name]
        if method == "ic":
            raw[name] = max(abs(_metric(m, "ic")), 0.0)
        elif method == "risk_adj":
            # IC / uncertainty style; fall back to IC * stability
            unc = _metric(m, "uncertainty", 1.0 - _metric(m, "stability", 0.5))
            raw[name] = abs(_metric(m, "ic")) / max(unc, 0.05)
        elif method == "corr_adj":
            pen = _clip01(_metric(m, "corr_penalty", 0.0))
            raw[name] = abs(_metric(m, "ic")) * (1.0 - pen) * _clip01(_metric(m, "stability", 0.5))
        elif method == "regime":
            base = signal_quality_score(m, score_weights=score_weights)
            boost = 1.0
            if regime_scores is not None:
                boost = max(float(regime_scores.get(name, 1.0)), 0.0)
            raw[name] = base * boost
        elif method in ("dynamic", "composite"):
            raw[name] = signal_quality_score(m, score_weights=score_weights)
        else:
            raise ValueError(f"unknown weighting method: {method}")

    return normalize_weights(raw, names, min_weight=min_weight)


def ic_weights(
    metrics_by_signal: Mapping[str, Mapping[str, Any]], **kwargs: Any
) -> dict[str, float]:
    return compute_ensemble_weights(metrics_by_signal, method="ic", **kwargs)


def risk_adjusted_weights(
    metrics_by_signal: Mapping[str, Mapping[str, Any]], **kwargs: Any
) -> dict[str, float]:
    return compute_ensemble_weights(metrics_by_signal, method="risk_adj", **kwargs)


def correlation_adjusted_weights(
    metrics_by_signal: Mapping[str, Mapping[str, Any]], **kwargs: Any
) -> dict[str, float]:
    return compute_ensemble_weights(metrics_by_signal, method="corr_adj", **kwargs)


def regime_weights(
    metrics_by_signal: Mapping[str, Mapping[str, Any]],
    regime_scores: Mapping[str, float],
    **kwargs: Any,
) -> dict[str, float]:
    return compute_ensemble_weights(
        metrics_by_signal, method="regime", regime_scores=regime_scores, **kwargs
    )


def dynamic_weights(
    metrics_by_signal: Mapping[str, Mapping[str, Any]], **kwargs: Any
) -> dict[str, float]:
    return compute_ensemble_weights(metrics_by_signal, method="dynamic", **kwargs)

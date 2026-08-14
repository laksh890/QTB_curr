"""Combine multiple alpha signals into an ensemble panel."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from iqrp.app.alpha.ensemble.weighting import compute_ensemble_weights, equal_weights


def _stack_signals(
    signals: Mapping[str, Any] | Sequence[Any],
    names: Sequence[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Return ``(names, array)`` with array shape ``(T, N, K)`` or ``(T, K)``."""
    if isinstance(signals, Mapping):
        keys = list(signals.keys()) if names is None else list(names)
        mats = [np.asarray(signals[k], dtype=np.float64) for k in keys]
    else:
        mats = [np.asarray(s, dtype=np.float64) for s in signals]
        keys = list(names) if names is not None else [f"s{i}" for i in range(len(mats))]
    if not mats:
        return [], np.empty((0, 0, 0), dtype=np.float64)

    shapes = {m.shape for m in mats}
    if len(shapes) != 1:
        raise ValueError(f"all signals must share the same shape, got {shapes}")
    stacked = np.stack(mats, axis=-1)
    return keys, stacked


def combine_signals(
    signals: Mapping[str, Any] | Sequence[Any],
    weights: Mapping[str, float] | Sequence[float] | None = None,
    *,
    names: Sequence[str] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Weighted sum of signals. Returns panel with same leading dims as inputs."""
    keys, stacked = _stack_signals(signals, names=names)
    if not keys:
        return np.asarray([], dtype=np.float64)

    if weights is None:
        wmap = equal_weights(keys)
    elif isinstance(weights, Mapping):
        wmap = {k: float(weights.get(k, 0.0)) for k in keys}
    else:
        arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if arr.size != len(keys):
            raise ValueError("weights length must match number of signals")
        wmap = {k: float(v) for k, v in zip(keys, arr)}

    w = np.asarray([wmap[k] for k in keys], dtype=np.float64)
    if normalize:
        s = float(np.sum(np.abs(w)))
        if s <= 0:
            w = np.full(len(keys), 1.0 / len(keys))
        else:
            w = w / np.sum(w) if float(np.sum(w)) > 0 else w / s

    # stacked (... , K)
    return np.nansum(stacked * w.reshape((1,) * (stacked.ndim - 1) + (-1,)), axis=-1)


def combine_from_metrics(
    signals: Mapping[str, Any],
    metrics_by_signal: Mapping[str, Mapping[str, Any]],
    *,
    method: str = "composite",
    score_weights: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute quality-aware weights then combine."""
    w = compute_ensemble_weights(
        metrics_by_signal,
        method=method,  # type: ignore[arg-type]
        score_weights=score_weights,
    )
    # only signals present in both
    common = {k: signals[k] for k in signals if k in w}
    w_common = {k: w[k] for k in common}
    return combine_signals(common, w_common), w_common


def rank_average_combine(
    signals: Mapping[str, Any] | Sequence[Any],
    *,
    names: Sequence[str] | None = None,
) -> np.ndarray:
    """Equal-weight average of cross-sectional percentile ranks."""
    from iqrp.app.alpha.cross_section.ranking import cross_sectional_percentile

    keys, stacked = _stack_signals(signals, names=names)
    if not keys:
        return np.asarray([], dtype=np.float64)
    ranked = []
    for k in range(stacked.shape[-1]):
        panel = stacked[..., k]
        if panel.ndim == 1:
            # time-series: use running rank vs history is not CS — zscore-like identity
            ranked.append(panel)
        else:
            ranked.append(cross_sectional_percentile(panel, axis=1))
    return np.nanmean(np.stack(ranked, axis=-1), axis=-1)


def majority_sign_combine(
    signals: Mapping[str, Any] | Sequence[Any],
    *,
    names: Sequence[str] | None = None,
) -> np.ndarray:
    """Sign of the average sign across members (majority vote)."""
    keys, stacked = _stack_signals(signals, names=names)
    if not keys:
        return np.asarray([], dtype=np.float64)
    return np.sign(np.nanmean(np.sign(stacked), axis=-1))

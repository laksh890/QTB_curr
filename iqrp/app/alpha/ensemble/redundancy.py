"""Detect redundant, highly correlated, and nested alpha signals."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from iqrp.app.alpha.ensemble.correlation import (
    correlation_penalty_vector,
    signal_correlation_matrix,
)


def find_high_correlation_pairs(
    corr: Mapping[str, Any] | np.ndarray,
    *,
    labels: Sequence[str] | None = None,
    threshold: float = 0.85,
) -> list[dict[str, Any]]:
    if isinstance(corr, Mapping) and "matrix" in corr:
        mat = np.asarray(corr["matrix"], dtype=np.float64)
        labs = list(corr.get("labels", labels or []))
    else:
        mat = np.asarray(corr, dtype=np.float64)
        labs = list(labels) if labels is not None else [f"s{i}" for i in range(mat.shape[0])]
    if len(labs) != mat.shape[0]:
        labs = [f"s{i}" for i in range(mat.shape[0])]
    pairs: list[dict[str, Any]] = []
    for i in range(mat.shape[0]):
        for j in range(i + 1, mat.shape[0]):
            c = mat[i, j]
            if np.isfinite(c) and abs(float(c)) >= threshold:
                pairs.append(
                    {
                        "a": labs[i],
                        "b": labs[j],
                        "correlation": float(c),
                        "abs_correlation": float(abs(c)),
                    }
                )
    pairs.sort(key=lambda p: -p["abs_correlation"])
    return pairs


def detect_nested_signals(
    signals: Mapping[str, Any],
    *,
    r2_threshold: float = 0.95,
    min_obs: int = 30,
) -> list[dict[str, Any]]:
    """Flag signal A as nested in B if OLS R² of A on B exceeds threshold."""
    names = list(signals.keys())
    series = {}
    for n in names:
        v = np.asarray(signals[n], dtype=np.float64)
        if v.ndim > 1:
            v = np.nanmean(v, axis=1)
        series[n] = v.reshape(-1)

    nested: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for b in names:
            if a == b:
                continue
            x, y = series[b], series[a]
            n = min(x.size, y.size)
            x, y = x[:n], y[:n]
            m = np.isfinite(x) & np.isfinite(y)
            if int(m.sum()) < min_obs:
                continue
            xf, yf = x[m], y[m]
            var = float(np.var(xf))
            if var < 1e-18:
                continue
            # simple univariate R²
            beta = float(np.cov(yf, xf, ddof=1)[0, 1] / var)
            alpha = float(np.mean(yf) - beta * np.mean(xf))
            pred = alpha + beta * xf
            ss_res = float(np.sum((yf - pred) ** 2))
            ss_tot = float(np.sum((yf - np.mean(yf)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            if r2 >= r2_threshold:
                nested.append(
                    {
                        "child": a,
                        "parent": b,
                        "r2": float(r2),
                        "beta": beta,
                    }
                )
    nested.sort(key=lambda d: -d["r2"])
    return nested


def feature_overlap(
    feature_sets: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Jaccard overlap between signal feature sets."""
    names = list(feature_sets.keys())
    sets = {n: set(feature_sets[n]) for n in names}
    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            inter = sets[a] & sets[b]
            union = sets[a] | sets[b]
            jacc = len(inter) / len(union) if union else 0.0
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "jaccard": float(jacc),
                    "shared_features": sorted(inter),
                    "n_shared": len(inter),
                }
            )
    pairs.sort(key=lambda p: -p["jaccard"])
    return {"name": "feature_overlap", "pairs": pairs}


def redundancy_report(
    signals: Mapping[str, Any],
    *,
    corr_threshold: float = 0.85,
    r2_threshold: float = 0.95,
    feature_sets: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Full redundancy diagnosis across a signal book."""
    corr = signal_correlation_matrix(signals, kind="prediction")
    pairs = find_high_correlation_pairs(corr, threshold=corr_threshold)
    nested = detect_nested_signals(signals, r2_threshold=r2_threshold)
    penalties = correlation_penalty_vector(corr)
    suggested: list[str] = []
    # suggest dropping the lower-|mean| member of each high-corr pair
    means = {
        k: float(np.nanmean(np.abs(np.asarray(v, dtype=np.float64))))
        for k, v in signals.items()
    }
    for p in pairs:
        drop = p["a"] if means.get(p["a"], 0) < means.get(p["b"], 0) else p["b"]
        if drop not in suggested:
            suggested.append(drop)
    for n in nested:
        if n["child"] not in suggested:
            suggested.append(n["child"])

    out: dict[str, Any] = {
        "name": "redundancy_report",
        "correlation": corr,
        "high_correlation_pairs": pairs,
        "nested_signals": nested,
        "corr_penalties": penalties,
        "suggested_removals": suggested,
    }
    if feature_sets is not None:
        out["feature_overlap"] = feature_overlap(feature_sets)
    return out

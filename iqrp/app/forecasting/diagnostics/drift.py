"""Prediction and feature drift detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class DriftReport:
    score: float
    method: str
    drifted: bool
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "method": self.method,
            "drifted": self.drifted,
            "threshold": self.threshold,
            "details": dict(self.details),
        }


def psi(expected: np.ndarray, actual: np.ndarray, *, n_bins: int = 10) -> float:
    """Population Stability Index between two univariate samples."""
    e = np.asarray(expected, dtype=np.float64).reshape(-1)
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    if e.size == 0 or a.size == 0:
        return 0.0
    edges = np.unique(np.quantile(e, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        return 0.0
    e_hist, _ = np.histogram(e, bins=edges)
    a_hist, _ = np.histogram(a, bins=edges)
    e_pct = np.clip(e_hist / max(e.size, 1), 1e-6, None)
    a_pct = np.clip(a_hist / max(a.size, 1), 1e-6, None)
    e_pct = e_pct / e_pct.sum()
    a_pct = a_pct / a_pct.sum()
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def ks_statistic(expected: np.ndarray, actual: np.ndarray) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (no p-value)."""
    e = np.sort(np.asarray(expected, dtype=np.float64).reshape(-1))
    a = np.sort(np.asarray(actual, dtype=np.float64).reshape(-1))
    if e.size == 0 or a.size == 0:
        return 0.0
    # empirical CDF difference on pooled grid
    grid = np.sort(np.concatenate([e, a]))
    cdf_e = np.searchsorted(e, grid, side="right") / e.size
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    return float(np.max(np.abs(cdf_e - cdf_a)))


def mean_shift(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.asarray(expected, dtype=np.float64).reshape(-1)
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    if e.size == 0 or a.size == 0:
        return 0.0
    return float(abs(np.mean(a) - np.mean(e)) / max(np.std(e), 1e-12))


def detect_prediction_drift(
    reference_predictions: np.ndarray,
    current_predictions: np.ndarray,
    *,
    method: str = "psi",
    threshold: float | None = None,
) -> DriftReport:
    m = method.lower()
    if m == "ks":
        score = ks_statistic(reference_predictions, current_predictions)
        thr = 0.1 if threshold is None else float(threshold)
    elif m == "mean_shift":
        score = mean_shift(reference_predictions, current_predictions)
        thr = 1.0 if threshold is None else float(threshold)
    else:
        score = psi(reference_predictions, current_predictions)
        thr = 0.2 if threshold is None else float(threshold)
    return DriftReport(
        score=score,
        method=m,
        drifted=score >= thr,
        threshold=thr,
        details={
            "n_reference": int(np.asarray(reference_predictions).size),
            "n_current": int(np.asarray(current_predictions).size),
        },
    )


def detect_feature_drift(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    threshold: float = 0.2,
) -> DriftReport:
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if cur.ndim == 1:
        cur = cur.reshape(-1, 1)
    f = min(ref.shape[1], cur.shape[1])
    names = feature_names or [f"f{i}" for i in range(f)]
    per_feature = {}
    for j in range(f):
        per_feature[names[j] if j < len(names) else f"f{j}"] = psi(ref[:, j], cur[:, j])
    score = float(np.mean(list(per_feature.values()))) if per_feature else 0.0
    return DriftReport(
        score=score,
        method="psi_features",
        drifted=score >= threshold,
        threshold=threshold,
        details={"per_feature": per_feature},
    )

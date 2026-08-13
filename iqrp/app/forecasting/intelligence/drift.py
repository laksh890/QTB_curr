"""Concept / data / prediction drift detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.forecasting.intelligence.config import DriftConfig


@dataclass(slots=True)
class DriftReport:
    feature_drift: dict[str, float]
    prediction_drift: float
    target_drift: float
    covariate_shift: float
    performance_degradation: float
    triggered: bool
    reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_drift": dict(self.feature_drift),
            "prediction_drift": self.prediction_drift,
            "target_drift": self.target_drift,
            "covariate_shift": self.covariate_shift,
            "performance_degradation": self.performance_degradation,
            "triggered": self.triggered,
            "reasons": list(self.reasons),
            "metadata": dict(self.metadata),
        }


def population_stability_index(expected: np.ndarray, actual: np.ndarray, *, bins: int = 10) -> float:
    e = np.asarray(expected, dtype=np.float64).reshape(-1)
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    qs = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(e, qs))
    if edges.size < 3:
        return 0.0
    e_hist, _ = np.histogram(e, bins=edges)
    a_hist, _ = np.histogram(a, bins=edges)
    e_pct = e_hist / max(e_hist.sum(), 1)
    a_pct = a_hist / max(a_hist.sum(), 1)
    e_pct = np.clip(e_pct, 1e-6, None)
    a_pct = np.clip(a_pct, 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    x = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    if x.size == 0 or y.size == 0:
        return 0.0
    # approximate KS via CDF on pooled grid
    grid = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, grid, side="right") / x.size
    cdf_y = np.searchsorted(y, grid, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def detect_drift(
    *,
    ref_features: np.ndarray,
    cur_features: np.ndarray,
    ref_preds: np.ndarray | None = None,
    cur_preds: np.ndarray | None = None,
    ref_target: np.ndarray | None = None,
    cur_target: np.ndarray | None = None,
    ref_metric: float | None = None,
    cur_metric: float | None = None,
    config: DriftConfig | None = None,
    feature_names: list[str] | None = None,
) -> DriftReport:
    cfg = config or DriftConfig()
    ref_f = np.asarray(ref_features, dtype=np.float64)
    cur_f = np.asarray(cur_features, dtype=np.float64)
    if ref_f.ndim == 1:
        ref_f = ref_f.reshape(-1, 1)
        cur_f = cur_f.reshape(-1, 1)
    n_feat = min(ref_f.shape[1], cur_f.shape[1])
    names = feature_names or [f"f{i}" for i in range(n_feat)]
    feature_drift = {
        names[i]: population_stability_index(ref_f[:, i], cur_f[:, i]) for i in range(n_feat)
    }
    pred_drift = 0.0
    if ref_preds is not None and cur_preds is not None:
        pred_drift = ks_statistic(ref_preds, cur_preds)
    target_drift = 0.0
    if ref_target is not None and cur_target is not None:
        target_drift = ks_statistic(ref_target, cur_target)
    covariate_shift = float(np.mean(list(feature_drift.values()))) if feature_drift else 0.0
    perf_deg = 0.0
    if ref_metric is not None and cur_metric is not None and abs(ref_metric) > 1e-12:
        perf_deg = float((cur_metric - ref_metric) / abs(ref_metric))
    reasons = []
    if any(v > cfg.feature_psi_threshold for v in feature_drift.values()):
        reasons.append("feature_drift")
    if pred_drift > cfg.prediction_ks_threshold:
        reasons.append("prediction_drift")
    if target_drift > cfg.prediction_ks_threshold:
        reasons.append("target_drift")
    if covariate_shift > cfg.feature_psi_threshold:
        reasons.append("covariate_shift")
    if perf_deg > cfg.performance_drop:
        reasons.append("performance_degradation")
    triggered = bool(reasons) if cfg.enabled else False
    return DriftReport(
        feature_drift=feature_drift,
        prediction_drift=pred_drift,
        target_drift=target_drift,
        covariate_shift=covariate_shift,
        performance_degradation=perf_deg,
        triggered=triggered,
        reasons=reasons,
    )

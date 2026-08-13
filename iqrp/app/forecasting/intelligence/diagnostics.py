"""Diagnostics for forecast intelligence pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.forecasting.intelligence.ranking import RankedModel


@dataclass
class DiagnosticReport:
    residual_mean: float
    residual_std: float
    residual_skew: float
    autocorrelation_lag1: float
    outlier_rate: float
    coverage_95: float | None
    notes: list[str] = field(default_factory=list)
    per_model: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "residual_mean": self.residual_mean,
            "residual_std": self.residual_std,
            "residual_skew": self.residual_skew,
            "autocorrelation_lag1": self.autocorrelation_lag1,
            "outlier_rate": self.outlier_rate,
            "coverage_95": self.coverage_95,
            "notes": list(self.notes),
            "per_model": {k: dict(v) for k, v in self.per_model.items()},
        }


def _skew(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 3:
        return 0.0
    m = float(np.mean(x))
    s = float(np.std(x)) + 1e-12
    return float(np.mean(((x - m) / s) ** 3))


def diagnose_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> DiagnosticReport:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = min(yt.size, yp.size)
    yt, yp = yt[:n], yp[:n]
    resid = yt - yp
    mean = float(np.mean(resid)) if n else 0.0
    std = float(np.std(resid)) if n else 0.0
    skew = _skew(resid)
    ac1 = 0.0
    if n >= 3:
        a, b = resid[:-1], resid[1:]
        if np.std(a) > 1e-12 and np.std(b) > 1e-12:
            ac1 = float(np.corrcoef(a, b)[0, 1])
    outlier = float(np.mean(np.abs(resid) > 3.0 * (std + 1e-12))) if n else 0.0
    cov = None
    if lower is not None and upper is not None:
        lo = np.asarray(lower, dtype=np.float64).reshape(-1)[:n]
        up = np.asarray(upper, dtype=np.float64).reshape(-1)[:n]
        cov = float(np.mean((yt >= lo) & (yt <= up)))
    notes: list[str] = []
    if abs(mean) > 0.5 * (std + 1e-12):
        notes.append("biased_residuals")
    if abs(ac1) > 0.3:
        notes.append("residual_autocorrelation")
    if outlier > 0.05:
        notes.append("heavy_tail_outliers")
    return DiagnosticReport(
        residual_mean=mean,
        residual_std=std,
        residual_skew=skew,
        autocorrelation_lag1=ac1,
        outlier_rate=outlier,
        coverage_95=cov,
        notes=notes,
    )


def diagnose_leaderboard(ranked: list[RankedModel]) -> dict[str, Any]:
    if not ranked:
        return {"n_models": 0, "spread": 0.0, "top": None}
    scores = [r.score for r in ranked]
    return {
        "n_models": len(ranked),
        "spread": float(max(scores) - min(scores)),
        "top": ranked[0].name,
        "metrics_top": dict(ranked[0].metrics),
        "per_model": {r.name: dict(r.metrics) for r in ranked},
    }

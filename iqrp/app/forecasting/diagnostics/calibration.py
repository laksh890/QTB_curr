"""Calibration diagnostics for probabilistic forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.forecasting.base.evaluator import brier_score, expected_calibration_error, log_loss


@dataclass(slots=True)
class CalibrationReport:
    ece: float
    brier: float
    log_loss: float
    bin_accuracy: list[float]
    bin_confidence: list[float]
    bin_counts: list[int]
    bias: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ece": self.ece,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "bin_accuracy": list(self.bin_accuracy),
            "bin_confidence": list(self.bin_confidence),
            "bin_counts": list(self.bin_counts),
            "bias": self.bias,
            "metadata": dict(self.metadata),
        }


def calibration_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> CalibrationReport:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if p.ndim == 1:
        conf = p
        pred = (p >= 0.5).astype(np.int64)
        proba_2d = np.column_stack([1.0 - p, p])
    else:
        conf = p.max(axis=1)
        pred = np.argmax(p, axis=1)
        proba_2d = p
    n = min(conf.size, y.size)
    conf, pred, y = conf[:n], pred[:n], y[:n]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    accs: list[float] = []
    confs: list[float] = []
    counts: list[int] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if not np.any(mask):
            accs.append(float("nan"))
            confs.append(float("nan"))
            counts.append(0)
            continue
        accs.append(float(np.mean(pred[mask] == y[mask])))
        confs.append(float(np.mean(conf[mask])))
        counts.append(int(np.sum(mask)))
    # bias: mean predicted prob of true class - empirical frequency
    true_p = []
    for i in range(n):
        lab = int(y[i])
        if 0 <= lab < proba_2d.shape[1]:
            true_p.append(float(proba_2d[i, lab]))
    emp = float(np.mean(pred == y)) if n else 0.0
    bias = float(np.mean(true_p) - emp) if true_p else 0.0
    return CalibrationReport(
        ece=expected_calibration_error(proba_2d, y, n_bins=n_bins),
        brier=brier_score(proba_2d, y),
        log_loss=log_loss(proba_2d, y),
        bin_accuracy=accs,
        bin_confidence=confs,
        bin_counts=counts,
        bias=bias,
        metadata={"n_bins": n_bins, "n": n},
    )


def detect_bias(y_true: np.ndarray, y_pred: np.ndarray, *, threshold: float = 0.05) -> dict[str, Any]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = min(yt.size, yp.size)
    if n == 0:
        return {"bias": 0.0, "biased": False, "threshold": threshold}
    bias = float(np.mean(yp[:n] - yt[:n]))
    scale = max(float(np.std(yt[:n])), 1e-12)
    return {
        "bias": bias,
        "normalized_bias": bias / scale,
        "biased": abs(bias / scale) >= threshold,
        "threshold": threshold,
    }

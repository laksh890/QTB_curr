"""Production monitoring for forecast intelligence."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from iqrp.app.forecasting.intelligence.config import MonitoringConfig
from iqrp.app.forecasting.intelligence.ranking import compute_metrics


@dataclass
class MonitorSnapshot:
    n_observations: int
    metrics: dict[str, float]
    latency_ms_p50: float
    latency_ms_p95: float
    throughput: float
    calibration_error: float
    feature_stability: float
    prediction_stability: float
    alerts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_observations": self.n_observations,
            "metrics": dict(self.metrics),
            "latency_ms_p50": self.latency_ms_p50,
            "latency_ms_p95": self.latency_ms_p95,
            "throughput": self.throughput,
            "calibration_error": self.calibration_error,
            "feature_stability": self.feature_stability,
            "prediction_stability": self.prediction_stability,
            "alerts": list(self.alerts),
        }


class ForecastMonitor:
    def __init__(self, config: MonitoringConfig | None = None) -> None:
        self.config = config or MonitoringConfig()
        self._y_true: deque[float] = deque(maxlen=max(self.config.window, 32))
        self._y_pred: deque[float] = deque(maxlen=max(self.config.window, 32))
        self._latencies: deque[float] = deque(maxlen=max(self.config.window, 32))
        self._pred_hist: deque[float] = deque(maxlen=max(self.config.window, 32))
        self._feat_hist: deque[np.ndarray] = deque(maxlen=8)
        self._n = 0
        self._t0 = perf_counter()

    def record(
        self,
        *,
        y_true: float | None = None,
        y_pred: float | None = None,
        latency_ms: float | None = None,
        features: np.ndarray | None = None,
    ) -> None:
        self._n += 1
        if y_true is not None:
            self._y_true.append(float(y_true))
        if y_pred is not None:
            self._y_pred.append(float(y_pred))
            self._pred_hist.append(float(y_pred))
        if latency_ms is not None:
            self._latencies.append(float(latency_ms))
        if features is not None:
            self._feat_hist.append(np.asarray(features, dtype=np.float64).reshape(-1))

    def snapshot(self) -> MonitorSnapshot:
        metrics: dict[str, float] = {}
        if len(self._y_true) >= 2 and len(self._y_pred) >= 2:
            n = min(len(self._y_true), len(self._y_pred))
            yt = np.asarray(list(self._y_true)[-n:], dtype=np.float64)
            yp = np.asarray(list(self._y_pred)[-n:], dtype=np.float64)
            metrics = compute_metrics(yt, yp)
        lat = (
            np.asarray(list(self._latencies), dtype=np.float64)
            if self._latencies
            else np.asarray([0.0])
        )
        elapsed = max(perf_counter() - self._t0, 1e-6)
        throughput = self._n / elapsed
        cal_err = float(metrics.get("calibration_error", metrics.get("brier", 0.0)))
        pred_stab = 1.0
        if len(self._pred_hist) >= 4:
            p = np.asarray(list(self._pred_hist), dtype=np.float64)
            pred_stab = float(1.0 / (1.0 + np.std(np.diff(p))))
        feat_stab = 1.0
        if len(self._feat_hist) >= 2:
            a, b = self._feat_hist[0], self._feat_hist[-1]
            m = min(a.size, b.size)
            if m:
                feat_stab = float(1.0 / (1.0 + np.mean(np.abs(a[:m] - b[:m]))))
        alerts: list[str] = []
        if metrics.get("mae", 0.0) > self.config.mae_alert:
            alerts.append("mae_high")
        if float(np.percentile(lat, 95)) > self.config.latency_ms_alert:
            alerts.append("latency_high")
        if cal_err > self.config.calibration_alert:
            alerts.append("calibration_poor")
        if pred_stab < self.config.stability_alert:
            alerts.append("prediction_unstable")
        return MonitorSnapshot(
            n_observations=self._n,
            metrics=metrics,
            latency_ms_p50=float(np.percentile(lat, 50)),
            latency_ms_p95=float(np.percentile(lat, 95)),
            throughput=throughput,
            calibration_error=cal_err,
            feature_stability=feat_stab,
            prediction_stability=pred_stab,
            alerts=alerts,
        )

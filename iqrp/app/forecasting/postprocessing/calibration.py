"""Probability calibration for probabilistic forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from iqrp.app.math.utils.numerical_stability import stable_softmax

CalibrationMethod = Literal["platt", "isotonic", "temperature", "none"]


@dataclass
class ProbabilityCalibrator:
    method: CalibrationMethod = "none"
    temperature: float = 1.0
    platt_a: float = 1.0
    platt_b: float = 0.0
    isotonic_x: np.ndarray | None = None
    isotonic_y: np.ndarray | None = None
    fitted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> ProbabilityCalibrator:
        p = np.asarray(probabilities, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        if p.ndim == 1:
            p = np.column_stack([1.0 - p, p])
        n = min(p.shape[0], y.size)
        if self.method == "none" or n < 5:
            self.fitted = True
            return self
        conf = p[:n].max(axis=1)
        correct = (np.argmax(p[:n], axis=1) == y[:n]).astype(np.float64)
        if self.method == "temperature":
            self.temperature = _fit_temperature(p[:n], y[:n])
        elif self.method == "platt":
            self.platt_a, self.platt_b = _fit_platt(conf, correct)
        elif self.method == "isotonic":
            order = np.argsort(conf)
            self.isotonic_x = conf[order]
            self.isotonic_y = _isotonic(correct[order])
        self.fitted = True
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.asarray(probabilities, dtype=np.float64)
        if p.ndim == 1:
            p = np.column_stack([1.0 - p, p])
        if not self.fitted or self.method == "none":
            return _row_norm(p)
        if self.method == "temperature":
            logits = np.log(np.clip(p, 1e-300, None))
            return stable_softmax(logits / max(self.temperature, 1e-6), axis=1)
        if self.method == "platt":
            conf = p.max(axis=1)
            z = 1.0 / (1.0 + np.exp(-(self.platt_a * conf + self.platt_b)))
            out = p.copy()
            pred = np.argmax(p, axis=1)
            for i in range(out.shape[0]):
                out[i, :] = (1.0 - z[i]) / max(out.shape[1] - 1, 1)
                out[i, pred[i]] = z[i]
            return _row_norm(out)
        # isotonic on confidence
        conf = p.max(axis=1)
        z = _isotonic_predict(conf, self.isotonic_x, self.isotonic_y)
        out = p.copy()
        pred = np.argmax(p, axis=1)
        for i in range(out.shape[0]):
            out[i, :] = (1.0 - z[i]) / max(out.shape[1] - 1, 1)
            out[i, pred[i]] = z[i]
        return _row_norm(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "temperature": self.temperature,
            "platt_a": self.platt_a,
            "platt_b": self.platt_b,
            "isotonic_x": None if self.isotonic_x is None else self.isotonic_x.tolist(),
            "isotonic_y": None if self.isotonic_y is None else self.isotonic_y.tolist(),
            "fitted": self.fitted,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProbabilityCalibrator:
        return cls(
            method=data.get("method", "none"),  # type: ignore[arg-type]
            temperature=float(data.get("temperature", 1.0)),
            platt_a=float(data.get("platt_a", 1.0)),
            platt_b=float(data.get("platt_b", 0.0)),
            isotonic_x=(
                None
                if data.get("isotonic_x") is None
                else np.asarray(data["isotonic_x"], dtype=np.float64)
            ),
            isotonic_y=(
                None
                if data.get("isotonic_y") is None
                else np.asarray(data["isotonic_y"], dtype=np.float64)
            ),
            fitted=bool(data.get("fitted", False)),
            metadata=dict(data.get("metadata") or {}),
        )


def _row_norm(p: np.ndarray) -> np.ndarray:
    s = np.clip(p.sum(axis=1, keepdims=True), 1e-300, None)
    return p / s


def _fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.optimize import minimize_scalar
    except Exception:  # noqa: BLE001
        return 1.0

    def nll(t: float) -> float:
        t = max(float(t), 1e-3)
        logits = np.log(np.clip(p, 1e-300, None)) / t
        probs = stable_softmax(logits, axis=1)
        ll = 0.0
        for i, lab in enumerate(y):
            if 0 <= int(lab) < probs.shape[1]:
                ll -= float(np.log(np.clip(probs[i, int(lab)], 1e-300, None)))
        return ll

    try:
        res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    except Exception:  # noqa: BLE001
        return 1.0
    return float(res.x) if getattr(res, "success", False) else 1.0


def _fit_platt(conf: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
    try:
        from scipy.optimize import minimize
    except Exception:  # noqa: BLE001
        return 1.0, 0.0

    def loss(theta: np.ndarray) -> float:
        a, b = float(theta[0]), float(theta[1])
        z = 1.0 / (1.0 + np.exp(-(a * conf + b)))
        z = np.clip(z, 1e-12, 1 - 1e-12)
        return float(-np.sum(correct * np.log(z) + (1 - correct) * np.log(1 - z)))

    res = minimize(loss, x0=np.array([1.0, 0.0]), method="L-BFGS-B")
    if not res.success:
        return 1.0, 0.0
    return float(res.x[0]), float(res.x[1])


def _isotonic(y: np.ndarray) -> np.ndarray:
    """PAVA isotonic regression."""
    y = np.asarray(y, dtype=np.float64).copy()
    n = y.size
    if n == 0:
        return y
    level = y.copy()
    weight = np.ones(n)
    i = 0
    while i < n - 1:
        if level[i] > level[i + 1]:
            total_w = weight[i] + weight[i + 1]
            avg = (level[i] * weight[i] + level[i + 1] * weight[i + 1]) / total_w
            level[i] = level[i + 1] = avg
            weight[i] = weight[i + 1] = total_w
            j = i
            while j > 0 and level[j - 1] > level[j]:
                total_w = weight[j - 1] + weight[j]
                avg = (level[j - 1] * weight[j - 1] + level[j] * weight[j]) / total_w
                level[j - 1] = level[j] = avg
                weight[j - 1] = weight[j] = total_w
                j -= 1
            i = j
        else:
            i += 1
    return level


def _isotonic_predict(
    conf: np.ndarray, x: np.ndarray | None, y: np.ndarray | None
) -> np.ndarray:
    if x is None or y is None or x.size == 0:
        return np.clip(conf, 0.0, 1.0)
    return np.interp(conf, x, y, left=float(y[0]), right=float(y[-1]))

"""Probability calibration for ensemble outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy import optimize  # type: ignore[import-untyped]
from scipy.special import softmax  # type: ignore[import-untyped]

from iqrp.app.math.utils.numerical_stability import stable_softmax

CalibrationMethod = Literal["platt", "isotonic", "temperature", "dirichlet", "none"]


@dataclass
class Calibrator:
    method: CalibrationMethod = "temperature"
    temperature: float = 1.0
    platt_a: float = 1.0
    platt_b: float = 0.0
    isotonic_x: np.ndarray | None = None
    isotonic_y: np.ndarray | None = None
    dirichlet_matrix: np.ndarray | None = None
    fitted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> Calibrator:
        p = np.asarray(probabilities, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        n, k = p.shape
        if self.method == "none" or n < 5:
            self.fitted = True
            return self
        if self.method == "temperature":
            self.temperature = _fit_temperature(p, y)
        elif self.method == "platt":
            conf = p.max(axis=1)
            correct = (np.argmax(p, axis=1) == y).astype(np.float64)
            self.platt_a, self.platt_b = _fit_platt(conf, correct)
        elif self.method == "isotonic":
            conf = p.max(axis=1)
            correct = (np.argmax(p, axis=1) == y).astype(np.float64)
            order = np.argsort(conf)
            self.isotonic_x = conf[order]
            self.isotonic_y = _isotonic_regression(correct[order])
        elif self.method == "dirichlet":
            self.dirichlet_matrix = _fit_dirichlet(p, y, k)
        self.fitted = True
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.asarray(probabilities, dtype=np.float64)
        if p.ndim == 1:
            p = p.reshape(-1, 1)
        if not self.fitted or self.method == "none":
            return _row_norm(p)
        if self.method == "temperature":
            logits = np.log(np.clip(p, 1e-300, None))
            return stable_softmax(logits / max(self.temperature, 1e-6), axis=1)
        if self.method == "platt":
            conf = p.max(axis=1)
            calib = 1.0 / (1.0 + np.exp(self.platt_a * conf + self.platt_b))
            # scale rows so max mass tracks calibrated confidence
            out = p.copy()
            for i in range(out.shape[0]):
                j = int(np.argmax(out[i]))
                remain = max(1.0 - calib[i], 0.0)
                out[i] = remain * out[i]
                out[i, j] = calib[i]
            return _row_norm(out)
        if (
            self.method == "isotonic"
            and self.isotonic_x is not None
            and self.isotonic_y is not None
        ):
            conf = p.max(axis=1)
            calib = np.interp(conf, self.isotonic_x, self.isotonic_y)
            out = p.copy()
            for i in range(out.shape[0]):
                j = int(np.argmax(out[i]))
                remain = max(1.0 - calib[i], 0.0)
                out[i] = remain * out[i]
                out[i, j] = calib[i]
            return _row_norm(out)
        if self.method == "dirichlet" and self.dirichlet_matrix is not None:
            logits = np.log(np.clip(p, 1e-300, None))
            return stable_softmax(logits @ self.dirichlet_matrix.T, axis=1)
        return _row_norm(p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "temperature": float(self.temperature),
            "platt_a": float(self.platt_a),
            "platt_b": float(self.platt_b),
            "isotonic_x": None if self.isotonic_x is None else self.isotonic_x.tolist(),
            "isotonic_y": None if self.isotonic_y is None else self.isotonic_y.tolist(),
            "dirichlet_matrix": (
                None if self.dirichlet_matrix is None else self.dirichlet_matrix.tolist()
            ),
            "fitted": self.fitted,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Calibrator:
        return cls(
            method=data.get("method", "temperature"),  # type: ignore[arg-type]
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
            dirichlet_matrix=(
                None
                if data.get("dirichlet_matrix") is None
                else np.asarray(data["dirichlet_matrix"], dtype=np.float64)
            ),
            fitted=bool(data.get("fitted", False)),
            metadata=dict(data.get("metadata") or {}),
        )


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 10
) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    conf = p.max(axis=1)
    pred = np.argmax(p, axis=1)
    correct = (pred == y).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = conf.size
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if not np.any(mask):
            continue
        ece += (np.sum(mask) / n) * abs(float(np.mean(correct[mask]) - np.mean(conf[mask])))
    return float(ece)


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if p.ndim == 1:
        p = p.reshape(-1, 1)
    k = p.shape[1]
    onehot = np.zeros_like(p)
    for i, lab in enumerate(y[: p.shape[0]]):
        if 0 <= int(lab) < k:
            onehot[i, int(lab)] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def _row_norm(p: np.ndarray) -> np.ndarray:
    s = np.clip(p.sum(axis=1, keepdims=True), 1e-300, None)
    return p / s


def _fit_temperature(p: np.ndarray, y: np.ndarray) -> float:
    logits = np.log(np.clip(p, 1e-300, None))

    def nll(t: np.ndarray) -> float:
        temp = max(float(t[0]), 1e-3)
        probs = softmax(logits / temp, axis=1)
        return float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], 1e-300, None))))

    res = optimize.minimize(nll, x0=np.array([1.0]), bounds=[(0.05, 10.0)])
    return float(res.x[0]) if res.success else 1.0


def _fit_platt(conf: np.ndarray, correct: np.ndarray) -> tuple[float, float]:
    def loss(ab: np.ndarray) -> float:
        a, b = float(ab[0]), float(ab[1])
        pred = 1.0 / (1.0 + np.exp(a * conf + b))
        pred = np.clip(pred, 1e-6, 1 - 1e-6)
        return float(-np.mean(correct * np.log(pred) + (1 - correct) * np.log(1 - pred)))

    res = optimize.minimize(loss, x0=np.array([1.0, 0.0]))
    if res.success:
        return float(res.x[0]), float(res.x[1])
    return 1.0, 0.0


def _isotonic_regression(y: np.ndarray) -> np.ndarray:
    """PAVA isotonic regression (non-decreasing)."""
    y = np.asarray(y, dtype=np.float64).copy()
    n = y.size
    if n == 0:
        return y
    level = y.copy()
    weight = np.ones(n)
    i = 0
    while i < n - 1:
        if level[i] > level[i + 1]:
            # merge pool
            total_w = weight[i] + weight[i + 1]
            avg = (level[i] * weight[i] + level[i + 1] * weight[i + 1]) / total_w
            level[i] = level[i + 1] = avg
            weight[i] = weight[i + 1] = total_w
            # backtrack
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


def _fit_dirichlet(p: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Simple diagonal Dirichlet-like matrix: scale logits by class reliability."""
    mat = np.eye(k)
    for c in range(k):
        mask = y == c
        if np.any(mask):
            acc = float(np.mean(np.argmax(p[mask], axis=1) == c))
            mat[c, c] = max(acc, 0.1)
    return mat

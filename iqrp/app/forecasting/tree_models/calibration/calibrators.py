"""Probability calibration: Platt, Isotonic, Temperature scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass
class Calibrator:
    method: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "params": dict(self.params)}


def fit_calibrator(
    y: np.ndarray,
    proba: np.ndarray,
    *,
    method: Literal["none", "platt", "isotonic", "temperature"] = "platt",
) -> Calibrator | None:
    if method in {"none", None}:
        return None
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    P = np.asarray(proba, dtype=np.float64)
    if P.ndim == 1:
        p = P
    else:
        p = P[:, -1] if P.shape[1] >= 2 else P[:, 0]
    # binary labels
    classes = np.unique(y)
    if classes.size == 2:
        yb = (y == classes.max()).astype(np.float64)
    else:
        yb = (y > np.median(y)).astype(np.float64)
    if method == "platt":
        return _fit_platt(yb, p)
    if method == "isotonic":
        return _fit_isotonic(yb, p)
    if method == "temperature":
        return _fit_temperature(yb, p)
    return None


def apply_calibration(cal: Calibrator | None, proba: np.ndarray) -> np.ndarray:
    if cal is None:
        return proba
    P = np.asarray(proba, dtype=np.float64)
    if P.ndim == 1:
        p = P.copy()
        out_1d = True
    else:
        p = P[:, -1].copy() if P.shape[1] >= 2 else P[:, 0].copy()
        out_1d = False
    if cal.method == "platt":
        a, b = float(cal.params["a"]), float(cal.params["b"])
        # logistic on logit
        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        logit = np.log(p / (1 - p))
        p = 1 / (1 + np.exp(-(a * logit + b)))
    elif cal.method == "isotonic":
        x = np.asarray(cal.params["x"], dtype=np.float64)
        y = np.asarray(cal.params["y"], dtype=np.float64)
        p = np.interp(p, x, y)
    elif cal.method == "temperature":
        t = max(float(cal.params["temperature"]), 1e-3)
        eps = 1e-6
        p = np.clip(p, eps, 1 - eps)
        logit = np.log(p / (1 - p)) / t
        p = 1 / (1 + np.exp(-logit))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if out_1d:
        return p
    out = P.copy()
    if out.shape[1] >= 2:
        out[:, -1] = p
        out[:, 0] = 1 - p
        if out.shape[1] > 2:
            # renormalize remaining
            rest = out[:, 1:-1]
            if rest.size:
                s = rest.sum(axis=1, keepdims=True)
                rest = rest / np.maximum(s, 1e-12) * (1 - out[:, 0:1] - out[:, -1:])
                out[:, 1:-1] = rest
    else:
        out[:, 0] = p
    return out


def _fit_platt(y: np.ndarray, p: np.ndarray) -> Calibrator:
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    # fit logistic: y ~ 1/(1+exp(-(a*logit+b)))
    X = np.column_stack([logit, np.ones(len(logit))])

    def nll(theta: np.ndarray) -> float:
        z = X @ theta
        pr = 1 / (1 + np.exp(-z))
        pr = np.clip(pr, eps, 1 - eps)
        return float(-np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr)))

    from scipy.optimize import minimize

    res = minimize(nll, np.array([1.0, 0.0]), method="L-BFGS-B")
    a, b = map(float, res.x)
    return Calibrator(method="platt", params={"a": a, "b": b})


def _fit_isotonic(y: np.ndarray, p: np.ndarray) -> Calibrator:
    order = np.argsort(p)
    p_s, y_s = p[order], y[order]
    # PAV algorithm
    y_hat = y_s.astype(np.float64).copy()
    n = y_hat.size
    # merge violators
    w = np.ones(n)
    i = 0
    list(range(n))
    while i < len(y_hat) - 1:
        if y_hat[i] > y_hat[i + 1]:
            # merge
            total_w = w[i] + w[i + 1]
            avg = (y_hat[i] * w[i] + y_hat[i + 1] * w[i + 1]) / total_w
            y_hat[i] = avg
            w[i] = total_w
            y_hat = np.delete(y_hat, i + 1)
            w = np.delete(w, i + 1)
            p_s = np.delete(p_s, i + 1)
            i = max(i - 1, 0)
        else:
            i += 1
    # expand back via interpolation points
    return Calibrator(
        method="isotonic",
        params={"x": p_s.tolist(), "y": y_hat.tolist()},
    )


def _fit_temperature(y: np.ndarray, p: np.ndarray) -> Calibrator:
    eps = 1e-6
    p = np.clip(p, eps, 1 - eps)
    logit = np.log(p / (1 - p))

    def nll(t: float) -> float:
        t = max(t, 1e-3)
        pr = 1 / (1 + np.exp(-logit / t))
        pr = np.clip(pr, eps, 1 - eps)
        return float(-np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr)))

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return Calibrator(method="temperature", params={"temperature": float(res.x)})

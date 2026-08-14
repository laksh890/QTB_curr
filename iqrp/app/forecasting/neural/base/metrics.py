"""Training / evaluation metrics for neural forecasting."""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _align(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _align(y_true, y_pred)
    return float(np.mean((a - b) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _align(y_true, y_pred)
    return float(np.mean(np.abs((a - b) / np.maximum(np.abs(a), 1e-8))))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _align(y_true, y_pred)
    return float(np.mean(2 * np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-8)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _align(y_true, y_pred)
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = _align(y_true, y_pred)
    if a.size < 2:
        return float("nan")
    return float(np.mean(np.sign(np.diff(a.reshape(-1))) == np.sign(np.diff(b.reshape(-1)))))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    proba: np.ndarray | None = None,
    task: str = "regression",
) -> dict[str, float]:
    out: dict[str, float] = {
        "mae": mae(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "n": float(min(np.asarray(y_true).size, np.asarray(y_pred).size)),
    }
    if proba is not None and task in {"binary", "classification", "probability", "multiclass"}:
        P = np.asarray(proba, dtype=np.float64)
        scores = P[:, -1] if P.ndim == 2 and P.shape[1] >= 2 else P.reshape(-1)
        yt = np.asarray(y_true, dtype=np.float64).reshape(-1)[: scores.size]
        classes = np.unique(yt)
        yb = (
            (yt == classes.max()).astype(np.float64)
            if classes.size >= 2
            else (yt > 0).astype(np.float64)
        )
        out["brier"] = float(np.mean((scores[: yb.size] - yb) ** 2))
        p = np.clip(scores[: yb.size], 1e-6, 1 - 1e-6)
        out["log_loss"] = float(-np.mean(yb * np.log(p) + (1 - yb) * np.log(1 - p)))
    return out


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(x.size, y.size)
    return x[:n], y[:n]

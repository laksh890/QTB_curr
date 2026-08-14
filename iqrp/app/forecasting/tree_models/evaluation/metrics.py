"""Tree forecasting evaluation metrics including trading diagnostics."""

from __future__ import annotations

import numpy as np


def evaluate_tree_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    proba: np.ndarray | None = None,
    task: str = "regression",
) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = min(yt.size, yp.size)
    yt, yp = yt[:n], yp[:n]
    out: dict[str, float] = {
        "mae": _mae(yt, yp),
        "rmse": _rmse(yt, yp),
        "mape": _mape(yt, yp),
        "smape": _smape(yt, yp),
        "r2": _r2(yt, yp),
        "directional_accuracy": _directional_accuracy(yt, yp),
        "n": float(n),
    }
    # trading-style metrics on prediction as signal
    rets = yt  # treat target as returns when appropriate
    signal = np.sign(yp)
    pnl = signal * rets
    out["sharpe_ratio"] = _sharpe(pnl)
    out["profit_factor"] = _profit_factor(pnl)
    out["max_drawdown"] = _max_drawdown(np.cumsum(pnl))
    if proba is not None and task in {"binary", "multiclass", "probability"}:
        P = np.asarray(proba, dtype=np.float64)
        scores = P[:, -1] if P.ndim == 2 and P.shape[1] >= 2 else P.reshape(-1)
        classes = np.unique(yt)
        yb = (
            (yt == classes.max()).astype(np.float64)
            if classes.size >= 2
            else (yt > 0).astype(np.float64)
        )
        out["roc_auc"] = _roc_auc(yb, scores[:n])
        out["pr_auc"] = _pr_auc(yb, scores[:n])
        out["brier_score"] = float(np.mean((scores[:n] - yb) ** 2))
        out["log_loss"] = _log_loss(yb, scores[:n])
    return out


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _mape(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(np.abs(a), 1e-8)
    return float(np.mean(np.abs((a - b) / denom)))


def _smape(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-8)
    return float(np.mean(2 * np.abs(a - b) / denom))


def _r2(a: np.ndarray, b: np.ndarray) -> float:
    ss_res = float(np.sum((a - b) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def _directional_accuracy(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float("nan")
    return float(np.mean(np.sign(a[1:] - a[:-1]) == np.sign(b[1:] - b[:-1])))


def _sharpe(pnl: np.ndarray) -> float:
    if pnl.size < 2:
        return 0.0
    mu, sig = float(np.mean(pnl)), float(np.std(pnl))
    return float(np.sqrt(252) * mu / sig) if sig > 1e-12 else 0.0


def _profit_factor(pnl: np.ndarray) -> float:
    gains = float(np.sum(pnl[pnl > 0]))
    losses = float(-np.sum(pnl[pnl < 0]))
    return gains / losses if losses > 1e-12 else float("inf") if gains > 0 else 0.0


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(np.max(dd))


def _roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    y = y[order]
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.arange(1, y.size + 1)
    sum_ranks = float(np.sum(ranks[y == 1]))
    return float((sum_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(float(np.sum(y)), 1e-12)
    # trapz
    trap = getattr(np, "trapezoid", None) or np.trapz
    return (
        float(trap(precision, recall))
        if recall.size > 1
        else float(precision[-1] if precision.size else 0.0)
    )


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

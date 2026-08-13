"""Forecast evaluation metrics and validation protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

ValidationMethod = Literal[
    "holdout",
    "cross_validation",
    "walk_forward",
    "rolling",
    "time_series_split",
]


def _as_1d(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(-1)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    n = min(yt.size, yp.size)
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(yt[:n] - yp[:n])))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    n = min(yt.size, yp.size)
    if n == 0:
        return float("nan")
    return float(np.mean((yt[:n] - yp[:n]) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-8) -> float:
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    n = min(yt.size, yp.size)
    if n == 0:
        return float("nan")
    denom = np.clip(np.abs(yt[:n]), eps, None)
    return float(np.mean(np.abs((yt[:n] - yp[:n]) / denom)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-8) -> float:
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    n = min(yt.size, yp.size)
    if n == 0:
        return float("nan")
    denom = np.clip(np.abs(yt[:n]) + np.abs(yp[:n]), eps, None)
    return float(np.mean(2.0 * np.abs(yt[:n] - yp[:n]) / denom) * 100.0)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    n = min(yt.size, yp.size)
    if n == 0:
        return float("nan")
    yt, yp = yt[:n], yp[:n]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot <= 1e-300:
        return 0.0 if ss_res <= 1e-300 else -float("inf")
    return 1.0 - ss_res / ss_tot


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    n = min(yt.size, yp.size)
    if n == 0:
        return float("nan")
    return float(np.mean(yt[:n] == yp[:n]))


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, *, average: str = "macro"
) -> dict[str, float]:
    yt = np.asarray(y_true).reshape(-1).astype(np.int64)
    yp = np.asarray(y_pred).reshape(-1).astype(np.int64)
    n = min(yt.size, yp.size)
    yt, yp = yt[:n], yp[:n]
    labels = np.unique(np.concatenate([yt, yp])) if n else np.array([], dtype=np.int64)
    if labels.size == 0:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    precs, recs, f1s = [], [], []
    for lab in labels:
        tp = float(np.sum((yp == lab) & (yt == lab)))
        fp = float(np.sum((yp == lab) & (yt != lab)))
        fn = float(np.sum((yp != lab) & (yt == lab)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precs.append(p)
        recs.append(r)
        f1s.append(f1)
    if average == "micro":
        tp = float(np.sum(yp == yt))
        total = float(n)
        acc = tp / total if total else 0.0
        return {"precision": acc, "recall": acc, "f1": acc}
    return {
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
        "f1": float(np.mean(f1s)),
    }


def roc_auc_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    yt = np.asarray(y_true).reshape(-1).astype(np.int64)
    sc = _as_1d(scores)
    n = min(yt.size, sc.size)
    yt, sc = yt[:n], sc[:n]
    pos = sc[yt == 1]
    neg = sc[yt == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Mann–Whitney U / AUC
    correct = 0.0
    for p in pos:
        correct += float(np.sum(p > neg) + 0.5 * np.sum(p == neg))
    return float(correct / (pos.size * neg.size))


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if p.ndim == 1:
        p = np.column_stack([1.0 - p, p])
    n, k = p.shape
    n = min(n, y.size)
    onehot = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        lab = int(y[i])
        if 0 <= lab < k:
            onehot[i, lab] = 1.0
    return float(np.mean(np.sum((p[:n] - onehot) ** 2, axis=1)))


def log_loss(probabilities: np.ndarray, labels: np.ndarray, *, eps: float = 1e-15) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if p.ndim == 1:
        p = np.column_stack([1.0 - p, p])
    n = min(p.shape[0], y.size)
    p = np.clip(p[:n], eps, 1.0 - eps)
    p = p / p.sum(axis=1, keepdims=True)
    ll = 0.0
    for i in range(n):
        lab = int(y[i])
        if 0 <= lab < p.shape[1]:
            ll -= float(np.log(p[i, lab]))
    return ll / max(n, 1)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, *, n_bins: int = 10
) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if p.ndim == 1:
        conf = p
        pred = (p >= 0.5).astype(np.int64)
    else:
        conf = p.max(axis=1)
        pred = np.argmax(p, axis=1)
    n = min(conf.size, y.size)
    conf, pred, y = conf[:n], pred[:n], y[:n]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if not np.any(mask):
            continue
        ece += (np.sum(mask) / n) * abs(float(np.mean((pred[mask] == y[mask])) - np.mean(conf[mask])))
    return float(ece)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    n = min(yt.size, yp.size)
    if n < 2:
        return float("nan")
    d_true = np.sign(np.diff(yt[:n]))
    d_pred = np.sign(np.diff(yp[:n]))
    return float(np.mean(d_true == d_pred))


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray, *, tol: float = 0.0) -> float:
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    n = min(yt.size, yp.size)
    if n == 0:
        return float("nan")
    return float(np.mean(np.abs(yt[:n] - yp[:n]) <= tol))


def _returns_from_signals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt, yp = _as_1d(y_true), _as_1d(y_pred)
    n = min(yt.size, yp.size)
    if n < 2:
        return np.array([], dtype=np.float64)
    # treat prediction direction as position; realize true returns
    ret = np.diff(yt[:n])
    pos = np.sign(np.diff(yp[:n]))
    return pos * ret


def profit_factor(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pnl = _returns_from_signals(y_true, y_pred)
    if pnl.size == 0:
        return float("nan")
    gains = float(np.sum(pnl[pnl > 0]))
    losses = float(-np.sum(pnl[pnl < 0]))
    if losses <= 1e-300:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-12) -> float:
    pnl = _returns_from_signals(y_true, y_pred)
    if pnl.size < 2:
        return float("nan")
    mu = float(np.mean(pnl))
    sd = float(np.std(pnl, ddof=1))
    return mu / max(sd, eps) * np.sqrt(pnl.size)


def sortino_ratio(y_true: np.ndarray, y_pred: np.ndarray, *, eps: float = 1e-12) -> float:
    pnl = _returns_from_signals(y_true, y_pred)
    if pnl.size < 2:
        return float("nan")
    downside = pnl[pnl < 0]
    dd = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    mu = float(np.mean(pnl))
    return mu / max(dd, eps) * np.sqrt(pnl.size)


def max_drawdown(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pnl = _returns_from_signals(y_true, y_pred)
    if pnl.size == 0:
        return float("nan")
    equity = np.cumsum(pnl)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(np.max(dd)) if dd.size else 0.0


@dataclass(slots=True)
class EvaluationReport:
    metrics: dict[str, float]
    method: str = "holdout"
    n_samples: int = 0
    folds: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "method": self.method,
            "n_samples": self.n_samples,
            "folds": list(self.folds),
            "metadata": dict(self.metadata),
        }


class ForecastEvaluator:
    """Compute regression / classification / probability / financial metrics."""

    def evaluate_regression(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        return {
            "mae": mae(y_true, y_pred),
            "mse": mse(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "mape": mape(y_true, y_pred),
            "smape": smape(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "directional_accuracy": directional_accuracy(y_true, y_pred),
            "hit_rate": hit_rate(y_true, y_pred),
            "profit_factor": profit_factor(y_true, y_pred),
            "sharpe": sharpe_ratio(y_true, y_pred),
            "sortino": sortino_ratio(y_true, y_pred),
            "max_drawdown": max_drawdown(y_true, y_pred),
        }

    def evaluate_classification(
        self, y_true: np.ndarray, y_pred: np.ndarray, *, scores: np.ndarray | None = None
    ) -> dict[str, float]:
        prf = precision_recall_f1(y_true, y_pred)
        out = {
            "accuracy": accuracy(y_true, y_pred),
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
        }
        if scores is not None:
            out["roc_auc"] = roc_auc_binary(y_true, scores)
        return out

    def evaluate_probability(
        self, probabilities: np.ndarray, labels: np.ndarray
    ) -> dict[str, float]:
        return {
            "brier": brier_score(probabilities, labels),
            "log_loss": log_loss(probabilities, labels),
            "calibration_error": expected_calibration_error(probabilities, labels),
        }

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        task: str = "regression",
        probabilities: np.ndarray | None = None,
        scores: np.ndarray | None = None,
    ) -> EvaluationReport:
        yt = np.asarray(y_true)
        yp = np.asarray(y_pred)
        n = min(yt.reshape(-1).size, yp.reshape(-1).size)
        if task == "classification":
            metrics = self.evaluate_classification(yt, yp, scores=scores)
        else:
            metrics = self.evaluate_regression(yt, yp)
        if probabilities is not None:
            metrics.update(self.evaluate_probability(probabilities, yt))
        return EvaluationReport(metrics=metrics, method="holdout", n_samples=n)

    def time_series_splits(
        self, n: int, *, n_splits: int = 5, min_train: int = 10
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if n < min_train + 1 or n_splits < 1:
            return []
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        test_size = max(1, (n - min_train) // n_splits)
        for i in range(n_splits):
            train_end = min_train + i * test_size
            test_end = min(train_end + test_size, n)
            if train_end >= n or train_end >= test_end:
                break
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)
            folds.append((train_idx, test_idx))
        return folds

    def walk_forward_splits(
        self, n: int, *, train_size: int, test_size: int = 1, step: int = 1
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        start = 0
        while start + train_size + test_size <= n:
            train_idx = np.arange(start, start + train_size)
            test_idx = np.arange(start + train_size, start + train_size + test_size)
            folds.append((train_idx, test_idx))
            start += step
        return folds

    def rolling_splits(
        self, n: int, *, window: int, test_size: int = 1, step: int = 1
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        return self.walk_forward_splits(n, train_size=window, test_size=test_size, step=step)

    def cross_validate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        method: ValidationMethod = "time_series_split",
        n_splits: int = 5,
        train_size: int | None = None,
        test_size: int = 1,
        window: int | None = None,
        step: int = 1,
    ) -> EvaluationReport:
        yt, yp = _as_1d(y_true), _as_1d(y_pred)
        n = min(yt.size, yp.size)
        if method == "walk_forward":
            ts = int(train_size or max(n // 2, 1))
            folds_idx = self.walk_forward_splits(n, train_size=ts, test_size=test_size, step=step)
        elif method == "rolling":
            w = int(window or train_size or max(n // 2, 1))
            folds_idx = self.rolling_splits(n, window=w, test_size=test_size, step=step)
        else:
            folds_idx = self.time_series_splits(n, n_splits=n_splits)
        fold_reports: list[dict[str, Any]] = []
        agg: dict[str, list[float]] = {}
        for i, (tr, te) in enumerate(folds_idx):
            # evaluate on test segment using aligned predictions
            m = self.evaluate_regression(yt[te], yp[te])
            fold_reports.append({"fold": i, "n_train": int(tr.size), "n_test": int(te.size), "metrics": m})
            for k, v in m.items():
                if np.isfinite(v):
                    agg.setdefault(k, []).append(float(v))
        metrics = {k: float(np.mean(vs)) for k, vs in agg.items()} if agg else {}
        return EvaluationReport(
            metrics=metrics,
            method=method,
            n_samples=n,
            folds=fold_reports,
        )

    def benchmark(
        self,
        results: dict[str, dict[str, float]],
        *,
        primary: str = "rmse",
    ) -> list[dict[str, Any]]:
        rows = []
        for name, metrics in results.items():
            rows.append({"model": name, "metrics": dict(metrics), "primary": metrics.get(primary)})
        rows.sort(key=lambda r: (float("inf") if r["primary"] is None else float(r["primary"])))
        for i, r in enumerate(rows):
            r["rank"] = i + 1
        return rows

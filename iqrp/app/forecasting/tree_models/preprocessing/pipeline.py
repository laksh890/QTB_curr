"""Preprocessing and feature selection for tree models."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


class TreePreprocessor:
    """Median impute + optional standardization (trees usually unscaled)."""

    def __init__(self, *, standardize: bool = False) -> None:
        self.standardize = standardize
        self.medians_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.scales_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> TreePreprocessor:
        X = np.asarray(X, dtype=np.float64)
        self.medians_ = np.nanmedian(X, axis=0)
        self.medians_ = np.where(np.isfinite(self.medians_), self.medians_, 0.0)
        if self.standardize:
            self.means_ = np.nanmean(X, axis=0)
            self.scales_ = np.nanstd(X, axis=0)
            self.scales_ = np.where(self.scales_ > 1e-12, self.scales_, 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64).copy()
        assert self.medians_ is not None
        inds = np.where(~np.isfinite(X))
        X[inds] = np.take(self.medians_, inds[1])
        if self.standardize and self.means_ is not None and self.scales_ is not None:
            X = (X - self.means_) / self.scales_
        return X

    def to_dict(self) -> dict[str, Any]:
        return {
            "standardize": self.standardize,
            "medians": None if self.medians_ is None else self.medians_.tolist(),
            "means": None if self.means_ is None else self.means_.tolist(),
            "scales": None if self.scales_ is None else self.scales_.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TreePreprocessor:
        obj = cls(standardize=bool(data.get("standardize", False)))
        m = data.get("medians")
        obj.medians_ = None if m is None else np.asarray(m, dtype=np.float64)
        means = data.get("means")
        obj.means_ = None if means is None else np.asarray(means, dtype=np.float64)
        scales = data.get("scales")
        obj.scales_ = None if scales is None else np.asarray(scales, dtype=np.float64)
        return obj


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    *,
    method: Literal[
        "none", "rfe", "permutation", "mutual_info", "correlation", "shap", "boruta"
    ] = "none",
    max_features: int | None = None,
    correlation_threshold: float = 0.95,
) -> list[str]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    names = list(names)
    n_feat = X.shape[1]
    k = int(max_features) if max_features is not None else n_feat
    k = max(1, min(k, n_feat))
    if method in {"none", None} or n_feat <= k and method == "none":
        if method == "correlation":
            return _correlation_filter(X, names, correlation_threshold)[:k]
        return names[:k] if max_features is not None else names
    if method == "correlation":
        return _correlation_filter(X, names, correlation_threshold)[:k]
    if method == "mutual_info":
        scores = _mutual_info(X, y)
        order = np.argsort(-scores)
        return [names[i] for i in order[:k]]
    if method in {"rfe", "permutation", "shap", "boruta"}:
        # importance proxy via absolute correlation + variance
        scores = np.abs(_corr_xy(X, y)) + 1e-3 * np.nanstd(X, axis=0)
        if method == "boruta":
            # shadow features: keep those beating median shadow score
            rng = np.random.default_rng(0)
            shadow = X.copy()
            for j in range(shadow.shape[1]):
                shadow[:, j] = rng.permutation(shadow[:, j])
            shadow_scores = np.abs(_corr_xy(shadow, y))
            thr = float(np.median(shadow_scores))
            keep = [names[i] for i in range(n_feat) if scores[i] > thr]
            if not keep:
                keep = [names[int(np.argmax(scores))]]
            return keep[:k]
        order = np.argsort(-scores)
        return [names[i] for i in order[:k]]
    return names[:k]


def _correlation_filter(X: np.ndarray, names: list[str], thr: float) -> list[str]:
    keep: list[str] = []
    kept_idx: list[int] = []
    for j, name in enumerate(names):
        ok = True
        for i in kept_idx:
            c = np.corrcoef(X[:, i], X[:, j])[0, 1]
            if np.isfinite(c) and abs(c) >= thr:
                ok = False
                break
        if ok:
            keep.append(name)
            kept_idx.append(j)
    return keep or names[:1]


def _corr_xy(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    y = y.reshape(-1)
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        a, b = X[:, j], y
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            out[j] = 0.0
        else:
            out[j] = float(np.corrcoef(a, b)[0, 1])
            if not np.isfinite(out[j]):
                out[j] = 0.0
    return out


def _mutual_info(X: np.ndarray, y: np.ndarray, bins: int = 10) -> np.ndarray:
    yb = np.digitize(y, np.quantile(y, np.linspace(0, 1, bins + 1)[1:-1]))
    scores = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        xb = np.digitize(X[:, j], np.quantile(X[:, j], np.linspace(0, 1, bins + 1)[1:-1]))
        scores[j] = _mi_discrete(xb, yb)
    return scores


def _mi_discrete(x: np.ndarray, y: np.ndarray) -> float:
    n = x.size
    px = np.bincount(x) / n
    py = np.bincount(y) / n
    joint = np.zeros((px.size, py.size))
    for a, b in zip(x, y):
        if a < px.size and b < py.size:
            joint[a, b] += 1
    joint /= n
    mi = 0.0
    for i in range(px.size):
        for j in range(py.size):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += joint[i, j] * np.log(joint[i, j] / (px[i] * py[j]))
    return float(mi)

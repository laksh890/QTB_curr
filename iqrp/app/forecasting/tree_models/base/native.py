"""Pure-numpy tree ensembles used when optional ML libraries are unavailable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass
class _Node:
    feature: int = -1
    threshold: float = 0.0
    left: int = -1
    right: int = -1
    value: float = 0.0
    is_leaf: bool = True


class _Tree:
    def __init__(self, max_depth: int = 3, min_leaf: int = 5, random_state: int = 0, extra: bool = False) -> None:
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.rng = np.random.default_rng(random_state)
        self.extra = extra
        self.nodes: list[_Node] = []
        self.feature_importances_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> _Tree:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n, p = X.shape
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        self.feature_importances_ = np.zeros(p)
        self.nodes = []
        self._build(X, y, w, np.arange(n), depth=0)
        s = float(self.feature_importances_.sum()) or 1.0
        self.feature_importances_ /= s
        return self

    def _build(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, idx: np.ndarray, depth: int) -> int:
        node_id = len(self.nodes)
        self.nodes.append(_Node())
        yw = y[idx]
        ww = w[idx]
        value = float(np.average(yw, weights=ww)) if ww.sum() > 0 else float(np.mean(yw))
        self.nodes[node_id].value = value
        if depth >= self.max_depth or idx.size <= self.min_leaf or np.unique(yw).size <= 1:
            self.nodes[node_id].is_leaf = True
            return node_id
        p = X.shape[1]
        feat_order = self.rng.permutation(p) if self.extra else np.arange(p)
        best_gain = 0.0
        best_f, best_t = -1, 0.0
        parent_ss = _weighted_sse(yw, ww)
        for f in feat_order[: max(1, int(np.sqrt(p)) + 1)]:
            col = X[idx, f]
            qs = np.unique(np.quantile(col, [0.25, 0.5, 0.75]))
            if self.extra and col.size:
                qs = np.unique(self.rng.uniform(col.min(), col.max(), size=min(3, col.size)))
            for thr in qs:
                left = idx[col <= thr]
                right = idx[col > thr]
                if left.size < self.min_leaf or right.size < self.min_leaf:
                    continue
                gain = parent_ss - (
                    _weighted_sse(y[left], w[left]) + _weighted_sse(y[right], w[right])
                )
                if gain > best_gain:
                    best_gain, best_f, best_t = gain, int(f), float(thr)
        if best_f < 0:
            self.nodes[node_id].is_leaf = True
            return node_id
        self.feature_importances_[best_f] += best_gain
        col = X[idx, best_f]
        left_idx = idx[col <= best_t]
        right_idx = idx[col > best_t]
        left_id = self._build(X, y, w, left_idx, depth + 1)
        right_id = self._build(X, y, w, right_idx, depth + 1)
        self.nodes[node_id] = _Node(
            feature=best_f,
            threshold=best_t,
            left=left_id,
            right=right_id,
            value=value,
            is_leaf=False,
        )
        return node_id

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            nid = 0
            while not self.nodes[nid].is_leaf:
                n = self.nodes[nid]
                nid = n.left if X[i, n.feature] <= n.threshold else n.right
            out[i] = self.nodes[nid].value
        return out


def _weighted_sse(y: np.ndarray, w: np.ndarray) -> float:
    if y.size == 0 or w.sum() <= 0:
        return 0.0
    mu = float(np.average(y, weights=w))
    return float(np.sum(w * (y - mu) ** 2))


class NativeForest:
    def __init__(
        self,
        *,
        n_estimators: int = 50,
        max_depth: int = 4,
        random_state: int = 0,
        task: Literal["regression", "classification"] = "regression",
        extra: bool = False,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.task = task
        self.extra = extra
        self.trees: list[_Tree] = []
        self.feature_importances_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> NativeForest:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if self.task == "classification":
            self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.random_state)
        self.trees = []
        imps = []
        n = X.shape[0]
        for i in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)
            tree = _Tree(
                max_depth=self.max_depth,
                random_state=self.random_state + i,
                extra=self.extra,
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
            imps.append(tree.feature_importances_)
        self.feature_importances_ = np.mean(np.stack(imps), axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.stack([t.predict(X) for t in self.trees], axis=0)
        if self.task == "classification":
            # majority via rounded mean for binary; nearest class for multi
            mean = preds.mean(axis=0)
            if self.classes_ is not None and self.classes_.size == 2:
                thr = float(np.mean(self.classes_))
                return np.where(mean >= thr, self.classes_[1], self.classes_[0]).astype(np.float64)
            return np.round(mean)
        return preds.mean(axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = np.stack([t.predict(X) for t in self.trees], axis=0).mean(axis=0)
        if self.classes_ is not None and self.classes_.size == 2:
            # map to [0,1]
            lo, hi = float(self.classes_[0]), float(self.classes_[1])
            p = (scores - lo) / max(hi - lo, 1e-12)
            p = np.clip(p, 0, 1)
            return np.column_stack([1 - p, p])
        # multiclass one-vs-rest soft scores
        k = int(self.classes_.size) if self.classes_ is not None else 2
        out = np.zeros((X.shape[0], k))
        pred = np.round(scores).astype(int)
        for i, c in enumerate(range(k)):
            out[:, i] = (pred == c).astype(np.float64)
        out = out + 1e-3
        out /= out.sum(axis=1, keepdims=True)
        return out


class NativeGBM:
    def __init__(
        self,
        *,
        n_estimators: int = 50,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        random_state: int = 0,
        task: Literal["regression", "classification"] = "regression",
        quantile_alpha: float | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.task = task
        self.quantile_alpha = quantile_alpha
        self.trees: list[_Tree] = []
        self.base_score: float = 0.0
        self.feature_importances_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> NativeGBM:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if self.task == "classification":
            self.classes_ = np.unique(y)
            y_fit = (y == self.classes_[-1]).astype(np.float64)
            self.base_score = float(np.clip(np.mean(y_fit), 1e-3, 1 - 1e-3))
            # work in logit space
            eps = 1e-6
            pred = np.full(y.size, np.log(self.base_score / (1 - self.base_score)))
        else:
            y_fit = y
            self.base_score = float(np.mean(y_fit))
            pred = np.full(y.size, self.base_score)
        self.trees = []
        imps = []
        for i in range(self.n_estimators):
            if self.task == "classification":
                p = 1 / (1 + np.exp(-pred))
                resid = y_fit - p
            elif self.quantile_alpha is not None:
                alpha = float(self.quantile_alpha)
                resid = np.where(y_fit > pred, alpha, alpha - 1.0)
            else:
                resid = y_fit - pred
            tree = _Tree(max_depth=self.max_depth, random_state=self.random_state + i)
            tree.fit(X, resid)
            update = tree.predict(X)
            pred = pred + self.learning_rate * update
            self.trees.append(tree)
            imps.append(tree.feature_importances_)
        self.feature_importances_ = np.mean(np.stack(imps), axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.task == "classification":
            pred = np.full(X.shape[0], np.log(self.base_score / (1 - self.base_score)))
        else:
            pred = np.full(X.shape[0], self.base_score)
        for t in self.trees:
            pred = pred + self.learning_rate * t.predict(X)
        if self.task == "classification":
            p = 1 / (1 + np.exp(-pred))
            if self.classes_ is not None and self.classes_.size >= 2:
                return np.where(p >= 0.5, self.classes_[-1], self.classes_[0]).astype(np.float64)
            return (p >= 0.5).astype(np.float64)
        return pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        pred = np.full(X.shape[0], np.log(max(self.base_score, 1e-6) / max(1 - self.base_score, 1e-6)))
        for t in self.trees:
            pred = pred + self.learning_rate * t.predict(X)
        p = 1 / (1 + np.exp(-pred))
        return np.column_stack([1 - p, p])

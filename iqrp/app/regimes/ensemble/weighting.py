"""Ensemble member weighting schemes."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

WeightMethod = Literal[
    "equal",
    "accuracy",
    "recent_accuracy",
    "log_likelihood",
    "calibration",
    "stability",
    "user",
    "rolling",
    "adaptive",
]


def normalize_weights(weights: np.ndarray, *, min_weight: float = 0.01) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.clip(w, 0.0, None)
    if float(w.sum()) <= 0:
        w = np.ones_like(w)
    # floor then renorm
    w = np.maximum(w, min_weight)
    return w / w.sum()


def equal_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / max(n, 1), dtype=np.float64)


def accuracy_weights(
    predictions: dict[str, np.ndarray],
    truth: np.ndarray,
    names: list[str],
    *,
    min_weight: float = 0.01,
) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    scores = []
    for name in names:
        pred = np.asarray(predictions[name], dtype=np.int64).reshape(-1)
        n = min(pred.size, truth.size)
        scores.append(float(np.mean(pred[:n] == truth[:n])) if n else 0.0)
    return normalize_weights(np.asarray(scores, dtype=np.float64), min_weight=min_weight)


def recent_accuracy_weights(
    predictions: dict[str, np.ndarray],
    truth: np.ndarray,
    names: list[str],
    *,
    lookback: int = 50,
    min_weight: float = 0.01,
) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.int64).reshape(-1)
    lb = max(1, int(lookback))
    scores = []
    for name in names:
        pred = np.asarray(predictions[name], dtype=np.int64).reshape(-1)
        n = min(pred.size, truth.size)
        sl = slice(max(0, n - lb), n)
        scores.append(float(np.mean(pred[sl] == truth[sl])) if n else 0.0)
    return normalize_weights(np.asarray(scores, dtype=np.float64), min_weight=min_weight)


def log_likelihood_weights(
    log_likes: dict[str, float],
    names: list[str],
    *,
    min_weight: float = 0.01,
) -> np.ndarray:
    vals = np.asarray([float(log_likes.get(n, -1e9)) for n in names], dtype=np.float64)
    # softmax of centered LL
    vals = vals - np.max(vals)
    w = np.exp(np.clip(vals, -50, 50))
    return normalize_weights(w, min_weight=min_weight)


def calibration_weights(
    ece_scores: dict[str, float],
    names: list[str],
    *,
    min_weight: float = 0.01,
) -> np.ndarray:
    # lower ECE → higher weight
    ece = np.asarray([max(float(ece_scores.get(n, 1.0)), 1e-6) for n in names], dtype=np.float64)
    return normalize_weights(1.0 / ece, min_weight=min_weight)


def stability_weights(
    proba_histories: dict[str, np.ndarray],
    names: list[str],
    *,
    min_weight: float = 0.01,
) -> np.ndarray:
    # higher weight for smoother probability paths (lower TV of mean max-proba)
    scores = []
    for name in names:
        p = np.asarray(proba_histories[name], dtype=np.float64)
        if p.ndim != 2 or p.shape[0] < 2:
            scores.append(0.5)
            continue
        mx = p.max(axis=1)
        tv = float(np.mean(np.abs(np.diff(mx))))
        scores.append(1.0 / (1.0 + tv))
    return normalize_weights(np.asarray(scores, dtype=np.float64), min_weight=min_weight)


def user_weights(
    user: dict[str, float] | None,
    names: list[str],
    *,
    min_weight: float = 0.01,
) -> np.ndarray:
    if not user:
        return equal_weights(len(names))
    w = np.asarray([float(user.get(n, min_weight)) for n in names], dtype=np.float64)
    return normalize_weights(w, min_weight=min_weight)


def rolling_weights(
    score_matrix: np.ndarray,
    *,
    min_weight: float = 0.01,
) -> np.ndarray:
    """``score_matrix`` shape ``(T, M)`` rolling mean of last rows."""
    s = np.asarray(score_matrix, dtype=np.float64)
    if s.ndim != 2 or s.size == 0:
        return equal_weights(1)
    mean = s.mean(axis=0)
    return normalize_weights(mean, min_weight=min_weight)


def adaptive_update(
    weights: np.ndarray,
    instant_scores: np.ndarray,
    *,
    rate: float = 0.05,
    min_weight: float = 0.01,
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    s = normalize_weights(instant_scores, min_weight=min_weight)
    mixed = (1.0 - rate) * w + rate * s
    return normalize_weights(mixed, min_weight=min_weight)


def compute_weights(
    method: WeightMethod,
    *,
    names: list[str],
    predictions: dict[str, np.ndarray] | None = None,
    truth: np.ndarray | None = None,
    log_likes: dict[str, float] | None = None,
    ece_scores: dict[str, float] | None = None,
    proba_histories: dict[str, np.ndarray] | None = None,
    user: dict[str, float] | None = None,
    score_matrix: np.ndarray | None = None,
    current: np.ndarray | None = None,
    lookback: int = 50,
    adaptive_rate: float = 0.05,
    min_weight: float = 0.01,
) -> np.ndarray:
    if method == "accuracy" and predictions is not None and truth is not None:
        return accuracy_weights(predictions, truth, names, min_weight=min_weight)
    if method == "recent_accuracy" and predictions is not None and truth is not None:
        return recent_accuracy_weights(
            predictions, truth, names, lookback=lookback, min_weight=min_weight
        )
    if method == "log_likelihood" and log_likes is not None:
        return log_likelihood_weights(log_likes, names, min_weight=min_weight)
    if method == "calibration" and ece_scores is not None:
        return calibration_weights(ece_scores, names, min_weight=min_weight)
    if method == "stability" and proba_histories is not None:
        return stability_weights(proba_histories, names, min_weight=min_weight)
    if method == "user":
        return user_weights(user, names, min_weight=min_weight)
    if method == "rolling" and score_matrix is not None:
        return rolling_weights(score_matrix, min_weight=min_weight)
    if method == "adaptive" and current is not None and score_matrix is not None:
        instant = score_matrix[-1] if score_matrix.ndim == 2 else score_matrix
        return adaptive_update(
            current, instant, rate=adaptive_rate, min_weight=min_weight
        )
    return equal_weights(len(names))

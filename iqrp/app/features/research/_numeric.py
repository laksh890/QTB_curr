"""Shared numeric helpers for the research engine (NumPy / SciPy only)."""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
from scipy import stats  # type: ignore[import-untyped]


def as_float_matrix(columns: Iterable[np.ndarray]) -> np.ndarray:
    cols = [np.asarray(c, dtype=np.float64) for c in columns]
    return np.column_stack(cols) if cols else np.empty((0, 0), dtype=np.float64)


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    m = finite_mask(x, y)
    if m.sum() < 3:
        return float("nan")
    xf, yf = x[m], y[m]
    if np.std(xf) == 0 or np.std(yf) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = stats.pearsonr(xf, yf)
    return float(r)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = finite_mask(x, y)
    if m.sum() < 3:
        return float("nan")
    xf, yf = x[m], y[m]
    if np.std(xf) == 0 or np.std(yf) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = stats.spearmanr(xf, yf)
    return float(r)


def kendall(x: np.ndarray, y: np.ndarray) -> float:
    m = finite_mask(x, y)
    if m.sum() < 3:
        return float("nan")
    xf, yf = x[m], y[m]
    if np.std(xf) == 0 or np.std(yf) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = stats.kendalltau(xf, yf)
    return float(r)


def rankdata(x: np.ndarray) -> np.ndarray:
    return cast(np.ndarray, stats.rankdata(x, method="average"))


def information_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson IC between feature and target."""
    return pearson(x, y)


def rank_information_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman / Rank IC."""
    return spearman(x, y)


def mutual_information(x: np.ndarray, y: np.ndarray, *, bins: int = 16) -> float:
    """Histogram-based mutual information (nats)."""
    m = finite_mask(x, y)
    if m.sum() < max(10, bins):
        return float("nan")
    xf, yf = x[m], y[m]
    c_xy, _, _ = np.histogram2d(xf, yf, bins=bins)
    p_xy = c_xy / c_xy.sum()
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    outer = np.outer(p_x, p_y)
    nz = p_xy > 0
    return float(np.sum(p_xy[nz] * np.log(p_xy[nz] / outer[nz])))


def distance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Szekely distance correlation."""
    m = finite_mask(x, y)
    if m.sum() < 4:
        return float("nan")
    a = x[m][:, None]
    b = y[m][:, None]
    a_dist = np.abs(a - a.T)
    b_dist = np.abs(b - b.T)
    a_centered = a_dist - a_dist.mean(axis=0) - a_dist.mean(axis=1)[:, None] + a_dist.mean()
    b_centered = b_dist - b_dist.mean(axis=0) - b_dist.mean(axis=1)[:, None] + b_dist.mean()
    dcov2 = (a_centered * b_centered).mean()
    dvar_x = (a_centered * a_centered).mean()
    dvar_y = (b_centered * b_centered).mean()
    if dvar_x <= 0 or dvar_y <= 0:
        return 0.0
    return float(np.sqrt(dcov2) / np.sqrt(np.sqrt(dvar_x) * np.sqrt(dvar_y)))


def try_mic(x: np.ndarray, y: np.ndarray) -> float | None:
    """Maximum Information Coefficient via minepy if installed."""
    m = finite_mask(x, y)
    if m.sum() < 20:
        return None
    try:
        minepy: Any = importlib.import_module("minepy")
        mine = minepy.MINE(alpha=0.6, c=15)
    except Exception:
        return None
    mine.compute_score(x[m], y[m])
    return float(mine.mic())


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    m = finite_mask(y_true, y_pred)
    if m.sum() < 3:
        return float("nan")
    yt, yp = y_true[m], y_pred[m]
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def ridge_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Univariate or multivariate ridge; x shape (n, p)."""
    if x_train.ndim == 1:
        x_train = x_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 1)
    m = finite_mask(y_train, *[x_train[:, j] for j in range(x_train.shape[1])])
    xt, yt = x_train[m], y_train[m]
    if len(yt) < x_train.shape[1] + 2:
        return np.full(len(x_test), np.nan)
    mu_x = xt.mean(axis=0)
    mu_y = yt.mean()
    xc = xt - mu_x
    yc = yt - mu_y
    p = xc.shape[1]
    gram = xc.T @ xc + alpha * np.eye(p)
    try:
        coef = np.linalg.solve(gram, xc.T @ yc)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(gram) @ (xc.T @ yc)
    return cast(np.ndarray, (x_test - mu_x) @ coef + mu_y)


def binary_classification_metrics(
    y_true: np.ndarray, scores: np.ndarray, *, threshold: float = 0.0
) -> dict[str, float]:
    m = finite_mask(y_true, scores)
    if m.sum() < 5:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "auc": float("nan"),
        }
    yt = (y_true[m] > threshold).astype(np.int32)
    pred = (scores[m] > threshold).astype(np.int32)
    tp = int(((pred == 1) & (yt == 1)).sum())
    tn = int(((pred == 0) & (yt == 0)).sum())
    fp = int(((pred == 1) & (yt == 0)).sum())
    fn = int(((pred == 0) & (yt == 1)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    auc = roc_auc(yt, scores[m])
    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC; y_true in {0,1}."""
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # P(score_pos > score_neg) + 0.5 P(equal)
    all_scores = np.concatenate([pos, neg])
    ranks = rankdata(all_scores)
    sum_pos = ranks[: len(pos)].sum()
    auc = (sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    return float(auc)


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, *, bins: int = 10
) -> float:
    m_ref = np.isfinite(reference)
    m_cur = np.isfinite(current)
    if m_ref.sum() < bins or m_cur.sum() < bins:
        return float("nan")
    edges = np.quantile(reference[m_ref], np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    ref_hist, _ = np.histogram(reference[m_ref], bins=edges)
    cur_hist, _ = np.histogram(current[m_cur], bins=edges)
    ref_p = ref_hist / max(ref_hist.sum(), 1)
    cur_p = cur_hist / max(cur_hist.sum(), 1)
    ref_p = np.clip(ref_p, 1e-6, None)
    cur_p = np.clip(cur_p, 1e-6, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    m_ref = np.isfinite(reference)
    m_cur = np.isfinite(current)
    if m_ref.sum() < 5 or m_cur.sum() < 5:
        return float("nan")
    stat, _ = stats.ks_2samp(reference[m_ref], current[m_cur])
    return float(stat)


def shannon_entropy(x: np.ndarray, *, bins: int = 20) -> float:
    m = np.isfinite(x)
    if m.sum() < bins:
        return float("nan")
    hist, _ = np.histogram(x[m], bins=bins, density=False)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def safe_nanmean(x: np.ndarray) -> float:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))

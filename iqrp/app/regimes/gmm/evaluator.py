"""Evaluation metrics for fitted GMMs."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math.statistics.entropy import cross_entropy, entropy
from iqrp.app.regimes.gmm.em import EMResult
from iqrp.app.regimes.gmm.mixture import GaussianMixtureParams
from iqrp.app.regimes.gmm.model_selection import score_result
from iqrp.app.regimes.gmm.prediction import cluster_stability
from iqrp.app.state_space.evaluation.diagnostics import probability_calibration


class GMMEvaluator:
    def evaluate(
        self,
        *,
        x: np.ndarray,
        params: GaussianMixtureParams,
        responsibilities: np.ndarray,
        log_likelihood: float,
        true_labels: np.ndarray | None = None,
    ) -> dict[str, Any]:
        pred = np.argmax(responsibilities, axis=1)
        result = EMResult(
            weights=params.weights,
            means=params.means,
            covars=params.covars,
            responsibilities=responsibilities,
            log_likelihood=log_likelihood,
            covariance_type=params.covariance_type,
            model_type=params.model_type,  # type: ignore[arg-type]
        )
        scores = score_result(result, x.shape[0])
        scores.update(
            {
                "mean_max_probability": float(np.mean(responsibilities.max(axis=1))),
                "mean_entropy": float(np.mean([entropy(row) for row in responsibilities])),
                "cluster_stability": float(cluster_stability(pred)),
                **_cluster_indices(x, pred),
            }
        )
        details: dict[str, Any] = {"n_params": params.n_params()}
        if true_labels is not None:
            truth = np.asarray(true_labels, dtype=np.int64).reshape(-1)
            scores["prediction_accuracy"] = _best_accuracy(pred, truth, params.n_components)
            scores["cross_entropy"] = float(_avg_ce(responsibilities, truth))
            details["calibration"] = probability_calibration(responsibilities, truth)
        return {"metrics": scores, "details": details}


def _avg_ce(proba: np.ndarray, truth: np.ndarray) -> float:
    if proba.ndim != 2 or proba.shape[0] != truth.size:
        return float("nan")
    total = 0.0
    for t, s in enumerate(truth):
        q = np.zeros(proba.shape[1])
        if 0 <= int(s) < len(q):
            q[int(s)] = 1.0
        total += float(cross_entropy(q, proba[t]))
    return total / max(len(truth), 1)


def _best_accuracy(pred: np.ndarray, truth: np.ndarray, n_states: int) -> float:
    from itertools import permutations

    k = min(n_states, 6)
    if n_states > 6:
        conf = np.zeros((n_states, n_states), dtype=np.float64)
        for a, b in zip(pred, truth, strict=False):
            if 0 <= a < n_states and 0 <= b < n_states:
                conf[a, b] += 1.0
        mapping = {i: int(np.argmax(conf[i])) for i in range(n_states)}
        mapped = np.array([mapping.get(int(x), int(x)) for x in pred])
        return float(np.mean(mapped == truth))
    best = 0.0
    for perm in permutations(range(k)):
        mapped = np.array([perm[int(x)] if 0 <= int(x) < k else int(x) for x in pred])
        best = max(best, float(np.mean(mapped == truth)))
    return best


def _cluster_indices(x: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    y = np.asarray(x, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    labs = np.asarray(labels, dtype=np.int64)
    uniq = np.unique(labs)
    if uniq.size < 2 or y.shape[0] < 3:
        return {"silhouette": 0.0, "davies_bouldin": 0.0, "calinski_harabasz": 0.0}
    centers = np.vstack([y[labs == u].mean(axis=0) for u in uniq])
    sil = []
    for i, lab in enumerate(labs):
        same = y[labs == lab]
        if same.shape[0] <= 1:
            continue
        a = float(np.mean(np.linalg.norm(same - y[i], axis=1)))
        b = np.inf
        for u, c in zip(uniq, centers, strict=False):
            if u == lab:
                continue
            b = min(b, float(np.linalg.norm(y[i] - c)))
        sil.append((b - a) / max(a, b, 1e-12))
    silhouette = float(np.mean(sil)) if sil else 0.0

    db = 0.0
    scatters = []
    for u, c in zip(uniq, centers, strict=False):
        pts = y[labs == u]
        scatters.append(float(np.mean(np.linalg.norm(pts - c, axis=1))))
    for i in range(len(uniq)):
        best = 0.0
        for j in range(len(uniq)):
            if i == j:
                continue
            sep = float(np.linalg.norm(centers[i] - centers[j]))
            best = max(best, (scatters[i] + scatters[j]) / max(sep, 1e-12))
        db += best
    db /= max(len(uniq), 1)

    overall = y.mean(axis=0)
    bgss = 0.0
    wgss = 0.0
    for u, c in zip(uniq, centers, strict=False):
        pts = y[labs == u]
        bgss += pts.shape[0] * float(np.sum((c - overall) ** 2))
        wgss += float(np.sum((pts - c) ** 2))
    ch = (bgss / max(len(uniq) - 1, 1)) / max(wgss / max(y.shape[0] - len(uniq), 1), 1e-12)
    return {
        "silhouette": silhouette,
        "davies_bouldin": float(db),
        "calinski_harabasz": float(ch),
    }

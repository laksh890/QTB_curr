"""Prediction and regime analysis for GMM assignments."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from iqrp.app.regimes.gmm.mixture import GaussianMixtureParams
from iqrp.app.state_space.base.forecast_result import ForecastResult


def hard_assignments(responsibilities: np.ndarray) -> np.ndarray:
    return np.argmax(np.asarray(responsibilities, dtype=np.float64), axis=1).astype(np.int64)


def soft_assignments(responsibilities: np.ndarray) -> np.ndarray:
    r = np.asarray(responsibilities, dtype=np.float64)
    return r / np.clip(r.sum(axis=1, keepdims=True), 1e-300, None)


def regime_occupancy(responsibilities: np.ndarray) -> np.ndarray:
    r = soft_assignments(responsibilities)
    return r.mean(axis=0)


def regime_persistence(hard: np.ndarray, n_components: int) -> np.ndarray:
    s = np.asarray(hard, dtype=np.int64).reshape(-1)
    k = int(n_components)
    counts = np.zeros((k, k), dtype=np.float64)
    for a, b in itertools.pairwise(s):
        if 0 <= a < k and 0 <= b < k:
            counts[a, b] += 1.0
    row = np.clip(counts.sum(axis=1), 1e-12, None)
    tm = counts / row[:, None]
    return np.diag(tm)


def transition_frequency(hard: np.ndarray, n_components: int) -> np.ndarray:
    s = np.asarray(hard, dtype=np.int64).reshape(-1)
    k = int(n_components)
    counts = np.zeros((k, k), dtype=np.float64)
    for a, b in itertools.pairwise(s):
        if 0 <= a < k and 0 <= b < k:
            counts[a, b] += 1.0
    total = max(float(counts.sum()), 1.0)
    return counts / total


def cluster_stability(hard: np.ndarray) -> float:
    s = np.asarray(hard, dtype=np.int64).reshape(-1)
    if s.size < 2:
        return 1.0
    return float(np.mean(s[1:] == s[:-1]))


def regime_similarity(means: np.ndarray) -> np.ndarray:
    m = np.asarray(means, dtype=np.float64)
    k = m.shape[0]
    sim = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            dist = float(np.linalg.norm(m[i] - m[j]))
            sim[i, j] = float(np.exp(-dist))
    return sim


def forecast_occupancy(
    responsibilities: np.ndarray,
    *,
    horizon: int = 5,
    state_names: tuple[str, ...] = (),
    confidence_level: float = 0.95,
) -> ForecastResult:
    """GMM is exchangeable; forecast uses current soft occupancy as stationary belief."""
    occ = regime_occupancy(responsibilities)
    h = max(1, int(horizon))
    steps = np.tile(occ[None, :], (h, 1))
    persist = regime_persistence(hard_assignments(responsibilities), occ.size)
    durations = {i: float(1.0 / max(1.0 - persist[i], 1e-6)) for i in range(occ.size)}
    return ForecastResult.from_probabilities(
        occ,
        horizon=h,
        expected_duration=durations,
        step_distributions=steps,
        state_names=state_names,
        confidence_level=confidence_level,
    )


def detect_outliers(
    params: GaussianMixtureParams,
    x: np.ndarray,
    *,
    density_quantile: float = 0.01,
) -> dict[str, Any]:
    dens = params.score_samples(x)
    thr = float(np.quantile(dens, density_quantile))
    mask = dens <= thr
    resp = params.responsibilities(x)
    occ = regime_occupancy(resp)
    return {
        "log_density": dens,
        "threshold": thr,
        "outlier_mask": mask,
        "n_outliers": int(mask.sum()),
        "occupancy": occ,
    }

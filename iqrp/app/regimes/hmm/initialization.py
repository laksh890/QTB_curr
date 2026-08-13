"""Parameter initialization strategies for HMMs."""

from __future__ import annotations

import itertools
from typing import Any, Literal

import numpy as np

from iqrp.app.math.matrices.matrix import normalize_rows
from iqrp.app.regimes.hmm.emissions import (
    DiscreteEmissionModel,
    EmissionModel,
    GaussianEmissionModel,
)
from iqrp.app.regimes.hmm.transitions import HMMTransitions

InitMethod = Literal["random", "uniform", "kmeans", "user"]


def initialize_parameters(
    observations: np.ndarray,
    n_states: int,
    *,
    method: InitMethod = "kmeans",
    emission_type: str = "gaussian",
    covariance_type: str = "diag",
    n_symbols: int | None = None,
    dirichlet_alpha: float = 1.0,
    user_params: dict[str, Any] | None = None,
    rng: np.random.Generator | None = None,
    min_covar: float = 1e-6,
) -> tuple[HMMTransitions, EmissionModel]:
    rng = rng or np.random.default_rng()
    y = np.asarray(observations)
    if emission_type == "discrete":
        y_int = np.asarray(y, dtype=np.int64).reshape(-1)
        n_sym = int(
            n_symbols if n_symbols is not None else (int(y_int.max()) + 1 if y_int.size else 2)
        )
        n_features = 1
    else:
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_features = int(y.shape[1])
        n_sym = None

    if method == "user" and user_params:
        trans = HMMTransitions.from_dict(user_params["transitions"])
        from iqrp.app.regimes.hmm.emissions import emission_from_dict

        emis = emission_from_dict(user_params["emissions"])
        return trans, emis

    if method == "uniform":
        trans = HMMTransitions(n_states, dirichlet_alpha=dirichlet_alpha)
        if emission_type == "discrete":
            emis = DiscreteEmissionModel(n_states, n_sym or 2, alpha=dirichlet_alpha)
        else:
            means = np.zeros((n_states, n_features))
            if y.size:
                means[:] = np.mean(y.reshape(-1, n_features), axis=0)
            emis = GaussianEmissionModel(
                n_states, n_features, means=means, covariance_type=covariance_type  # type: ignore[arg-type]
            )
        return trans, emis

    # random or kmeans
    if emission_type == "discrete":
        probs = rng.dirichlet(np.full(n_sym or 2, dirichlet_alpha), size=n_states)
        emis = DiscreteEmissionModel(n_states, n_sym or 2, probs=probs, alpha=dirichlet_alpha)
        labels = y_int % n_states if y_int.size else np.zeros(0, dtype=np.int64)
    else:
        if method == "kmeans" and y.shape[0] >= n_states:
            means, labels = _kmeans(y, n_states, rng=rng, n_iter=20)
        else:
            idx = rng.choice(y.shape[0], size=n_states, replace=(y.shape[0] < n_states))
            means = y[idx].copy()
            labels = _nearest_labels(y, means)
        covars = _init_covars(y, labels, n_states, n_features, covariance_type, min_covar)
        emis = GaussianEmissionModel(
            n_states,
            n_features,
            means=means,
            covars=covars,
            covariance_type=covariance_type,  # type: ignore[arg-type]
        )

    tm = np.full((n_states, n_states), 1.0 / n_states)
    if labels.size >= 2:
        counts = np.full((n_states, n_states), dirichlet_alpha)
        for a, b in itertools.pairwise(labels):
            if 0 <= int(a) < n_states and 0 <= int(b) < n_states:
                counts[int(a), int(b)] += 1.0
        tm = normalize_rows(counts)
    pi = (
        np.bincount(labels, minlength=n_states).astype(np.float64)
        if labels.size
        else np.ones(n_states)
    )
    pi = pi / max(float(pi.sum()), 1e-300)
    if method == "random":
        tm = normalize_rows(rng.dirichlet(np.full(n_states, dirichlet_alpha), size=n_states))
        pi = rng.dirichlet(np.full(n_states, dirichlet_alpha))
    trans = HMMTransitions(n_states, transition=tm, initial=pi, dirichlet_alpha=dirichlet_alpha)
    return trans, emis


def _kmeans(
    y: np.ndarray,
    k: int,
    *,
    rng: np.random.Generator,
    n_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    n = y.shape[0]
    centers = y[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        labels = _nearest_labels(y, centers)
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                centers[j] = y[mask].mean(axis=0)
    return centers, labels


def _nearest_labels(y: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # (T, K) squared distances
    d = ((y[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d, axis=1).astype(np.int64)


def _init_covars(
    y: np.ndarray,
    labels: np.ndarray,
    n_states: int,
    n_features: int,
    covariance_type: str,
    min_covar: float,
) -> np.ndarray:
    if covariance_type == "diag":
        cov = np.ones((n_states, n_features), dtype=np.float64)
        for k in range(n_states):
            mask = labels == k
            if np.any(mask):
                cov[k] = np.clip(np.var(y[mask], axis=0), min_covar, None)
            else:
                cov[k] = np.clip(np.var(y, axis=0), min_covar, None)
        return cov
    cov = np.array([np.eye(n_features) for _ in range(n_states)], dtype=np.float64)
    for k in range(n_states):
        mask = labels == k
        subset = y[mask] if np.any(mask) else y
        if subset.shape[0] > 1:
            c = np.cov(subset, rowvar=False)
            if np.ndim(c) == 0:
                c = np.array([[float(c)]])
            cov[k] = c + min_covar * np.eye(n_features)
    return cov

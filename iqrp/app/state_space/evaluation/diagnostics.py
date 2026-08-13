"""Diagnostics for state-space inference quality."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np

from iqrp.app.math.matrices.matrix import normalize_rows
from iqrp.app.math.statistics.descriptive import mean, variance
from iqrp.app.math.stochastic.markov_utils import empirical_transition


class StateSpaceDiagnostics:
    """Occupancy, persistence, calibration, residual, and convergence checks."""

    def analyze(
        self,
        *,
        states: np.ndarray,
        probabilities: np.ndarray,
        transition_matrix: np.ndarray | None = None,
        observations: np.ndarray | None = None,
        expected_observations: np.ndarray | None = None,
        log_likelihood_history: list[float] | None = None,
        n_states: int | None = None,
    ) -> dict[str, Any]:
        s = np.asarray(states, dtype=np.int64).reshape(-1)
        proba = np.asarray(probabilities, dtype=np.float64)
        k = int(
            n_states
            if n_states is not None
            else (proba.shape[1] if proba.ndim == 2 else (s.max() + 1 if s.size else 0))
        )
        return {
            "occupancy": state_occupancy_analysis(s, k),
            "transition_frequency": transition_frequency_analysis(s, k),
            "persistence": persistence_analysis(s, transition_matrix),
            "calibration": probability_calibration(proba, s),
            "residuals": residual_diagnostics(observations, expected_observations),
            "likelihood_convergence": likelihood_convergence_checks(log_likelihood_history),
            "empirical_transition": empirical_transition(s, n_states=k).tolist(),
        }


def state_occupancy_analysis(states: np.ndarray, n_states: int) -> dict[str, Any]:
    s = np.asarray(states, dtype=np.int64).reshape(-1)
    counts = np.bincount(s, minlength=n_states).astype(np.float64)
    total = max(float(counts.sum()), 1.0)
    return {
        "counts": counts.tolist(),
        "frequencies": (counts / total).tolist(),
        "entropy": float(-np.sum((counts / total) * np.log(np.clip(counts / total, 1e-300, None)))),
    }


def transition_frequency_analysis(states: np.ndarray, n_states: int) -> dict[str, Any]:
    s = np.asarray(states, dtype=np.int64).reshape(-1)
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    switches = 0
    for a, b in pairwise(s):
        if 0 <= a < n_states and 0 <= b < n_states:
            counts[a, b] += 1.0
            if a != b:
                switches += 1
    return {
        "counts": counts.tolist(),
        "row_normalized": normalize_rows(counts).tolist(),
        "n_switches": int(switches),
        "switch_rate": float(switches / max(len(s) - 1, 1)),
    }


def persistence_analysis(
    states: np.ndarray,
    transition_matrix: np.ndarray | None = None,
) -> dict[str, Any]:
    s = np.asarray(states, dtype=np.int64).reshape(-1)
    run_lengths: list[int] = []
    if s.size:
        run = 1
        for a, b in pairwise(s):
            if a == b:
                run += 1
            else:
                run_lengths.append(run)
                run = 1
        run_lengths.append(run)
    empirical_mean = float(mean(run_lengths)) if run_lengths else 0.0
    model_expected: dict[str, float] = {}
    if transition_matrix is not None:
        tm = np.asarray(transition_matrix, dtype=np.float64)
        for i in range(tm.shape[0]):
            model_expected[str(i)] = float(1.0 / max(1.0 - float(tm[i, i]), 1e-12))
    return {
        "run_lengths": run_lengths,
        "mean_run_length": empirical_mean,
        "max_run_length": int(max(run_lengths) if run_lengths else 0),
        "model_expected_duration": model_expected,
    }


def probability_calibration(probabilities: np.ndarray, states: np.ndarray) -> dict[str, Any]:
    """Reliability of predicted max-probability vs hit rate in bins."""
    proba = np.asarray(probabilities, dtype=np.float64)
    s = np.asarray(states, dtype=np.int64).reshape(-1)
    if proba.ndim != 2 or proba.shape[0] != s.size:
        return {"bins": [], "ece": float("nan")}
    conf = proba.max(axis=1)
    hits = (np.argmax(proba, axis=1) == s).astype(np.float64)
    edges = np.linspace(0.0, 1.0, 11)
    bins: list[dict[str, float]] = []
    ece = 0.0
    for i in range(10):
        mask = (conf >= edges[i]) & (conf < edges[i + 1] if i < 9 else conf <= edges[i + 1])
        if not np.any(mask):
            continue
        avg_conf = float(np.mean(conf[mask]))
        avg_acc = float(np.mean(hits[mask]))
        weight = float(np.mean(mask))
        ece += weight * abs(avg_conf - avg_acc)
        bins.append(
            {
                "lo": float(edges[i]),
                "hi": float(edges[i + 1]),
                "avg_confidence": avg_conf,
                "avg_accuracy": avg_acc,
                "weight": weight,
            }
        )
    return {"bins": bins, "ece": float(ece)}


def residual_diagnostics(
    observations: np.ndarray | None,
    expected_observations: np.ndarray | None,
) -> dict[str, Any]:
    if observations is None or expected_observations is None:
        return {"available": False}
    y = np.asarray(observations, dtype=np.float64)
    mu = np.asarray(expected_observations, dtype=np.float64)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if mu.ndim == 1:
        mu = mu.reshape(-1, 1)
    n = min(len(y), len(mu))
    resid = y[:n] - mu[:n]
    flat = resid.reshape(-1)
    return {
        "available": True,
        "mean": float(mean(flat)),
        "variance": float(variance(flat)),
        "rmse": float(np.sqrt(np.mean(flat**2))),
        "max_abs": float(np.max(np.abs(flat))) if flat.size else 0.0,
    }


def likelihood_convergence_checks(history: list[float] | None) -> dict[str, Any]:
    if not history:
        return {"available": False, "converged": None}
    h = np.asarray(history, dtype=np.float64)
    deltas = np.diff(h)
    nondecreasing = bool(np.all(deltas >= -1e-8))
    final_delta = float(deltas[-1]) if deltas.size else 0.0
    return {
        "available": True,
        "n_iters": int(h.size),
        "final_log_likelihood": float(h[-1]),
        "final_delta": final_delta,
        "nondecreasing": nondecreasing,
        "converged": bool(nondecreasing and abs(final_delta) < 1e-6),
    }

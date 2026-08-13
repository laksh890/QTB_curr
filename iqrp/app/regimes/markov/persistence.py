"""Persistence and occupancy analysis for discrete Markov chains."""

from __future__ import annotations

from itertools import pairwise
from typing import Any

import numpy as np

from iqrp.app.math.matrices.matrix import normalize_rows


class PersistenceAnalyzer:
    def analyze(
        self,
        states: Any,
        transition: Any | None = None,
        *,
        n_states: int | None = None,
    ) -> dict[str, Any]:
        s = np.asarray(states, dtype=np.int64).reshape(-1)
        k = int(n_states if n_states is not None else (int(s.max()) + 1 if s.size else 0))
        run_lengths = _run_lengths(s)
        occupancy = state_occupancy(s, k)
        avg_duration = {i: float(np.mean(run_lengths.get(i, [0.0]) or [0.0])) for i in range(k)}
        expected = expected_duration(transition) if transition is not None else {}
        return {
            "average_state_duration": avg_duration,
            "expected_duration": expected,
            "persistence_score": persistence_score(
                transition if transition is not None else np.eye(k)
            ),
            "transition_frequency": transition_frequency(s, k),
            "state_occupancy": occupancy,
            "run_lengths": {str(i): run_lengths.get(i, []) for i in range(k)},
            "mean_run_length": float(
                np.mean([x for xs in run_lengths.values() for x in xs])
                if any(run_lengths.values())
                else 0.0
            ),
        }

    def expected_duration(self, transition: Any) -> dict[int, float]:
        return expected_duration(transition)


def expected_duration(transition: Any) -> dict[int, float]:
    p = np.asarray(transition, dtype=np.float64)
    out: dict[int, float] = {}
    for i in range(p.shape[0]):
        out[i] = float(1.0 / max(1.0 - float(p[i, i]), 1e-12))
    return out


def persistence_score(transition: Any) -> dict[int, float]:
    """Diagonal self-transition probabilities as persistence scores."""
    p = np.asarray(transition, dtype=np.float64)
    return {i: float(p[i, i]) for i in range(p.shape[0])}


def state_occupancy(states: Any, n_states: int) -> dict[str, Any]:
    s = np.asarray(states, dtype=np.int64).reshape(-1)
    counts = np.bincount(np.clip(s, 0, max(n_states - 1, 0)), minlength=n_states).astype(np.float64)
    total = max(float(counts.sum()), 1.0)
    return {"counts": counts.tolist(), "frequencies": (counts / total).tolist()}


def transition_frequency(states: Any, n_states: int) -> dict[str, Any]:
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


def _run_lengths(states: np.ndarray) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    if states.size == 0:
        return out
    run = 1
    cur = int(states[0])
    for a, b in pairwise(states):
        if a == b:
            run += 1
        else:
            out.setdefault(cur, []).append(run)
            cur = int(b)
            run = 1
    out.setdefault(cur, []).append(run)
    return out

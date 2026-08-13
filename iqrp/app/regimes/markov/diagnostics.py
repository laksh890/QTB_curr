"""Diagnostics for fitted Markov chains."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math.statistics.entropy import entropy
from iqrp.app.regimes.markov.persistence import PersistenceAnalyzer
from iqrp.app.regimes.markov.stationary import StationaryAnalyzer


class MarkovDiagnostics:
    def generate(
        self,
        *,
        states: np.ndarray,
        transition: np.ndarray,
        counts: np.ndarray | None = None,
        min_count_warning: int = 5,
        state_names: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        p = np.asarray(transition, dtype=np.float64)
        s = np.asarray(states, dtype=np.int64).reshape(-1)
        k = p.shape[0]
        names = state_names or tuple(f"state_{i}" for i in range(k))
        row_entropy = [float(entropy(p[i])) for i in range(k)]
        c = counts if counts is not None else np.zeros_like(p)
        row_sums = c.sum(axis=1) if c is not None else np.zeros(k)
        low_sample = [int(i) for i in range(k) if float(row_sums[i]) < min_count_warning]
        occ = np.bincount(np.clip(s, 0, k - 1), minlength=k).astype(np.float64)
        rare = [int(i) for i in range(k) if occ[i] < min_count_warning]
        persist = PersistenceAnalyzer().analyze(s, p, n_states=k)
        stationary = StationaryAnalyzer().analyze(p)
        return {
            "transition_matrix_summary": {
                "shape": list(p.shape),
                "diagonal": np.diag(p).tolist(),
                "min": float(p.min()),
                "max": float(p.max()),
                "mean_self_transition": float(np.mean(np.diag(p))),
            },
            "state_frequencies": {
                names[i] if i < len(names) else f"state_{i}": float(occ[i] / max(occ.sum(), 1.0))
                for i in range(k)
            },
            "persistence_report": persist,
            "entropy_of_transitions": row_entropy,
            "mean_transition_entropy": float(np.mean(row_entropy)) if row_entropy else 0.0,
            "transition_uncertainty": {
                "row_entropy": row_entropy,
                "max_entropy": float(np.log(k)) if k > 0 else 0.0,
            },
            "low_sample_warnings": low_sample,
            "rare_states": rare,
            "stationary": {
                "distribution": stationary["stationary_distribution"].tolist(),
                "is_ergodic": stationary["is_ergodic"],
                "mixing_time": stationary["mixing_time"],
            },
        }

"""Diagnostics for fitted Hidden Markov Models."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.math.statistics.entropy import entropy
from iqrp.app.regimes.markov.persistence import PersistenceAnalyzer


class HMMDiagnostics:
    def generate(
        self,
        *,
        states: np.ndarray,
        probabilities: np.ndarray,
        transition: np.ndarray,
        emissions: Any,
        history: list[float] | None = None,
        converged: bool | None = None,
        n_iter: int | None = None,
        min_occupancy: int = 5,
        state_names: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        s = np.asarray(states, dtype=np.int64).reshape(-1)
        proba = np.asarray(probabilities, dtype=np.float64)
        p = np.asarray(transition, dtype=np.float64)
        k = p.shape[0]
        names = state_names or tuple(f"state_{i}" for i in range(k))
        occ = np.bincount(np.clip(s, 0, k - 1), minlength=k).astype(np.float64)
        rare = [int(i) for i in range(k) if occ[i] < min_occupancy]
        row_ent = [float(entropy(p[i])) for i in range(k)]
        persist = PersistenceAnalyzer().analyze(s, p, n_states=k)
        emis = emissions.to_dict() if hasattr(emissions, "to_dict") else {}
        hist = list(history or [])
        deltas = np.diff(hist) if len(hist) > 1 else np.array([])
        return {
            "convergence": {
                "history": hist,
                "n_iter": n_iter,
                "converged": converged,
                "final_log_likelihood": hist[-1] if hist else None,
                "final_delta": float(deltas[-1]) if deltas.size else None,
                "nondecreasing": bool(np.all(deltas >= -1e-8)) if deltas.size else None,
            },
            "transition_matrix": p.tolist(),
            "emission_parameters": emis,
            "posterior_summary": {
                "mean_max": float(np.mean(proba.max(axis=1))) if proba.ndim == 2 else None,
                "mean_entropy": (
                    float(np.mean([entropy(row) for row in proba])) if proba.ndim == 2 else None
                ),
            },
            "state_occupancy": {
                names[i] if i < len(names) else f"state_{i}": float(occ[i] / max(occ.sum(), 1.0))
                for i in range(k)
            },
            "state_duration": persist,
            "entropy_of_transitions": row_ent,
            "rare_states": rare,
        }

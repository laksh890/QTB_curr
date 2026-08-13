"""Reusable probability engine for regime states and transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ProbabilityBundle:
    """Container for common regime probability views."""

    state_probabilities: np.ndarray  # shape (T, K) or (K,)
    transition_matrix: np.ndarray  # shape (K, K)
    joint_probabilities: np.ndarray | None = None  # optional (K, K)
    conditional_probabilities: np.ndarray | None = None  # optional (K, K)
    forecast_probabilities: np.ndarray | None = None  # shape (H, K)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_probabilities": np.asarray(self.state_probabilities).tolist(),
            "transition_matrix": np.asarray(self.transition_matrix).tolist(),
            "joint_probabilities": (
                None
                if self.joint_probabilities is None
                else np.asarray(self.joint_probabilities).tolist()
            ),
            "conditional_probabilities": (
                None
                if self.conditional_probabilities is None
                else np.asarray(self.conditional_probabilities).tolist()
            ),
            "forecast_probabilities": (
                None
                if self.forecast_probabilities is None
                else np.asarray(self.forecast_probabilities).tolist()
            ),
        }


class ProbabilityEngine:
    """Compute state, transition, joint, conditional, and forecast probabilities."""

    @staticmethod
    def normalize_rows(matrix: np.ndarray) -> np.ndarray:
        mat = np.asarray(matrix, dtype=np.float64)
        if mat.ndim == 1:
            s = mat.sum()
            return mat / s if s > 0 else np.full_like(mat, 1.0 / max(len(mat), 1))
        out = mat.copy()
        row_sums = out.sum(axis=1, keepdims=True)
        zero = row_sums.ravel() <= 0
        if zero.any():
            k = out.shape[1]
            out[zero] = 1.0 / max(k, 1)
            row_sums = out.sum(axis=1, keepdims=True)
        return out / row_sums

    @staticmethod
    def state_probability(proba: np.ndarray, state_id: int | None = None) -> np.ndarray | float:
        arr = np.asarray(proba, dtype=np.float64)
        if state_id is None:
            return arr
        if arr.ndim == 1:
            return float(arr[state_id])
        return arr[:, state_id]

    @staticmethod
    def transition_probability(transition_matrix: np.ndarray, i: int, j: int) -> float:
        tm = np.asarray(transition_matrix, dtype=np.float64)
        return float(tm[i, j])

    @staticmethod
    def joint_probability(transition_matrix: np.ndarray, stationary: np.ndarray) -> np.ndarray:
        """P(i,j) ≈ π_i * P(j|i)."""
        tm = ProbabilityEngine.normalize_rows(transition_matrix)
        pi = np.asarray(stationary, dtype=np.float64)
        pi = pi / pi.sum() if pi.sum() > 0 else np.full_like(pi, 1.0 / len(pi))
        return np.asarray(pi[:, None] * tm, dtype=np.float64)

    @staticmethod
    def conditional_probability(transition_matrix: np.ndarray) -> np.ndarray:
        """P(j|i) — row-normalized transition matrix."""
        return ProbabilityEngine.normalize_rows(transition_matrix)

    @staticmethod
    def forecast_probability(
        current: np.ndarray,
        transition_matrix: np.ndarray,
        steps: int,
    ) -> np.ndarray:
        """Iterate π_{t+h} = π_t P^h for h=1..steps. Returns shape (steps, K)."""
        tm = ProbabilityEngine.normalize_rows(transition_matrix)
        pi = np.asarray(current, dtype=np.float64)
        if pi.ndim != 1:
            pi = pi[-1]
        pi = pi / pi.sum() if pi.sum() > 0 else np.full_like(pi, 1.0 / len(pi))
        out = np.zeros((max(steps, 0), len(pi)), dtype=np.float64)
        cur = pi
        for h in range(steps):
            cur = cur @ tm
            out[h] = cur
        return out

    @classmethod
    def bundle(
        cls,
        state_probabilities: np.ndarray,
        transition_matrix: np.ndarray,
        *,
        forecast_steps: int = 0,
    ) -> ProbabilityBundle:
        tm = cls.normalize_rows(transition_matrix)
        sp = np.asarray(state_probabilities, dtype=np.float64)
        stationary = sp[-1] if sp.ndim == 2 else sp
        joint = cls.joint_probability(tm, stationary)
        cond = cls.conditional_probability(tm)
        forecast = (
            cls.forecast_probability(stationary, tm, forecast_steps) if forecast_steps > 0 else None
        )
        return ProbabilityBundle(
            state_probabilities=sp,
            transition_matrix=tm,
            joint_probabilities=joint,
            conditional_probabilities=cond,
            forecast_probabilities=forecast,
        )

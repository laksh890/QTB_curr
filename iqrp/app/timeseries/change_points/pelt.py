"""PELT change-point detection with L2 cost."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import ChangePointResult, TemporalMode, as_float_array


def pelt_detect(
    x: np.ndarray | list[float],
    *,
    penalty: float | None = None,
    min_size: int = 5,
) -> ChangePointResult:
    """Pruned Exact Linear Time (PELT) for mean shifts, L2 cost (FULL_SAMPLE)."""
    y = as_float_array(x)
    n = y.size
    if n < max(2 * min_size, 8):
        return ChangePointResult(
            method="pelt",
            indices=[],
            scores=None,
            kind="mean",
            parameters={"penalty": penalty, "min_size": min_size},
            temporal_mode=TemporalMode.FULL_SAMPLE,
            metadata={"status": "insufficient_data", "n": n},
        )
    y = np.where(np.isfinite(y), y, np.nanmean(y) if np.isfinite(y).any() else 0.0)
    pen = float(penalty) if penalty is not None else float(np.log(n) * max(np.var(y), 1e-12))
    ms = max(int(min_size), 1)

    # prefix sums
    csum = np.concatenate([[0.0], np.cumsum(y)])
    csum2 = np.concatenate([[0.0], np.cumsum(y**2)])

    def cost(a: int, b: int) -> float:
        """Cost of segment (a, b] (1-indexed ends via prefix arrays)."""
        length = b - a
        if length <= 0:
            return 0.0
        s1 = csum[b] - csum[a]
        s2 = csum2[b] - csum2[a]
        return float(s2 - (s1 * s1) / length)

    # F[t] = min cost of segmentation of y[0:t]
    F = np.full(n + 1, np.inf, dtype=np.float64)
    F[0] = -pen
    last_cp = np.zeros(n + 1, dtype=np.int64)
    R: list[int] = [0]  # candidate last changepoint starts

    for t in range(ms, n + 1):
        # candidates that can end a valid segment at t
        candidates = [r for r in R if t - r >= ms]
        if not candidates:
            candidates = [0]
        costs = np.array([F[r] + cost(r, t) + pen for r in candidates], dtype=np.float64)
        best_i = int(np.argmin(costs))
        F[t] = costs[best_i]
        last_cp[t] = candidates[best_i]

        # pruning: keep r if F[r] + cost(r,t) <= F[t]
        R = [r for r in R if F[r] + cost(r, t) <= F[t] + 1e-12]
        if t - ms + 1 >= 0:
            R.append(t - ms + 1)
        # unique preserve order
        seen: set[int] = set()
        R2: list[int] = []
        for r in R:
            if r not in seen and r < t:
                seen.add(r)
                R2.append(r)
        R = R2

    # backtrack
    cps: list[int] = []
    cur = n
    while cur > 0:
        prev = int(last_cp[cur])
        if prev > 0:
            cps.append(prev)
        cur = prev
    cps = sorted(cps)
    scores = np.zeros(n, dtype=np.float64)
    for c in cps:
        if 0 < c < n:
            scores[c] = float(cost(0, n) - (cost(0, c) + cost(c, n)))

    return ChangePointResult(
        method="pelt",
        indices=cps,
        scores=scores,
        kind="mean",
        parameters={"penalty": pen, "min_size": ms},
        temporal_mode=TemporalMode.FULL_SAMPLE,
        metadata={"n": n, "total_cost": float(F[n])},
    )

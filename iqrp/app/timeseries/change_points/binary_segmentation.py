"""Binary segmentation change-point detection."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import ChangePointResult, TemporalMode, as_float_array


def binseg_detect(
    x: np.ndarray | list[float],
    *,
    max_cps: int = 5,
    min_size: int = 10,
    penalty: float | None = None,
) -> ChangePointResult:
    """Binary segmentation with L2 (mean) cost (FULL_SAMPLE)."""
    y = as_float_array(x)
    n = y.size
    if n < max(min_size * 2, 10) or not np.isfinite(y).all():
        y = np.where(np.isfinite(y), y, np.nanmean(y) if np.isfinite(y).any() else 0.0)
    if n < max(min_size * 2, 10):
        return ChangePointResult(
            method="binseg",
            indices=[],
            scores=None,
            kind="mean",
            parameters={"max_cps": max_cps, "min_size": min_size, "penalty": penalty},
            temporal_mode=TemporalMode.FULL_SAMPLE,
            metadata={"status": "insufficient_data", "n": n},
        )

    pen = float(penalty) if penalty is not None else float(np.log(n) * np.var(y))
    cps: list[int] = []
    gains: list[float] = []
    segments: list[tuple[int, int]] = [(0, n)]
    score = np.zeros(n, dtype=np.float64)

    for _ in range(max(int(max_cps), 1)):
        best_gain = -np.inf
        best_tau = -1
        best_seg_i = -1
        for si, (a, b) in enumerate(segments):
            if b - a < 2 * min_size:
                continue
            tau, gain = _best_split(y, a, b, min_size)
            if gain > best_gain:
                best_gain = gain
                best_tau = tau
                best_seg_i = si
        if best_tau < 0 or best_gain <= pen:
            break
        cps.append(best_tau)
        gains.append(float(best_gain))
        score[best_tau] = best_gain
        a, b = segments.pop(best_seg_i)
        segments.extend([(a, best_tau), (best_tau, b)])
        segments.sort(key=lambda s: s[0])

    cps_sorted = sorted(cps)
    return ChangePointResult(
        method="binseg",
        indices=cps_sorted,
        scores=score,
        kind="mean",
        parameters={"max_cps": max_cps, "min_size": min_size, "penalty": pen},
        temporal_mode=TemporalMode.FULL_SAMPLE,
        metadata={"gains": gains, "n": n},
    )


def _segment_cost(y: np.ndarray, a: int, b: int) -> float:
    """L2 cost of a constant-mean segment [a, b)."""
    if b <= a:
        return 0.0
    seg = y[a:b]
    mu = float(np.mean(seg))
    return float(np.sum((seg - mu) ** 2))


def _best_split(y: np.ndarray, a: int, b: int, min_size: int) -> tuple[int, float]:
    """Find tau in (a+min_size, b-min_size] maximizing cost reduction."""
    full = _segment_cost(y, a, b)
    best_tau = -1
    best_gain = -np.inf
    # prefix sums for O(n) search
    seg = y[a:b]
    m = seg.size
    csum = np.cumsum(seg)
    csum2 = np.cumsum(seg ** 2)

    def cost_len(s2: float, s1: float, length: int) -> float:
        if length <= 0:
            return 0.0
        return float(s2 - (s1 * s1) / length)

    for t in range(min_size, m - min_size + 1):
        left = cost_len(csum2[t - 1], csum[t - 1], t)
        right_s1 = csum[m - 1] - csum[t - 1]
        right_s2 = csum2[m - 1] - csum2[t - 1]
        right = cost_len(right_s2, right_s1, m - t)
        gain = full - (left + right)
        if gain > best_gain:
            best_gain = gain
            best_tau = a + t
    return best_tau, float(best_gain)

"""Simple shapelet discovery."""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


def discover_shapelets(
    x: np.ndarray | list[float],
    *,
    labels: np.ndarray | list[int] | None = None,
    lengths: tuple[int, ...] = (8, 16, 32),
    top_k: int = 3,
    n_candidates: int = 50,
    seed: int = 42,
) -> AnalysisResult:
    """Discover discriminative or high-variance shapelets (FULL_SAMPLE).

    If ``labels`` is provided, candidates are scored by information gain on the
    distance-to-shapelet threshold; otherwise by within-series variance contrast
    (high local variance relative to global).
    """
    y = as_float_array(x)
    finite = np.isfinite(y)
    if not finite.all():
        y = y.copy()
        y[~finite] = float(np.nanmean(y[finite])) if finite.any() else 0.0
    n = y.size
    k = max(int(top_k), 1)
    lens = [int(L) for L in lengths if 2 <= int(L) <= n]
    if not lens or n < min(lens) + 2:
        return AnalysisResult(
            method="discover_shapelets",
            value="insufficient_data",
            temporal_mode=TemporalMode.FULL_SAMPLE,
            null_hypothesis="no discriminative / distinctive shapelets",
            alternative_hypothesis="one or more informative shapelets exist",
            parameters={"lengths": lengths, "top_k": k, "n_candidates": n_candidates},
        )
    rng = np.random.default_rng(int(seed))
    lab = None if labels is None else np.asarray(labels).reshape(-1)
    if lab is not None and lab.size != n:
        lab = None  # fall back to unsupervised if misaligned

    candidates: list[dict] = []
    for L in lens:
        n_sub = n - L + 1
        n_take = min(int(n_candidates), n_sub)
        starts = (
            rng.choice(n_sub, size=n_take, replace=False) if n_sub > n_take else np.arange(n_sub)
        )
        for s in starts:
            s = int(s)
            shape = y[s : s + L]
            score = _score_shapelet(shape, y, lab)
            candidates.append(
                {
                    "start": s,
                    "length": L,
                    "score": float(score),
                    "shapelet": shape.tolist(),
                }
            )
    candidates.sort(key=lambda c: c["score"], reverse=True)
    # diversify by exclusion
    selected: list[dict] = []
    for c in candidates:
        if any(
            abs(c["start"] - s["start"]) < c["length"] // 2 and c["length"] == s["length"]
            for s in selected
        ):
            continue
        selected.append(c)
        if len(selected) >= k:
            break
    return AnalysisResult(
        method="discover_shapelets",
        value=selected,
        statistic=selected[0]["score"] if selected else np.nan,
        temporal_mode=TemporalMode.FULL_SAMPLE,
        null_hypothesis="no discriminative / distinctive shapelets",
        alternative_hypothesis="one or more informative shapelets exist",
        significant=len(selected) > 0,
        parameters={
            "lengths": lens,
            "top_k": k,
            "n_candidates": n_candidates,
            "seed": seed,
            "supervised": lab is not None,
        },
        metadata={"n": n, "n_evaluated": len(candidates)},
    )


def _score_shapelet(shape: np.ndarray, series: np.ndarray, labels: np.ndarray | None) -> float:
    L = shape.size
    n_sub = series.size - L + 1
    sn = _znorm(shape)
    dists = np.empty(n_sub, dtype=np.float64)
    for i in range(n_sub):
        dists[i] = float(np.sqrt(np.sum((_znorm(series[i : i + L]) - sn) ** 2)))
    if labels is None:
        # unsupervised: prefer shapelets that are rare (high min distance elsewhere)
        # vs locally high energy
        local_var = float(np.var(shape))
        global_var = float(np.var(series)) + 1e-12
        rarity = float(np.percentile(dists, 10))
        return local_var / global_var + rarity
    # map subsequence distances to point labels via nearest start index labels
    # use information gain with best threshold
    # label each subsequence start by majority of labels in window
    sub_labels = np.array([int(np.round(np.mean(labels[i : i + L]))) for i in range(n_sub)])
    order = np.argsort(dists)
    dists[order]
    lab_sorted = sub_labels[order]
    best_ig = -np.inf
    for t in range(1, n_sub):
        left, right = lab_sorted[:t], lab_sorted[t:]
        ig = _entropy(lab_sorted) - (
            left.size / n_sub * _entropy(left) + right.size / n_sub * _entropy(right)
        )
        if ig > best_ig:
            best_ig = ig
    return float(best_ig)


def _entropy(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + 1e-300)))


def _znorm(v: np.ndarray) -> np.ndarray:
    sd = float(np.std(v))
    if sd < 1e-12:
        return v - np.mean(v)
    return (v - np.mean(v)) / sd

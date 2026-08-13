"""Online / streaming CUSUM change-point detector."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from iqrp.app.timeseries.base import ChangePointResult, TemporalMode, as_float_array
from iqrp.app.timeseries.rolling import incremental_mean_var


@dataclass
class OnlineCUSUMState:
    """Mutable state for streaming CUSUM."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    s_pos: float = 0.0
    s_neg: float = 0.0
    indices: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)


def online_cusum(
    x: np.ndarray | list[float],
    *,
    threshold: float = 5.0,
    drift: float = 0.5,
    warmup: int = 20,
    state: OnlineCUSUMState | None = None,
) -> ChangePointResult:
    """Causal Page-Hinkley / two-sided CUSUM with online mean/variance (CAUSAL).

    Can be called repeatedly with new batches by passing the returned state
    via ``metadata['state']`` or the ``state`` argument.
    """
    y = as_float_array(x)
    st = state if state is not None else OnlineCUSUMState()
    thr = float(threshold)
    d = float(max(drift, 0.0))
    scores = np.full(y.size, np.nan, dtype=np.float64)
    new_indices: list[int] = []

    if y.size == 0:
        return ChangePointResult(
            method="online_cusum",
            indices=list(st.indices),
            scores=np.asarray(st.scores, dtype=np.float64) if st.scores else None,
            kind="mean",
            parameters={"threshold": thr, "drift": d, "warmup": warmup},
            temporal_mode=TemporalMode.CAUSAL,
            metadata={"status": "insufficient_data", "state": st, "n": st.n},
        )

    for i, xt in enumerate(y):
        if not np.isfinite(xt):
            continue
        st.n, st.mean, st.m2 = incremental_mean_var(st.n, st.mean, st.m2, float(xt))
        if st.n < max(int(warmup), 2):
            scores[i] = 0.0
            st.scores.append(0.0)
            continue
        var = st.m2 / max(st.n - 1, 1)
        sd = float(np.sqrt(var)) if var > 1e-18 else 1.0
        z = (float(xt) - st.mean) / sd
        st.s_pos = max(0.0, st.s_pos + z - d)
        st.s_neg = max(0.0, st.s_neg - z - d)
        score = max(st.s_pos, st.s_neg)
        scores[i] = score
        st.scores.append(score)
        if score >= thr:
            global_idx = st.n - 1  # index in the stream so far
            st.indices.append(global_idx)
            new_indices.append(global_idx)
            # reset after detection
            st.s_pos = 0.0
            st.s_neg = 0.0

    return ChangePointResult(
        method="online_cusum",
        indices=list(st.indices),
        scores=scores,
        kind="mean",
        parameters={"threshold": thr, "drift": d, "warmup": warmup},
        temporal_mode=TemporalMode.CAUSAL,
        metadata={
            "n": st.n,
            "new_indices": new_indices,
            "state": st,
            "mean": st.mean,
            "std": float(np.sqrt(st.m2 / max(st.n - 1, 1))) if st.n > 1 else 0.0,
        },
    )

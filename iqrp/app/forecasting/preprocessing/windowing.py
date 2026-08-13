"""Sliding-window construction for sequence / multi-step forecasting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class WindowBatch:
    """Supervised tensors from a sliding window."""

    X: np.ndarray  # (N, window, F) or (N, window*F)
    y: np.ndarray  # (N,) or (N, horizon)
    indices: np.ndarray  # end index of each window in original series
    metadata: dict[str, Any] | None = None


def make_windows(
    features: np.ndarray,
    target: np.ndarray | None = None,
    *,
    window_size: int = 32,
    horizon: int = 1,
    flatten: bool = False,
) -> WindowBatch:
    """Build supervised windows ``X[t-w:t] -> y[t:t+h]``."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    t = x.shape[0]
    w = max(int(window_size), 1)
    h = max(int(horizon), 1)
    if target is None:
        y_src = x[:, 0]
    else:
        y_src = np.asarray(target, dtype=np.float64).reshape(-1)
    n = t - w - h + 1
    if n <= 0:
        empty_x = np.zeros((0, w, x.shape[1]) if not flatten else (0, w * x.shape[1]))
        empty_y = np.zeros((0, h) if h > 1 else (0,))
        return WindowBatch(X=empty_x, y=empty_y, indices=np.array([], dtype=np.int64))
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    idx: list[int] = []
    for i in range(n):
        sl = x[i : i + w]
        xs.append(sl.reshape(-1) if flatten else sl)
        y_sl = y_src[i + w : i + w + h]
        ys.append(y_sl if h > 1 else y_sl[0])
        idx.append(i + w - 1)
    return WindowBatch(
        X=np.stack(xs, axis=0),
        y=np.stack(ys, axis=0) if h > 1 else np.asarray(ys, dtype=np.float64),
        indices=np.asarray(idx, dtype=np.int64),
        metadata={"window_size": w, "horizon": h, "flatten": flatten},
    )


def recursive_path(
    last_window: np.ndarray,
    step_fn: Any,
    *,
    horizon: int,
) -> np.ndarray:
    """Generate a recursive multi-step path given a one-step predictor ``step_fn``."""
    w = np.asarray(last_window, dtype=np.float64).copy()
    if w.ndim == 1:
        w = w.reshape(-1, 1)
    preds: list[float] = []
    for _ in range(max(int(horizon), 1)):
        yhat = float(step_fn(w))
        preds.append(yhat)
        # roll window: append prediction on first feature channel
        nxt = np.concatenate([w[1:], w[-1:]], axis=0)
        nxt[-1, 0] = yhat
        w = nxt
    return np.asarray(preds, dtype=np.float64)

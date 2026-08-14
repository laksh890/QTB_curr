"""Signal correlation matrices across returns, positions, predictions, IC, drawdowns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

CorrKind = Literal["return", "position", "prediction", "ic", "drawdown"]


def _as_cols(
    data: Mapping[str, Any] | np.ndarray, names: Sequence[str] | None = None
) -> tuple[list[str], np.ndarray]:
    """Build ``(T, K)`` matrix of series."""
    if isinstance(data, Mapping):
        keys = list(data.keys()) if names is None else list(names)
        cols = []
        for k in keys:
            v = np.asarray(data[k], dtype=np.float64)
            if v.ndim > 1:
                # collapse panel to portfolio-style series
                v = np.nanmean(v, axis=1)
            cols.append(v.reshape(-1))
        n = max((c.size for c in cols), default=0)
        mat = np.full((n, len(cols)), np.nan, dtype=np.float64)
        for j, c in enumerate(cols):
            mat[: c.size, j] = c
        return keys, mat

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("data must be mapping or 2D array")
    keys = list(names) if names is not None else [f"s{i}" for i in range(arr.shape[1])]
    return keys, arr


def _corrcoef_pairwise(mat: np.ndarray) -> np.ndarray:
    k = mat.shape[1]
    out = np.eye(k, dtype=np.float64)
    for i in range(k):
        for j in range(i + 1, k):
            a, b = mat[:, i], mat[:, j]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 3 or np.std(a[m]) < 1e-15 or np.std(b[m]) < 1e-15:
                out[i, j] = out[j, i] = float("nan")
            else:
                c = float(np.corrcoef(a[m], b[m])[0, 1])
                out[i, j] = out[j, i] = c
    return out


def signal_correlation_matrix(
    data: Mapping[str, Any] | np.ndarray,
    *,
    names: Sequence[str] | None = None,
    kind: CorrKind = "prediction",
    method: str = "pearson",
) -> dict[str, Any]:
    """Correlation matrix across signals.

    ``data`` is typically a mapping of signal name → 1D series or ``(T, N)`` panel
    (panels are collapsed by cross-sectional mean for return/position/prediction).
    """
    keys, mat = _as_cols(data, names=names)
    if method == "spearman":
        ranked = np.full_like(mat, np.nan)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            m = np.isfinite(col)
            if m.sum() == 0:
                continue
            order = col[m].argsort().argsort().astype(np.float64)
            ranked[m, j] = order
            use = ranked
    else:
        use = mat

    # kind is metadata for caller; drawdown transforms series first
    if kind == "drawdown":
        use = _to_drawdown_series(use)

    corr = _corrcoef_pairwise(use)
    return {
        "name": "signal_correlation_matrix",
        "kind": kind,
        "method": method,
        "labels": keys,
        "matrix": corr.tolist(),
        "mean_abs_offdiag": _mean_abs_offdiag(corr),
        "max_abs_offdiag": _max_abs_offdiag(corr),
    }


def _to_drawdown_series(mat: np.ndarray) -> np.ndarray:
    out = np.full_like(mat, np.nan)
    for j in range(mat.shape[1]):
        x = mat[:, j]
        # treat as returns → equity → drawdown
        r = np.where(np.isfinite(x), x, 0.0)
        equity = np.cumprod(1.0 + r)
        peak = np.maximum.accumulate(equity)
        dd = equity / np.maximum(peak, 1e-12) - 1.0
        dd[~np.isfinite(x)] = np.nan
        out[:, j] = dd
    return out


def _mean_abs_offdiag(corr: np.ndarray) -> float:
    k = corr.shape[0]
    if k < 2:
        return 0.0
    mask = ~np.eye(k, dtype=bool)
    vals = np.abs(corr[mask])
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else float("nan")


def _max_abs_offdiag(corr: np.ndarray) -> float:
    k = corr.shape[0]
    if k < 2:
        return 0.0
    mask = ~np.eye(k, dtype=bool)
    vals = np.abs(corr[mask])
    vals = vals[np.isfinite(vals)]
    return float(np.max(vals)) if vals.size else float("nan")


def ic_correlation_matrix(
    ic_series: Mapping[str, Any] | np.ndarray,
    *,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    return signal_correlation_matrix(ic_series, names=names, kind="ic")


def return_correlation_matrix(
    returns: Mapping[str, Any] | np.ndarray,
    *,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    return signal_correlation_matrix(returns, names=names, kind="return")


def position_correlation_matrix(
    positions: Mapping[str, Any] | np.ndarray,
    *,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    return signal_correlation_matrix(positions, names=names, kind="position")


def prediction_correlation_matrix(
    predictions: Mapping[str, Any] | np.ndarray,
    *,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    return signal_correlation_matrix(predictions, names=names, kind="prediction")


def drawdown_correlation_matrix(
    returns: Mapping[str, Any] | np.ndarray,
    *,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    return signal_correlation_matrix(returns, names=names, kind="drawdown")


def correlation_penalty_vector(
    corr: Mapping[str, Any] | np.ndarray,
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    """Per-signal mean absolute correlation with others (penalty in [0, 1])."""
    if isinstance(corr, Mapping) and "matrix" in corr:
        mat = np.asarray(corr["matrix"], dtype=np.float64)
        labels = list(corr.get("labels", labels or []))
    else:
        mat = np.asarray(corr, dtype=np.float64)
    k = mat.shape[0]
    if labels is None or len(labels) != k:
        labels = [f"s{i}" for i in range(k)]
    out: dict[str, float] = {}
    for i, name in enumerate(labels):
        others = np.delete(np.abs(mat[i]), i)
        others = others[np.isfinite(others)]
        out[str(name)] = float(np.mean(others)) if others.size else 0.0
    return out

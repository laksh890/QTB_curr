"""Leakage-safe time-series transformations."""

from __future__ import annotations

from typing import Literal

import numpy as np

from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array


class TimeSeriesTransformer:
    """Fit/transform API with explicit temporal contracts."""

    def __init__(
        self,
        method: Literal[
            "log_return",
            "simple_return",
            "diff",
            "seasonal_diff",
            "zscore",
            "robust",
            "rank",
            "winsorize",
            "log",
        ] = "log_return",
        *,
        window: int = 64,
        period: int = 24,
        temporal_mode: Literal["rolling", "expanding", "training_only"] = "rolling",
        limits: tuple[float, float] = (0.01, 0.99),
    ) -> None:
        self.method = method
        self.window = window
        self.period = period
        self.temporal_mode = temporal_mode
        self.limits = limits
        self._center: float | None = None
        self._scale: float | None = None
        self._fitted = False

    def fit(self, x: np.ndarray | list[float]) -> TimeSeriesTransformer:
        arr = as_float_array(x)
        finite = arr[np.isfinite(arr)]
        if (
            self.method in {"zscore", "robust", "winsorize", "rank"}
            and self.temporal_mode == "training_only"
        ):
            if self.method == "zscore":
                self._center = float(np.mean(finite)) if finite.size else 0.0
                self._scale = float(np.std(finite)) if finite.size else 1.0
            elif self.method == "robust":
                self._center = float(np.median(finite)) if finite.size else 0.0
                q75, q25 = np.percentile(finite, [75, 25]) if finite.size else (1.0, 0.0)
                self._scale = float(q75 - q25) or 1.0
            else:
                self._center = 0.0
                self._scale = 1.0
        self._fitted = True
        return self

    def transform(self, x: np.ndarray | list[float]) -> np.ndarray:
        arr = as_float_array(x)
        if self.method == "log_return":
            return _log_returns(arr)
        if self.method == "simple_return":
            return _simple_returns(arr)
        if self.method == "diff":
            return _diff(arr, 1)
        if self.method == "seasonal_diff":
            return _diff(arr, self.period)
        if self.method == "log":
            return np.log(np.clip(arr, 1e-12, None))
        if self.method == "zscore":
            return _causal_zscore(arr, self.window, self.temporal_mode, self._center, self._scale)
        if self.method == "robust":
            return _causal_robust(arr, self.window, self.temporal_mode, self._center, self._scale)
        if self.method == "rank":
            return _causal_rank(arr, self.window)
        if self.method == "winsorize":
            return _causal_winsorize(arr, self.window, self.limits)
        return arr.copy()

    def fit_transform(self, x: np.ndarray | list[float]) -> np.ndarray:
        return self.fit(x).transform(x)

    def analyze(self, x: np.ndarray | list[float]) -> AnalysisResult:
        out = self.fit_transform(x)
        mode = {
            "rolling": TemporalMode.ROLLING,
            "expanding": TemporalMode.EXPANDING,
            "training_only": TemporalMode.TRAINING_ONLY,
        }.get(self.temporal_mode, TemporalMode.CAUSAL)
        return AnalysisResult(
            method=f"transform.{self.method}",
            value=out,
            window=self.window,
            parameters={"method": self.method, "period": self.period},
            temporal_mode=mode,
            metadata={"leakage_safe": True, "fitted": self._fitted},
        )


def log_returns(x: np.ndarray | list[float]) -> np.ndarray:
    return _log_returns(as_float_array(x))


def simple_returns(x: np.ndarray | list[float]) -> np.ndarray:
    return _simple_returns(as_float_array(x))


def differencing(x: np.ndarray | list[float], order: int = 1) -> np.ndarray:
    return _diff(as_float_array(x), order)


def normalize(
    x: np.ndarray | list[float],
    *,
    method: Literal["zscore", "robust", "minmax"] = "zscore",
    window: int = 64,
) -> np.ndarray:
    arr = as_float_array(x)
    if method == "zscore":
        return _causal_zscore(arr, window, "rolling", None, None)
    if method == "robust":
        return _causal_robust(arr, window, "rolling", None, None)
    # causal minmax
    out = np.full_like(arr, np.nan)
    for i in range(arr.size):
        chunk = arr[max(0, i - window + 1) : i + 1]
        lo, hi = np.nanmin(chunk), np.nanmax(chunk)
        out[i] = 0.0 if hi <= lo else (arr[i] - lo) / (hi - lo)
    return out


def rank_transform(x: np.ndarray | list[float], window: int = 64) -> np.ndarray:
    return _causal_rank(as_float_array(x), window)


def _log_returns(arr: np.ndarray) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    if arr.size < 2:
        return out
    out[1:] = np.diff(np.log(np.clip(arr, 1e-12, None)))
    return out


def _simple_returns(arr: np.ndarray) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    if arr.size < 2:
        return out
    prev = arr[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = np.where(np.abs(prev) > 1e-12, (arr[1:] - prev) / prev, np.nan)
    return out


def _diff(arr: np.ndarray, order: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    k = max(int(order), 1)
    if arr.size <= k:
        return out
    out[k:] = arr[k:] - arr[:-k]
    return out


def _causal_zscore(
    arr: np.ndarray,
    window: int,
    mode: str,
    center: float | None,
    scale: float | None,
) -> np.ndarray:
    if mode == "training_only" and center is not None and scale is not None:
        return (arr - center) / (scale if abs(scale) > 1e-12 else 1.0)
    out = np.full_like(arr, np.nan)
    w = max(int(window), 2)
    for i in range(arr.size):
        start = 0 if mode == "expanding" else max(0, i - w + 1)
        chunk = arr[start : i + 1]
        mu, sd = np.nanmean(chunk), np.nanstd(chunk)
        out[i] = (arr[i] - mu) / (sd if sd > 1e-12 else 1.0)
    return out


def _causal_robust(
    arr: np.ndarray,
    window: int,
    mode: str,
    center: float | None,
    scale: float | None,
) -> np.ndarray:
    if mode == "training_only" and center is not None and scale is not None:
        return (arr - center) / (scale if abs(scale) > 1e-12 else 1.0)
    out = np.full_like(arr, np.nan)
    w = max(int(window), 2)
    for i in range(arr.size):
        start = 0 if mode == "expanding" else max(0, i - w + 1)
        chunk = arr[start : i + 1]
        med = float(np.nanmedian(chunk))
        mad = float(np.nanmedian(np.abs(chunk - med))) * 1.4826
        out[i] = (arr[i] - med) / (mad if mad > 1e-12 else 1.0)
    return out


def _causal_rank(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    w = max(int(window), 2)
    for i in range(arr.size):
        chunk = arr[max(0, i - w + 1) : i + 1]
        # rank of last value among window
        out[i] = float(np.sum(chunk <= arr[i]) / chunk.size)
    return out


def _causal_winsorize(arr: np.ndarray, window: int, limits: tuple[float, float]) -> np.ndarray:
    out = arr.copy()
    w = max(int(window), 2)
    lo_q, hi_q = limits
    for i in range(arr.size):
        chunk = arr[max(0, i - w + 1) : i + 1]
        lo, hi = np.nanpercentile(chunk, [lo_q * 100, hi_q * 100])
        out[i] = float(np.clip(arr[i], lo, hi))
    return out

"""Causal normalization helpers (never full-dataset statistics)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def causal_rolling_zscore(series: pd.Series, *, window: int = 20) -> pd.Series:
    """Rolling z-score using only past+current observations (min_periods=window)."""
    w = max(int(window), 2)
    s = pd.Series(series, dtype=np.float64)
    mu = s.rolling(w, min_periods=w).mean()
    sd = s.rolling(w, min_periods=w).std(ddof=1)
    out = (s - mu) / sd.replace(0.0, np.nan)
    return out


def causal_rank(series: pd.Series, *, window: int = 20) -> pd.Series:
    """Rolling percentile rank in [0,1] using trailing window only."""
    w = max(int(window), 2)
    s = pd.Series(series, dtype=np.float64)

    def _rank(x: np.ndarray) -> float:
        if x.size < w or not np.isfinite(x[-1]):
            return float("nan")
        return float(np.mean(x <= x[-1]))

    return s.rolling(w, min_periods=w).apply(_rank, raw=True)


def causal_vol_normalize(series: pd.Series, *, window: int = 20) -> pd.Series:
    w = max(int(window), 2)
    s = pd.Series(series, dtype=np.float64)
    vol = s.rolling(w, min_periods=w).std(ddof=1)
    return s / vol.replace(0.0, np.nan)


def causal_zscore_expanding(series: pd.Series, *, min_periods: int = 20) -> pd.Series:
    """Expanding z-score (still causal — no future). Prefer rolling for live-like research."""
    s = pd.Series(series, dtype=np.float64)
    mu = s.expanding(min_periods=min_periods).mean()
    sd = s.expanding(min_periods=min_periods).std(ddof=1)
    return (s - mu) / sd.replace(0.0, np.nan)


__all__ = [
    "causal_rank",
    "causal_rolling_zscore",
    "causal_vol_normalize",
    "causal_zscore_expanding",
]

"""Reusable analytical features for the Feature Engineering Platform.

These are measurements — not alpha signals.
"""

from __future__ import annotations

import numpy as np

from iqrp.app.timeseries.autocorrelation.acf import acf
from iqrp.app.timeseries.base import AnalysisResult, TemporalMode, as_float_array
from iqrp.app.timeseries.change_points.pelt import pelt_detect
from iqrp.app.timeseries.decomposition.seasonal import seasonal_strength
from iqrp.app.timeseries.decomposition.stl import stl_decompose
from iqrp.app.timeseries.decomposition.trend import trend_strength
from iqrp.app.timeseries.nonlinear.entropy import shannon_entropy
from iqrp.app.timeseries.nonlinear.hurst import hurst_exponent
from iqrp.app.timeseries.nonlinear.permutation_entropy import permutation_entropy
from iqrp.app.timeseries.rolling import rolling_apply
from iqrp.app.timeseries.spectral.fft import dominant_frequencies
from iqrp.app.timeseries.stationarity.variance_ratio import variance_ratio


def trend_features(
    x: np.ndarray | list[float], *, period: int = 24, window: int = 64
) -> dict[str, float]:
    arr = as_float_array(x)
    dec = stl_decompose(arr, period=period)
    slope = float(np.polyfit(np.arange(arr.size), arr, 1)[0]) if arr.size >= 2 else 0.0
    return {
        "trend_strength": trend_strength(dec.trend, dec.residual),
        "trend_slope": slope,
        "rolling_slope": (
            float(
                rolling_apply(arr, window, lambda c: float(np.polyfit(np.arange(c.size), c, 1)[0]))[
                    -1
                ]
            )
            if arr.size >= window
            else slope
        ),
    }


def cycle_features(x: np.ndarray | list[float], *, period: int = 24) -> dict[str, float]:
    arr = as_float_array(x)
    dec = stl_decompose(arr, period=period)
    dom = dominant_frequencies(arr, top_k=1)
    freqs = dom.value if isinstance(dom.value, list) else []
    return {
        "seasonal_strength": seasonal_strength(dec.seasonal, dec.residual),
        "dominant_frequency": float(freqs[0]["frequency"]) if freqs else 0.0,
        "dominant_period": float(freqs[0]["period"]) if freqs else float(period),
        "cycle_strength": (
            float(freqs[0].get("amplitude", freqs[0].get("power", 0.0))) if freqs else 0.0
        ),
    }


def volatility_features(x: np.ndarray | list[float], *, window: int = 64) -> dict[str, float]:
    arr = as_float_array(x)
    r = np.diff(arr)
    if r.size < 2:
        return {"realized_vol": 0.0, "vol_of_vol": 0.0, "parkinson_proxy": 0.0}
    rv = float(np.std(r))
    roll = rolling_apply(r, max(window // 4, 5), lambda c: float(np.std(c)))
    return {
        "realized_vol": rv,
        "vol_of_vol": float(np.nanstd(roll)),
        "parkinson_proxy": float((np.nanmax(arr) - np.nanmin(arr)) / max(np.sqrt(arr.size), 1.0)),
    }


def entropy_features(x: np.ndarray | list[float]) -> dict[str, float]:
    arr = as_float_array(x)
    h = shannon_entropy(arr)
    pe = permutation_entropy(arr)
    return {
        "shannon_entropy": float(h.value) if isinstance(h.value, (int, float)) else float("nan"),
        "permutation_entropy": (
            float(pe.value) if isinstance(pe.value, (int, float)) else float("nan")
        ),
    }


def memory_features(x: np.ndarray | list[float]) -> dict[str, float]:
    arr = as_float_array(x)
    h = hurst_exponent(arr)
    vr = variance_ratio(arr, lags=2)
    ac = acf(arr, nlags=min(10, max(arr.size // 5, 1)))
    ac_vals = ac.value if isinstance(ac.value, (list, np.ndarray)) else []
    ac1 = float(ac_vals[1]) if len(ac_vals) > 1 else 0.0
    return {
        "hurst": float(h.value) if isinstance(h.value, (int, float)) else float("nan"),
        "variance_ratio": float(vr.value) if isinstance(vr.value, (int, float)) else float("nan"),
        "acf_lag1": ac1,
        "mean_reversion_score": (
            float(
                max(
                    0.0,
                    1.0 - abs(float(h.value) if isinstance(h.value, (int, float)) else 0.5) * 2 + 1,
                )
            )
            if isinstance(h.value, (int, float))
            else float("nan")
        ),
    }


def spectral_features(x: np.ndarray | list[float]) -> dict[str, float]:
    return cycle_features(x)


def change_point_proximity(x: np.ndarray | list[float], *, min_size: int = 10) -> dict[str, float]:
    arr = as_float_array(x)
    cp = pelt_detect(arr, min_size=min_size)
    if not cp.indices:
        return {"n_change_points": 0.0, "nearest_cp_distance": float(arr.size), "cp_density": 0.0}
    last = cp.indices[-1]
    return {
        "n_change_points": float(len(cp.indices)),
        "nearest_cp_distance": float(arr.size - 1 - last),
        "cp_density": float(len(cp.indices) / max(arr.size, 1)),
    }


def extract_features(
    x: np.ndarray | list[float],
    *,
    period: int = 24,
    window: int = 64,
    include_entropy: bool = True,
    include_hurst: bool = True,
    include_spectral: bool = True,
) -> AnalysisResult:
    feats: dict[str, float] = {}
    feats.update(trend_features(x, period=period, window=window))
    feats.update(volatility_features(x, window=window))
    feats.update(change_point_proximity(x))
    if include_spectral:
        feats.update(cycle_features(x, period=period))
    if include_entropy:
        feats.update(entropy_features(x))
    if include_hurst:
        feats.update(memory_features(x))
    return AnalysisResult(
        method="features.extract",
        value=feats,
        window=window,
        parameters={"period": period, "window": window},
        temporal_mode=TemporalMode.FULL_SAMPLE,
        metadata={"n_features": len(feats), "note": "analytical measurements, not trading signals"},
    )

"""Forecast → LONG/SHORT/FLAT signal mapping (no model logic)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.adapters.types import (
    OutputMappingKind,
    SignalMappingConfig,
)
from iqrp.app.forecasting.base.forecast import Forecast


def _sign_thresh(x: np.ndarray, long_thr: float, short_thr: float, *, allow_short: bool) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    out = np.where(x > long_thr, 1.0, out)
    if allow_short:
        out = np.where(x < short_thr, -1.0, out)
    else:
        out = np.where(x < short_thr, 0.0, out)
    out = np.where(np.isfinite(x), out, np.nan)
    return out


def map_values_to_signal(
    values: np.ndarray | pd.Series,
    mapping: SignalMappingConfig,
    *,
    probabilities: np.ndarray | None = None,
) -> np.ndarray:
    """Convert numeric forecast path / series into {-1,0,1} (or continuous)."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    kind = mapping.kind

    if kind == OutputMappingKind.CONTINUOUS_PASSTHROUGH:
        return x

    if kind == OutputMappingKind.RETURN_THRESHOLD:
        return _sign_thresh(
            x, mapping.long_threshold, mapping.short_threshold, allow_short=mapping.allow_short
        )

    if kind == OutputMappingKind.PROBABILITY_UP:
        if probabilities is not None:
            p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
            if p.ndim > 1:
                # assume column 1 is up / last class
                p = p[:, -1] if p.shape[1] > 1 else p.reshape(-1)
        else:
            # interpret values as probabilities
            p = x
        out = np.zeros_like(p, dtype=np.float64)
        out = np.where(p >= mapping.long_prob, 1.0, out)
        if mapping.allow_short:
            out = np.where(p <= mapping.short_prob, -1.0, out)
        out = np.where(np.isfinite(p), out, np.nan)
        return out

    if kind in {OutputMappingKind.VOLATILITY_EXPANSION, OutputMappingKind.VOLATILITY_CONTRACTION}:
        s = pd.Series(x)
        med = s.rolling(mapping.vol_lookback, min_periods=max(mapping.vol_lookback // 2, 2)).median()
        mad = (s - med).abs().rolling(mapping.vol_lookback, min_periods=max(mapping.vol_lookback // 2, 2)).median()
        z = (s - med) / (mad.replace(0, np.nan) + 1e-12)
        z_arr = z.to_numpy(dtype=np.float64)
        if kind == OutputMappingKind.VOLATILITY_EXPANSION:
            # high vol → short risk-off tilt (research diagnostic)
            return _sign_thresh(
                z_arr, mapping.vol_z_threshold, -mapping.vol_z_threshold, allow_short=mapping.allow_short
            ) * -1.0  # expansion → SHORT, contraction → LONG
        # contraction mapping: high z → LONG (fade vol)
        return _sign_thresh(
            -z_arr, mapping.vol_z_threshold, -mapping.vol_z_threshold, allow_short=mapping.allow_short
        )

    if kind == OutputMappingKind.REGIME_LABEL_MAP:
        # values expected as integer/string codes already handled in regime_adapter
        return x

    raise ValueError(f"unsupported mapping kind: {kind}")


def forecast_to_signal_series(
    forecast: Forecast | np.ndarray,
    index: pd.Index,
    mapping: SignalMappingConfig,
    *,
    align: str = "tail",
) -> pd.Series:
    """Map a Forecast object (often length=horizon) or array onto a frame index.

    For multi-step Forecast.values of length H, by default the point forecast is
    written only on the last index (inference timestamp), elsewhere NaN — caller
    should use rolling/OOS builders for full paths.
    """
    if isinstance(forecast, Forecast):
        vals = np.asarray(forecast.values, dtype=np.float64).reshape(-1)
        probs = forecast.probabilities
    else:
        vals = np.asarray(forecast, dtype=np.float64).reshape(-1)
        probs = None

    mapped = map_values_to_signal(vals, mapping, probabilities=probs)
    out = np.full(len(index), np.nan, dtype=np.float64)
    if mapped.size == len(index):
        out = mapped
    elif mapped.size == 1 and len(index) > 0:
        out[-1] = mapped[0]
    elif align == "tail" and mapped.size <= len(index):
        out[-mapped.size :] = mapped
    else:
        n = min(mapped.size, len(index))
        out[:n] = mapped[:n]
    return pd.Series(out, index=index, dtype=np.float64)


def metadata_bundle(
    *,
    source_model: str,
    model_version: str,
    forecast_timestamp: Any,
    signal_timestamp: Any,
    source_timeframe: str,
    execution_timeframe: str,
    lookback: int,
    horizon: int,
    threshold_config: dict[str, Any],
    configuration_id: str,
) -> dict[str, Any]:
    return {
        "source_model": source_model,
        "model_version": model_version,
        "forecast_timestamp": str(forecast_timestamp),
        "signal_timestamp": str(signal_timestamp),
        "source_timeframe": source_timeframe,
        "execution_timeframe": execution_timeframe,
        "lookback": int(lookback),
        "horizon": int(horizon),
        "threshold": threshold_config,
        "configuration_id": configuration_id,
        "disclaimer": "Research wiring validation only — not a profitability claim.",
    }


__all__ = [
    "forecast_to_signal_series",
    "map_values_to_signal",
    "metadata_bundle",
]

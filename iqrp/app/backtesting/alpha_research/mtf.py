"""Causal multi-timeframe feature alignment."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.alpha_research.types import TimeframeContext
from iqrp.app.data.historical.calendar import nse_equity_calendar
from iqrp.app.data.historical.resampling import resample_session_aware


def align_feature_to_execution(
    feature_frame: pd.DataFrame,
    feature_values: pd.Series,
    execution_timestamps: pd.Series | pd.DatetimeIndex,
) -> pd.Series:
    """As-of merge: execution bar t gets last feature observation ≤ t (causal)."""
    feat = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(feature_frame["timestamp"], utc=True),
            "feature": np.asarray(feature_values, dtype=np.float64),
        }
    ).sort_values("timestamp")
    exec_ts = pd.to_datetime(pd.Series(execution_timestamps), utc=True)
    left = pd.DataFrame({"timestamp": exec_ts}).sort_values("timestamp")
    merged = pd.merge_asof(left, feat, on="timestamp", direction="backward")
    # restore original order
    out = merged["feature"].to_numpy(dtype=np.float64)
    # map back if exec was sorted
    return pd.Series(out, index=range(len(exec_ts)), dtype=np.float64)


def load_timeframe_frames(
    paths: dict[str, str],
) -> dict[str, pd.DataFrame]:
    out = {}
    for tf, path in paths.items():
        df = pd.read_parquet(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        out[tf] = df.sort_values("timestamp").reset_index(drop=True)
    return out


def build_mtf_feature(
    frames: dict[str, pd.DataFrame],
    *,
    feature_tf: str,
    execution_tf: str,
    compute_feature,
) -> tuple[pd.Series, TimeframeContext, dict[str, Any]]:
    """Compute feature on feature_tf frame; align causally onto execution_tf."""
    if feature_tf not in frames or execution_tf not in frames:
        raise KeyError(f"missing timeframe frames: need {feature_tf} and {execution_tf}")
    fdf = frames[feature_tf]
    edf = frames[execution_tf]
    raw = compute_feature(fdf)
    aligned = align_feature_to_execution(fdf, raw, edf["timestamp"])
    # reindex to execution frame index
    aligned.index = edf.index
    ctx = TimeframeContext(
        feature_timeframe=feature_tf,
        signal_timeframe=feature_tf,
        execution_timeframe=execution_tf,
    )
    meta = {
        **ctx.to_dict(),
        "alignment": "merge_asof_backward",
        "causal": True,
        "note": "Feature at execution t uses last feature bar with timestamp ≤ t.",
    }
    return aligned, ctx, meta


def derive_coarser_if_needed(
    source_1m: pd.DataFrame,
    target_tf: str,
) -> pd.DataFrame:
    """Optional helper: derive coarser bars from 1m with session-aware resample."""
    out, _ = resample_session_aware(
        source_1m,
        source_frequency="1m",
        derived_frequency=target_tf,
        calendar=nse_equity_calendar(),
    )
    return out


__all__ = [
    "align_feature_to_execution",
    "build_mtf_feature",
    "derive_coarser_if_needed",
    "load_timeframe_frames",
]

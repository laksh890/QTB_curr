"""Programmatic temporal firewall audit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.backtesting.frozen_2025_holdout.protocol import DISCLAIMER, HOLDOUT_START, RESEARCH_END


def audit_firewall(
    *,
    research_frames: dict[str, pd.DataFrame],
    holdout_frames: dict[str, pd.DataFrame],
    concat_frames: dict[str, pd.DataFrame],
    research_end: str = RESEARCH_END,
    holdout_start: str = HOLDOUT_START,
) -> dict[str, Any]:
    """Hard-stop conditions encoded as pass/fail checks."""
    re = pd.Timestamp(research_end)
    hs = pd.Timestamp(holdout_start)
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    for tf, fr in research_frames.items():
        ts = pd.to_datetime(fr["timestamp"], utc=True)
        ok = bool(ts.max() <= re) and bool((ts > re).sum() == 0)
        checks[f"research_{tf}_no_holdout_rows"] = ok
        details[f"research_{tf}_max"] = str(ts.max()) if len(ts) else None

    for tf, fr in holdout_frames.items():
        ts = pd.to_datetime(fr["timestamp"], utc=True)
        ok = bool(ts.min() >= hs) and bool((ts < hs).sum() == 0) and bool((ts <= re).sum() == 0)
        checks[f"holdout_{tf}_no_research_rows"] = ok
        details[f"holdout_{tf}_min"] = str(ts.min()) if len(ts) else None
        details[f"holdout_{tf}_max"] = str(ts.max()) if len(ts) else None

    for tf, fr in concat_frames.items():
        ts = pd.to_datetime(fr["timestamp"], utc=True)
        # monotonic
        checks[f"concat_{tf}_monotonic"] = bool(ts.is_monotonic_increasing)
        # research prefix then holdout
        research_mask = ts <= re
        holdout_mask = ts >= hs
        if research_mask.any() and holdout_mask.any():
            last_r = int(np.where(research_mask.to_numpy())[0].max())
            first_h = int(np.where(holdout_mask.to_numpy())[0].min())
            checks[f"concat_{tf}_research_before_holdout"] = last_r < first_h
        else:
            checks[f"concat_{tf}_research_before_holdout"] = False

    # No overlap of research and holdout timestamps
    for tf in research_frames:
        if tf not in holdout_frames:
            continue
        rset = set(pd.to_datetime(research_frames[tf]["timestamp"], utc=True).astype("int64"))
        hset = set(pd.to_datetime(holdout_frames[tf]["timestamp"], utc=True).astype("int64"))
        checks[f"no_timestamp_intersection_{tf}"] = len(rset & hset) == 0

    passed = all(checks.values())
    return {
        "disclaimer": DISCLAIMER,
        "status": "PASS" if passed else "FAIL",
        "hard_stop": not passed,
        "checks": checks,
        "details": details,
        "policy": {
            "research_may_consume": "timestamps <= 2024-12-31 only",
            "holdout_may_consume": "2025 calendar only for evaluation",
            "forbidden": [
                "holdout in feature fit",
                "holdout in scaler/cov/vol estimates used for selection",
                "holdout in parameter/threshold selection",
            ],
        },
    }


__all__ = ["audit_firewall"]

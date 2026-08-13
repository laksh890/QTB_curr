"""Model drift / residual stability monitor."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_returns


def model_drift(
    residuals: Any,
    *,
    reference_window: int = 60,
    test_window: int = 20,
) -> RiskMeasure:
    """Detect residual mean/vol shift between reference and recent windows.

    Uses only trailing data. Score is a simple combined z-style drift metric;
    higher means more drift.
    """
    r = as_returns(residuals)
    ref_w = max(int(reference_window), 2)
    test_w = max(int(test_window), 1)
    need = ref_w + test_w
    if r.size < need:
        return RiskMeasure(
            name="model_drift",
            value=0.0,
            unit="score",
            method="residual_window_shift",
            parameters={
                "n_obs": int(r.size),
                "reference_window": ref_w,
                "test_window": test_w,
                "insufficient_data": True,
            },
        )

    recent = r[-test_w:]
    reference = r[-(need):-test_w]
    mu_ref = float(np.mean(reference))
    mu_new = float(np.mean(recent))
    sig_ref = float(np.std(reference, ddof=1)) if reference.size > 1 else 0.0
    sig_new = float(np.std(recent, ddof=1)) if recent.size > 1 else 0.0

    mean_shift = abs(mu_new - mu_ref) / max(sig_ref, 1e-12)
    vol_shift = abs(sig_new - sig_ref) / max(sig_ref, 1e-12)
    score = float(mean_shift + 0.5 * vol_shift)

    return RiskMeasure(
        name="model_drift",
        value=score,
        unit="score",
        method="residual_window_shift",
        parameters={
            "n_obs": int(r.size),
            "reference_window": ref_w,
            "test_window": test_w,
            "mean_ref": mu_ref,
            "mean_recent": mu_new,
            "vol_ref": sig_ref,
            "vol_recent": sig_new,
            "mean_shift": mean_shift,
            "vol_shift": vol_shift,
        },
    )

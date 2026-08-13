"""Cross-model disagreement monitor."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure


def model_disagreement(
    model_forecasts: Any,
    *,
    axis: int = 0,
) -> RiskMeasure:
    """Dispersion across models at the latest common observation.

    ``model_forecasts`` shape (M, T) or (T, M) — dispersion measured as
    cross-sectional std of the last column/row of finite values.
    """
    arr = np.asarray(model_forecasts, dtype=np.float64)
    if arr.ndim == 1:
        vals = arr[np.isfinite(arr)]
        disp = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        n_models = int(vals.size)
    elif arr.ndim == 2:
        if axis == 0:
            # models x time
            col = arr[:, -1]
        else:
            col = arr[-1, :]
        vals = col[np.isfinite(col)]
        disp = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        n_models = int(vals.size)
    else:
        raise ValueError("model_forecasts must be 1-D or 2-D")

    range_ = float(np.ptp(vals)) if vals.size else 0.0
    return RiskMeasure(
        name="model_disagreement",
        value=disp,
        unit="dispersion",
        method="cross_model_std",
        parameters={"n_models": n_models, "range": range_, "axis": int(axis)},
    )

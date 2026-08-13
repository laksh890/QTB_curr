"""Overnight / open gap risk estimation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from iqrp.app.risk.base import RiskMeasure, as_returns


def gap_risk(
    overnight_returns: Any,
    *,
    confidence: float = 0.95,
    method: str = "historical",
) -> RiskMeasure:
    """Estimate overnight jump (gap) loss at a given confidence.

    ``overnight_returns`` should be open-to-previous-close (or similar) gaps —
    caller supplies the series; this function does not peek at future data.

    Returns a positive loss magnitude (absolute value of the left-tail quantile).
    """
    r = as_returns(overnight_returns)
    conf = float(confidence)
    if conf not in (0.90, 0.95, 0.99):
        conf = float(np.clip(conf, 0.5, 0.999))

    if r.size == 0:
        return RiskMeasure(
            name="gap_risk",
            value=0.0,
            unit="return",
            confidence=conf,
            method=method,
            parameters={"n_obs": 0},
        )

    alpha = 1.0 - conf
    if method == "parametric":
        mu = float(np.mean(r))
        sigma = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
        q = float(stats.norm.ppf(alpha, loc=mu, scale=max(sigma, 1e-12)))
        loss = float(max(-q, 0.0))
        used_method = "parametric"
        params: dict[str, Any] = {"mu": mu, "sigma": sigma, "n_obs": int(r.size), "quantile": q}
    else:
        q = float(np.quantile(r, alpha))
        loss = float(max(-q, 0.0))
        used_method = "historical"
        params = {
            "n_obs": int(r.size),
            "mean_gap": float(np.mean(r)),
            "std_gap": float(np.std(r, ddof=1)) if r.size > 1 else 0.0,
            "max_down_gap": float(abs(min(np.min(r), 0.0))),
            "max_up_gap": float(max(np.max(r), 0.0)),
            "quantile": q,
        }

    return RiskMeasure(
        name="gap_risk",
        value=loss,
        unit="return",
        confidence=conf,
        method=used_method,
        parameters=params,
    )

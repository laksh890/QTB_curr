"""Empirical tail dependence coefficient."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_returns


def empirical_tail_dependence(
    x: Any,
    y: Any,
    *,
    quantile: float = 0.05,
    tail: str = "lower",
) -> RiskMeasure:
    """Empirical tail dependence: P(Y in tail | X in tail).

    For lower tail uses the ``quantile`` empirical threshold;
    for upper tail uses ``1 - quantile``.
    """
    a = as_returns(x)
    b = as_returns(y)
    n = int(min(a.size, b.size))
    q = float(np.clip(quantile, 1e-4, 0.5))
    if n < 10:
        return RiskMeasure(
            name="empirical_tail_dependence",
            value=0.0,
            unit="probability",
            method="empirical",
            parameters={"quantile": q, "tail": tail, "n_obs": n},
        )

    a = a[-n:]
    b = b[-n:]
    side = str(tail).lower()
    if side == "upper":
        thr_a = float(np.quantile(a, 1.0 - q))
        thr_b = float(np.quantile(b, 1.0 - q))
        mask_a = a >= thr_a
        joint = np.sum(mask_a & (b >= thr_b))
    else:
        thr_a = float(np.quantile(a, q))
        thr_b = float(np.quantile(b, q))
        mask_a = a <= thr_a
        joint = np.sum(mask_a & (b <= thr_b))
        side = "lower"

    n_tail = int(np.sum(mask_a))
    value = float(joint / n_tail) if n_tail > 0 else 0.0

    return RiskMeasure(
        name="empirical_tail_dependence",
        value=value,
        unit="probability",
        method="empirical",
        parameters={
            "quantile": q,
            "tail": side,
            "n_obs": n,
            "n_tail": n_tail,
            "threshold_x": thr_a,
            "threshold_y": thr_b,
        },
    )

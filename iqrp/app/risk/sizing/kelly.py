"""Kelly criterion sizing with hard cap — never returns unbounded Kelly."""

from __future__ import annotations

import numpy as np

from iqrp.app.risk.base import RiskMeasure


def kelly_fraction(
    *,
    edge: float,
    odds: float = 1.0,
    win_prob: float | None = None,
    variance: float | None = None,
    max_kelly: float = 0.5,
) -> RiskMeasure:
    """Capped Kelly fraction.

    Forms
    -----
    - Binary: f* = p - (1-p)/b  when ``win_prob`` and ``odds`` (b) provided
    - Continuous: f* = edge / variance when ``variance`` provided
    - Fallback: f* = edge / odds when only edge/odds given

    Always clipped to [0, max_kelly]. Never returns raw unbounded Kelly.
    """
    cap = max(float(max_kelly), 0.0)
    raw: float

    if win_prob is not None:
        p = float(np.clip(win_prob, 0.0, 1.0))
        b = max(float(odds), 1e-12)
        raw = p - (1.0 - p) / b
    elif variance is not None:
        var = max(float(variance), 1e-12)
        raw = float(edge) / var
    else:
        b = max(float(odds), 1e-12)
        raw = float(edge) / b

    if not np.isfinite(raw):
        raw = 0.0

    capped = float(np.clip(raw, 0.0, cap))
    return RiskMeasure(
        name="kelly_fraction",
        value=capped,
        unit="fraction",
        method="kelly_capped",
        parameters={
            "edge": float(edge),
            "odds": float(odds),
            "win_prob": win_prob,
            "variance": variance,
            "max_kelly": cap,
            "raw_kelly": float(raw),
            "capped": True,
        },
    )

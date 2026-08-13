"""Historical stress using caller-supplied event masks/windows — no hard-coded events."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.risk.base import RiskMeasure, as_returns, as_weights


def historical_stress(
    returns: Any,
    *,
    event_mask: Any | None = None,
    event_window: Any | None = None,
    weights: Any | None = None,
) -> dict[str, Any]:
    """Replay historical returns over an event window or boolean mask.

    Parameters
    ----------
    returns :
        (T,) portfolio returns or (T, N) asset returns.
    event_mask :
        Boolean mask of length T selecting stress observations.
    event_window :
        Index array / slice bounds selecting stress rows. Accepted forms:
        - 1-D index array
        - (start, end) inclusive/exclusive pair as length-2 sequence
    weights :
        Optional weights if ``returns`` is (T, N); PnL = r @ w per day.

    Notes
    -----
    No hard-coded crisis dates — the caller must supply the event definition.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim == 1:
        port = as_returns(r)
        asset_mode = False
    elif r.ndim == 2:
        asset_mode = True
        t, n = r.shape
        if weights is None:
            w = np.full(n, 1.0 / max(n, 1))
        else:
            w = as_weights(weights, n=n)
        port = r @ w
        port = port[np.isfinite(port)]
        # Keep alignment with original rows for masking
        port_full = np.nan_to_num(r @ w, nan=0.0)
    else:
        raise ValueError("returns must be 1-D or 2-D")

    if not asset_mode:
        port_full = np.asarray(returns, dtype=np.float64).reshape(-1)
        port_full = np.nan_to_num(port_full, nan=0.0)

    t = port_full.size
    selected = np.zeros(t, dtype=bool)

    if event_mask is not None:
        mask = np.asarray(event_mask, dtype=bool).reshape(-1)
        if mask.size != t:
            raise ValueError(f"event_mask length {mask.size} != returns length {t}")
        selected |= mask

    if event_window is not None:
        ew = np.asarray(event_window).reshape(-1)
        if ew.size == 2 and np.issubdtype(ew.dtype, np.number):
            start, end = int(ew[0]), int(ew[1])
            start = max(start, 0)
            end = min(end, t)
            selected[start:end] = True
        else:
            idx = ew.astype(int)
            idx = idx[(idx >= 0) & (idx < t)]
            selected[idx] = True

    if not np.any(selected):
        stressed = np.zeros(0, dtype=np.float64)
    else:
        stressed = port_full[selected]

    if stressed.size == 0:
        cumulative = 0.0
        worst = 0.0
        mean_ret = 0.0
    else:
        cumulative = float(np.prod(1.0 + stressed) - 1.0)
        wealth = np.cumprod(1.0 + stressed)
        peak = np.maximum.accumulate(wealth)
        dd = 1.0 - wealth / np.maximum(peak, 1e-12)
        worst = float(np.max(dd)) if dd.size else 0.0
        mean_ret = float(np.mean(stressed))

    loss = float(max(-cumulative, 0.0))

    return {
        "name": "historical_stress",
        "n_event_days": int(np.sum(selected)),
        "cumulative_return": cumulative,
        "loss": loss,
        "worst_drawdown": worst,
        "mean_return": mean_ret,
        "min_return": float(np.min(stressed)) if stressed.size else 0.0,
        "measures": {
            "stress_loss": RiskMeasure(
                name="historical_stress_loss",
                value=loss,
                unit="return",
                method="historical_event",
                parameters={"n_event_days": int(np.sum(selected))},
            ).to_dict(),
        },
    }

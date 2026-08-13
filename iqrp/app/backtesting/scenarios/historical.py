"""Historical scenario windows — user-defined only (no hard-coded crises)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqrp.app.backtesting.performance.drawdown import max_drawdown
from iqrp.app.backtesting.performance.returns import as_returns, total_return
from iqrp.app.backtesting.performance.risk_adjusted import sharpe_ratio

__all__ = ["HistoricalScenario", "run_historical_scenario", "slice_window"]


@dataclass
class HistoricalScenario:
    """User-defined historical scenario specification.

    No crisis calendars are embedded — callers must supply start/end (or mask).
    """

    name: str
    start: int | None = None
    end: int | None = None
    mask: Any | None = None
    assets: list[str] | None = None
    market_conditions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def slice_window(
    returns: Any,
    *,
    start: int | None = None,
    end: int | None = None,
    mask: Any | None = None,
) -> np.ndarray:
    """Extract a return window from indices or a boolean mask."""
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim == 0:
        raise ValueError("returns must be at least 1-D")
    t = r.shape[0]
    selected = np.zeros(t, dtype=bool)

    if mask is not None:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        if m.size != t:
            raise ValueError(f"mask length {m.size} != returns length {t}")
        selected |= m

    if start is not None or end is not None:
        s = 0 if start is None else max(int(start), 0)
        e = t if end is None else min(int(end), t)
        if s > e:
            raise ValueError(f"invalid window start={s} end={e}")
        selected[s:e] = True

    if not np.any(selected) and start is None and end is None and mask is None:
        selected[:] = True

    return r[selected]


def run_historical_scenario(
    returns: Any,
    scenario: HistoricalScenario | dict[str, Any],
    *,
    weights: Any | None = None,
    periods_per_year: float = 252.0,
) -> dict[str, Any]:
    """Replay returns over a user-defined historical window.

    ``returns`` may be (T,) portfolio or (T, N) asset returns. With asset
    returns, optional ``weights`` produce portfolio PnL.
    """
    if isinstance(scenario, dict):
        scenario = HistoricalScenario(
            name=str(scenario.get("name", "historical")),
            start=scenario.get("start"),
            end=scenario.get("end"),
            mask=scenario.get("mask"),
            assets=scenario.get("assets"),
            market_conditions=dict(scenario.get("market_conditions") or {}),
            metadata=dict(scenario.get("metadata") or {}),
        )

    raw = np.asarray(returns, dtype=np.float64)
    if raw.ndim == 2:
        n = raw.shape[1]
        if weights is None:
            w = np.full(n, 1.0 / max(n, 1))
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.size != n:
                raise ValueError("weights length must match assets")
        port_full = raw @ w
    else:
        port_full = as_returns(raw)

    window = slice_window(
        port_full, start=scenario.start, end=scenario.end, mask=scenario.mask
    )
    window = window[np.isfinite(window)]

    return {
        "name": scenario.name,
        "kind": "historical",
        "n_obs": int(window.size),
        "start": scenario.start,
        "end": scenario.end,
        "assets": scenario.assets,
        "market_conditions": dict(scenario.market_conditions),
        "returns": window,
        "total_return": total_return(window),
        "max_drawdown": max_drawdown(window),
        "sharpe": sharpe_ratio(window, periods_per_year=periods_per_year),
        "mean": float(np.mean(window)) if window.size else 0.0,
        "volatility": float(np.std(window, ddof=1)) if window.size > 1 else 0.0,
        "metadata": dict(scenario.metadata),
    }

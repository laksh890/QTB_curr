"""Market event injectors."""

from __future__ import annotations

import numpy as np

from iqrp.app.simulation.events.exchange_outage import inject_exchange_outages
from iqrp.app.simulation.events.flash_crash import inject_flash_crashes
from iqrp.app.simulation.events.news import (
    inject_gap_opens,
    inject_momentum_bursts,
    inject_news_shocks,
    inject_slow_trends,
)
from iqrp.app.simulation.events.volatility_spike import (
    inject_liquidity_collapse,
    inject_volatility_spikes,
)


def apply_event_suite(
    prices: np.ndarray,
    volumes: np.ndarray,
    volatility: np.ndarray,
    spreads_bps: np.ndarray,
    *,
    rng: np.random.Generator,
    flash_crash_prob: float = 0.002,
    news_shock_prob: float = 0.01,
    gap_open_prob: float = 0.005,
    liquidity_collapse_prob: float = 0.003,
    outage_prob: float = 0.001,
    vol_spike_prob: float = 0.008,
    momentum_burst_prob: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Apply the full event suite; returns updated series + event masks."""
    masks: dict[str, np.ndarray] = {}
    p, m = inject_flash_crashes(prices, probability=flash_crash_prob, rng=rng)
    masks["flash_crash"] = m
    p, m = inject_news_shocks(p, probability=news_shock_prob, rng=rng)
    masks["news"] = m
    p, m = inject_gap_opens(p, probability=gap_open_prob, rng=rng)
    masks["gap_open"] = m
    p, m = inject_momentum_bursts(p, probability=momentum_burst_prob, rng=rng)
    masks["momentum_burst"] = m
    p, m = inject_slow_trends(p, rng=rng)
    masks["slow_trend"] = m
    p, vol, m = inject_volatility_spikes(p, volatility, probability=vol_spike_prob, rng=rng)
    masks["vol_spike"] = m
    spreads, volumes, m = inject_liquidity_collapse(
        spreads_bps, volumes, probability=liquidity_collapse_prob, rng=rng
    )
    masks["liquidity_collapse"] = m
    p, volumes, m = inject_exchange_outages(p, volumes, probability=outage_prob, rng=rng)
    masks["exchange_outage"] = m
    return p, volumes, vol, spreads, masks


__all__ = [
    "apply_event_suite",
    "inject_exchange_outages",
    "inject_flash_crashes",
    "inject_gap_opens",
    "inject_liquidity_collapse",
    "inject_momentum_bursts",
    "inject_news_shocks",
    "inject_slow_trends",
    "inject_volatility_spikes",
]

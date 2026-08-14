"""Adaptive execution reacting to spread, volatility, liquidity, and fill rate."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.execution.algorithms.base import (
    URGENCY_LIMIT_AGGRESSION,
    ChildSlice,
    ExecutionAlgorithm,
    MarketContext,
    approved_quantity,
    context_float,
    context_side,
    limit_hint,
    n_slices_for_urgency,
    redistribute_to_parent,
    schedule_offsets,
    urgency_from_context,
)
from iqrp.app.execution.types import Urgency


class AdaptiveAlgorithm(ExecutionAlgorithm):
    """Dynamically reshape schedule from live market / execution feedback.

    Responds to:
    - spread (widen → slow / more passive)
    - volume / liquidity (thin → smaller slices)
    - volatility (spike → slow unless urgency HIGH+)
    - order-book conditions (imbalance)
    - fill rate (poor → more aggressive limits, not more qty)
    - market impact / price movement / execution progress
    """

    name = "adaptive"

    def __init__(
        self,
        *,
        n_slices: int = 10,
        horizon_seconds: float = 300.0,
        base_participation: float = 0.10,
        default_urgency: Urgency | str = Urgency.NORMAL,
    ) -> None:
        super().__init__(default_urgency=default_urgency)
        self.n_slices = max(int(n_slices), 1)
        self.horizon_seconds = max(float(horizon_seconds), 0.0)
        self.base_participation = max(float(base_participation), 0.0)

    def plan(
        self,
        parent_qty: float,
        market_context: MarketContext | None = None,
    ) -> list[ChildSlice]:
        ctx: dict[str, Any] = dict(market_context or {})
        approved = approved_quantity(parent_qty, ctx)
        if approved <= 0.0:
            return []

        urg = urgency_from_context(ctx, self.default_urgency)
        n = n_slices_for_urgency(int(ctx.get("n_slices", self.n_slices)), urg)
        horizon = context_float(ctx, "horizon_seconds", self.horizon_seconds)
        mid = context_float(ctx, "mid", context_float(ctx, "price", 100.0))
        spread = context_float(ctx, "spread", 0.0)
        vol = context_float(ctx, "volatility", 0.02)
        adv = context_float(ctx, "adv", 1e6)
        side = context_side(ctx)

        # --- Market regime factors (multiplicative, clipped) ---
        spread_bps = (spread / max(mid, 1e-12)) * 1e4
        # Reference "normal" spread ~ 5 bps
        spread_factor = 1.0 / (1.0 + max(spread_bps - 5.0, 0.0) / 25.0)

        vol_ref = float(ctx.get("vol_ref", 0.02))
        vol_ratio = vol / max(vol_ref, 1e-8)
        if urg in {Urgency.HIGH, Urgency.CRITICAL}:
            # Under high urgency, do not slow as much on vol spikes
            vol_factor = 1.0 / (1.0 + max(vol_ratio - 1.0, 0.0) * 0.25)
        else:
            vol_factor = 1.0 / (1.0 + max(vol_ratio - 1.0, 0.0) * 0.75)

        liquidity = float(ctx.get("liquidity", ctx.get("liquidity_score", 1.0)))
        liquidity = min(max(liquidity, 0.05), 2.0)

        fill_rate = float(ctx.get("fill_rate", ctx.get("realized_fill_rate", 1.0)))
        fill_rate = min(max(fill_rate, 0.05), 1.5)

        # Order book imbalance: positive = more bid liquidity (good for sells)
        imbalance = float(ctx.get("book_imbalance", ctx.get("imbalance", 0.0)))
        imbalance = min(max(imbalance, -1.0), 1.0)
        if side == "buy":
            # Prefer ask-side liquidity → negative imbalance helps buys
            imb_factor = 1.0 + 0.25 * (-imbalance)
        else:
            imb_factor = 1.0 + 0.25 * imbalance

        # Execution progress: remaining fraction of parent
        filled = float(ctx.get("filled_quantity", ctx.get("executed_qty", 0.0)))
        progress = min(max(filled / max(approved + filled, 1e-12), 0.0), 1.0)
        remaining_frac = 1.0 - progress
        # Time progress
        elapsed = float(ctx.get("elapsed_seconds", 0.0))
        time_frac = min(elapsed / max(horizon, 1e-8), 1.0) if horizon > 0 else 0.0
        # Behind schedule → accelerate sizing of remaining schedule
        behind = max(time_frac - (1.0 - remaining_frac), 0.0)

        # Price movement / impact feedback
        arrival = context_float(ctx, "arrival_price", mid)
        if side == "buy":
            move_bps = (mid - arrival) / max(arrival, 1e-12) * 1e4
        else:
            move_bps = (arrival - mid) / max(arrival, 1e-12) * 1e4
        impact_feedback = float(ctx.get("realized_impact_bps", 0.0))
        # If impact high and urgency low, slow; if behind + adverse, keep pace
        impact_factor = 1.0 / (1.0 + max(impact_feedback - 5.0, 0.0) / 30.0)
        if urg in {Urgency.HIGH, Urgency.CRITICAL}:
            impact_factor = max(impact_factor, 0.85)

        pace = (
            spread_factor
            * vol_factor
            * liquidity
            * imb_factor
            * impact_factor
            * (0.7 + 0.3 * fill_rate)
        )
        pace = float(np.clip(pace, 0.25, 2.0))

        # Schedule shape
        if behind > 0.05 or urg == Urgency.CRITICAL:
            weights = np.linspace(1.6, 0.6, n)
        elif fill_rate < 0.5:
            # Poor fills: keep size but we'll aggress limits
            weights = np.ones(n)
        else:
            weights = np.ones(n)
        weights = weights * pace
        weights = np.maximum(weights, 1e-8)
        weights = weights / float(np.sum(weights))

        # Participation throttle
        day_seconds = context_float(ctx, "trading_day_seconds", 23400.0)
        horizon_frac = min(horizon / max(day_seconds, 1.0), 1.0)
        expected_vol = max(adv, 1e-12) * max(horizon_frac, 1e-12)
        base_part = float(ctx.get("base_participation", self.base_participation))
        max_trade = min(approved, expected_vol * base_part * pace * 1.5)
        # Always allow full parent if capacity permits; never exceed parent
        target = min(approved, max(max_trade, approved * min(pace, 1.0)))
        if urg in {Urgency.HIGH, Urgency.CRITICAL}:
            target = approved
        qtys = redistribute_to_parent((weights * target).tolist(), min(target, approved))

        # Limit aggression reacts to fill rate (not quantity)
        agg_boost = 0.0
        if fill_rate < 0.6:
            agg_boost = (0.6 - fill_rate) * 0.8
        # Map boost onto urgency ladder for limit hints
        base_agg = URGENCY_LIMIT_AGGRESSION[urg]
        effective_agg = min(base_agg + agg_boost, 1.0)
        # Synthesize limit directly
        half = 0.5 * max(spread, 0.0)
        hints: list[float | None] = []
        for _ in range(n):
            if mid <= 0:
                hints.append(None)
            elif side == "buy":
                hints.append(float(mid + half * effective_agg))
            else:
                hints.append(float(mid - half * effective_agg))

        # Fallback to standard helper if no spread
        if spread <= 0 and mid > 0:
            hints = [limit_hint(mid, spread, side, urg) for _ in range(n)]

        offsets = schedule_offsets(n, horizon)
        return self._finalize_slices(
            qtys,
            offsets,
            parent_qty=approved,
            market_context=ctx,
            urgency=urg,
            limit_prices=hints,
            metadata=[
                {
                    "algo": self.name,
                    "slice_index": i,
                    "pace": pace,
                    "spread_factor": spread_factor,
                    "vol_factor": vol_factor,
                    "fill_rate": fill_rate,
                    "behind": behind,
                    "move_bps": move_bps,
                }
                for i in range(n)
            ],
        )


__all__ = ["AdaptiveAlgorithm"]

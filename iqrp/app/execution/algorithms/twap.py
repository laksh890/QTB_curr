"""Time-Weighted Average Price (TWAP) execution algorithm."""

from __future__ import annotations

from typing import Any

import numpy as np

from iqrp.app.execution.algorithms.base import (
    ChildSlice,
    ExecutionAlgorithm,
    MarketContext,
    apply_participation_cap,
    approved_quantity,
    context_float,
    n_slices_for_urgency,
    redistribute_to_parent,
    schedule_offsets,
    urgency_from_context,
)
from iqrp.app.execution.types import Urgency


class TWAPAlgorithm(ExecutionAlgorithm):
    """Time-sliced execution with interval, participation cap, residual, and optional jitter.

    Parameters
    ----------
    n_slices:
        Base number of child slices (adjusted downward by high urgency).
    horizon_seconds:
        Total schedule horizon. If ``interval_seconds`` is set and ``n_slices``
        is not provided via constructor default path, horizon/interval drives count.
    interval_seconds:
        Optional fixed interval between slices. When set, ``n_slices`` defaults to
        ``ceil(horizon / interval)`` unless explicitly provided.
    participation_cap:
        Max fraction of expected ADV over the horizon that may be traded.
    jitter:
        Relative timing jitter in ``[0, 1]`` as a fraction of the slice interval.
    seed:
        RNG seed for reproducible jitter.
    """

    name = "twap"

    def __init__(
        self,
        *,
        n_slices: int | None = 5,
        horizon_seconds: float = 300.0,
        interval_seconds: float | None = None,
        participation_cap: float | None = None,
        jitter: float = 0.0,
        seed: int | None = None,
        default_urgency: Urgency | str = Urgency.NORMAL,
    ) -> None:
        super().__init__(default_urgency=default_urgency)
        self.horizon_seconds = max(float(horizon_seconds), 0.0)
        self.interval_seconds = float(interval_seconds) if interval_seconds is not None else None
        if n_slices is None and self.interval_seconds and self.interval_seconds > 0.0:
            self.n_slices = max(int(np.ceil(self.horizon_seconds / self.interval_seconds)), 1)
        else:
            self.n_slices = max(int(n_slices or 1), 1)
        self.participation_cap = (
            float(participation_cap) if participation_cap is not None else None
        )
        self.jitter = max(float(jitter), 0.0)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

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
        n = n_slices_for_urgency(self.n_slices, urg)
        # Context may override slice count / horizon
        if "n_slices" in ctx:
            n = n_slices_for_urgency(int(ctx["n_slices"]), urg)
        horizon = context_float(ctx, "horizon_seconds", self.horizon_seconds)
        if self.interval_seconds and self.interval_seconds > 0.0 and "n_slices" not in ctx:
            n = n_slices_for_urgency(
                max(int(np.ceil(horizon / self.interval_seconds)), 1),
                urg,
            )

        # Equal time weights; liquidity-aware optional sizing via ADV / depth
        weights = np.ones(n, dtype=np.float64)
        depth = ctx.get("depth")
        if depth is not None:
            d = np.asarray(depth, dtype=np.float64).reshape(-1)
            if d.size >= n and float(np.sum(d[:n])) > 0.0:
                weights = np.maximum(d[:n], 0.0)
            elif d.size > 0 and float(np.sum(d)) > 0.0:
                # Broadcast mean depth as soft liquidity prior (still equal if uniform)
                weights = weights * max(float(np.mean(d)), 1e-12)

        weights = weights / max(float(np.sum(weights)), 1e-12)
        raw = (weights * approved).tolist()

        # Participation cap may reduce slices; residual re-allocated to remaining capacity
        adv = context_float(ctx, "adv", context_float(ctx, "average_daily_volume", 1e18))
        # Treat horizon as fraction of a trading day (~6.5h = 23400s) when ADV provided
        day_seconds = context_float(ctx, "trading_day_seconds", 23400.0)
        horizon_frac = min(horizon / max(day_seconds, 1.0), 1.0) if day_seconds > 0 else 1.0
        capped = apply_participation_cap(
            raw,
            adv=adv,
            participation_cap=self.participation_cap
            if self.participation_cap is not None
            else ctx.get("participation_cap"),
            horizon_fraction=horizon_frac,
        )
        # Residual handling: redistribute unmet qty into uncapped capacity without exceeding parent
        shortfall = approved - float(sum(capped))
        if shortfall > 1e-12:
            # Try to put residual on later slices (still respecting per-slice cap if any)
            cap_rate = self.participation_cap
            if cap_rate is None and ctx.get("participation_cap") is not None:
                cap_rate = float(ctx["participation_cap"])
            if cap_rate is None or cap_rate <= 0.0:
                capped = redistribute_to_parent(capped, approved)
            else:
                expected_vol = max(adv, 1e-12) * max(horizon_frac, 1e-12)
                per_slice_cap = (expected_vol * float(cap_rate)) / max(n, 1)
                for i in range(n):
                    if shortfall <= 1e-12:
                        break
                    room = max(per_slice_cap - capped[i], 0.0)
                    take = min(room, shortfall)
                    capped[i] += take
                    shortfall -= take
                # Any remaining residual that still fits under parent but not caps is dropped
                # (hard participation constraint) — never exceed parent
                capped = redistribute_to_parent(capped, min(float(sum(capped)), approved))

        jitter = float(ctx.get("jitter", self.jitter))
        offsets = schedule_offsets(n, horizon, jitter=jitter, rng=self._rng)
        return self._finalize_slices(
            capped,
            offsets,
            parent_qty=approved,
            market_context=ctx,
            urgency=urg,
            metadata=[{"algo": self.name, "slice_index": i, "n_slices": n} for i in range(n)],
        )


__all__ = ["TWAPAlgorithm"]

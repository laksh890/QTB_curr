"""Assumed OHLCV microstructure fill model (not observed bid/ask)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class FillRecordDetail:
    signal_timestamp: str
    order_timestamp: str
    fill_timestamp: str
    candidate_id: str
    instrument: str
    side: str  # BUY/SELL
    requested_qty: float
    filled_qty: float
    requested_price: float  # mid at order
    fill_price: float
    fees: float
    slippage_bps: float
    half_spread_bps: float
    latency_bars: int
    status: str  # FILLED/PARTIAL/REJECTED/CANCELLED
    cost_model_label: str = "ASSUMED_OHLCV_MICROSTRUCTURE"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AssumedFillModel:
    """Research-grade simulated fills from mid prices + assumed spread/slippage."""

    def __init__(self, cfg: dict[str, float], *, rng: np.random.Generator) -> None:
        self.cfg = dict(cfg)
        self.rng = rng
        self.label = "ASSUMED_OHLCV_MICROSTRUCTURE"

    def simulate(
        self,
        *,
        side: str,
        qty: float,
        mid: float,
        signal_ts: str,
        order_ts: str,
        fill_ts: str,
        candidate_id: str,
        instrument: str = "BTCUSDT",
        force_status: str | None = None,
    ) -> FillRecordDetail:
        if mid <= 0 or qty == 0:
            return FillRecordDetail(
                signal_timestamp=signal_ts,
                order_timestamp=order_ts,
                fill_timestamp=fill_ts,
                candidate_id=candidate_id,
                instrument=instrument,
                side=side,
                requested_qty=float(qty),
                filled_qty=0.0,
                requested_price=float(mid),
                fill_price=float(mid),
                fees=0.0,
                slippage_bps=0.0,
                half_spread_bps=0.0,
                latency_bars=int(self.cfg.get("latency_bars") or 1),
                status="CANCELLED",
                cost_model_label=self.label,
            )

        status = force_status
        if status is None:
            u = float(self.rng.random())
            if u < float(self.cfg.get("reject_prob") or 0):
                status = "REJECTED"
            elif u < float(self.cfg.get("reject_prob") or 0) + float(self.cfg.get("partial_fill_prob") or 0):
                status = "PARTIAL"
            else:
                status = "FILLED"

        if status == "REJECTED":
            return FillRecordDetail(
                signal_timestamp=signal_ts,
                order_timestamp=order_ts,
                fill_timestamp=fill_ts,
                candidate_id=candidate_id,
                instrument=instrument,
                side=side,
                requested_qty=float(qty),
                filled_qty=0.0,
                requested_price=float(mid),
                fill_price=float(mid),
                fees=0.0,
                slippage_bps=0.0,
                half_spread_bps=float(self.cfg.get("half_spread_bps") or 0),
                latency_bars=int(self.cfg.get("latency_bars") or 1),
                status="REJECTED",
                cost_model_label=self.label,
                meta={"reason": "simulated_reject"},
            )

        fill_frac = 1.0 if status == "FILLED" else float(self.rng.uniform(0.3, 0.7))
        filled = float(qty) * fill_frac
        half = float(self.cfg.get("half_spread_bps") or 0)
        var = float(self.cfg.get("variable_spread_bps") or 0) * float(self.rng.uniform(0, 1))
        slip = float(self.cfg.get("slippage_bps") or 0)
        # adverse: buy pays mid*(1+bps), sell receives mid*(1-bps)
        total_bps = half + var + slip
        if side.upper() in {"BUY", "LONG"}:
            fill_px = mid * (1.0 + total_bps / 1e4)
        else:
            fill_px = mid * (1.0 - total_bps / 1e4)
        notional = abs(filled) * fill_px
        fees = notional * float(self.cfg.get("commission_bps") or 0) / 1e4
        return FillRecordDetail(
            signal_timestamp=signal_ts,
            order_timestamp=order_ts,
            fill_timestamp=fill_ts,
            candidate_id=candidate_id,
            instrument=instrument,
            side=side.upper(),
            requested_qty=float(qty),
            filled_qty=float(filled) if status != "REJECTED" else 0.0,
            requested_price=float(mid),
            fill_price=float(fill_px),
            fees=float(fees),
            slippage_bps=slip + var,
            half_spread_bps=half,
            latency_bars=int(self.cfg.get("latency_bars") or 1),
            status="PARTIAL" if fill_frac < 1.0 - 1e-12 else "FILLED",
            cost_model_label=self.label,
        )


__all__ = ["AssumedFillModel", "FillRecordDetail"]

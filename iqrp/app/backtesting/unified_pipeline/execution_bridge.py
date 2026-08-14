"""Execution and accounting bridges to existing ExecutionEngine / ledgers."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from iqrp.app.backtesting.accounting import (
    CapitalState,
    FillLog,
    FillRecord,
    OrderLog,
    OrderRecord,
    PositionBook,
    TradeLedger,
    full_accounting_audit,
    reconcile_capital,
)
from iqrp.app.backtesting.unified_pipeline.types import LineageRecord, StageOutcome
from iqrp.app.execution import (
    ExecutionEngine,
    ExecutionSettings,
    KillSwitch,
    SimulatedVenue,
)
from iqrp.app.execution.smart_routing.venue_state import VenueState


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_execution_engine() -> ExecutionEngine:
    settings = ExecutionSettings.default()
    settings = settings.model_copy(
        update={
            "tick_lot": settings.tick_lot.model_copy(
                update={"default_lot_size": 1e-8, "min_qty": 1e-8}
            )
        }
    )
    return ExecutionEngine(settings=settings, kill_switch=KillSwitch())


def plan_and_execute(
    *,
    engine: ExecutionEngine,
    current_qty: dict[str, float],
    target_qty: dict[str, float],
    prices: dict[str, float],
    instrument: str,
    lineage: LineageRecord,
    simulation_mode: str = "fill",
    spread: float = 0.01,
) -> dict[str, Any]:
    """Create orders from deltas via ExecutionEngine and simulate fills."""
    mid = float(prices.get(instrument, 0.0) or 0.0)
    orders = engine.plan_from_targets(
        {k: float(v) for k, v in current_qty.items()},
        {k: float(v) for k, v in target_qty.items()},
        prices={k: float(v) for k, v in prices.items()},
        # Crypto / high-price instruments require fractional lots; equity min_qty=1
        # would silently drop sub-unit deltas (e.g. 0.05 * $1e6 / $97k ≈ 0.5).
        lot_size=1e-8,
        min_qty=1e-8,
        round_lots=True,
    )
    for o in orders:
        meta = getattr(o, "metadata", None)
        if meta is None:
            try:
                o.metadata = {}  # type: ignore[attr-defined]
                meta = o.metadata
            except Exception:  # noqa: BLE001
                meta = {}
        if isinstance(meta, dict):
            meta.update(lineage.to_dict())
            meta["candidate_id"] = lineage.candidate_id
            meta["risk_decision_id"] = lineage.risk_decision_id
            meta["portfolio_decision_id"] = lineage.portfolio_decision_id
        try:
            o.strategy_id = lineage.candidate_id
        except Exception:  # noqa: BLE001
            pass

    if not orders:
        return {
            "outcome": StageOutcome.SKIPPED_FLAT.value,
            "orders": [],
            "fills": [],
            "report": None,
            "lineage": lineage.to_dict(),
        }

    venue_state = VenueState(
        venue_id="SIM",
        instruments={str(i).upper() for i in (set(prices) | {instrument})},
        available_qty=1e12,
        adv=1e6,
        mid=mid if mid > 0 else 100.0,
        bid=(mid if mid > 0 else 100.0) - 0.005,
        ask=(mid if mid > 0 else 100.0) + 0.005,
        min_qty=1e-8,
        lot_size=1e-8,
    )
    venue = SimulatedVenue(
        venue_id="SIM",
        state=venue_state,
        mode=simulation_mode,
        instruments=set(prices) | {instrument},
        mid=mid if mid > 0 else 100.0,
    )
    market_context = {
        inst: {"mid": float(px), "spread": spread, "adv": 1e6, "volatility": 0.02}
        for inst, px in prices.items()
    }
    report = engine.execute(
        orders,
        algo="market",
        venues=[venue],
        market_context=market_context,
        simulation_mode=True,
        strategy_id=lineage.candidate_id,
    )
    fills = list(getattr(report, "fills", None) or [])
    # normalize fill dicts
    fill_dicts: list[dict[str, Any]] = []
    for f in fills:
        if isinstance(f, dict):
            fill_dicts.append(dict(f))
        elif hasattr(f, "to_dict"):
            fill_dicts.append(f.to_dict())
        else:
            fill_dicts.append({"raw": str(f)})

    outcome = StageOutcome.FILL_COMPLETE
    if simulation_mode == "partial":
        outcome = StageOutcome.FILL_PARTIAL
    elif simulation_mode == "reject" or (hasattr(report, "errors") and report.errors):
        # may still have no fills
        if not fill_dicts:
            outcome = StageOutcome.ORDER_REJECTED

    # stamp order ids into lineage
    order_ids = [getattr(o, "order_id", "") for o in orders]
    lineage.order_id = order_ids[0] if order_ids else lineage.order_id
    lineage.fill_ids = [str(f.get("event_id") or f.get("fill_id") or "") for f in fill_dicts]
    lineage.extra["all_order_ids"] = order_ids
    lineage.extra["execution_report"] = (
        report.to_dict() if hasattr(report, "to_dict") else {"fills": fill_dicts}
    )

    return {
        "outcome": outcome.value,
        "orders": [
            {
                "order_id": o.order_id,
                "instrument": o.instrument,
                "side": o.side.value if hasattr(o.side, "value") else str(o.side),
                "quantity": float(o.quantity),
                "lineage": lineage.to_dict(),
            }
            for o in orders
        ],
        "fills": fill_dicts,
        "report": report.to_dict() if hasattr(report, "to_dict") else None,
        "lineage": lineage.to_dict(),
    }


def apply_fills_to_ledgers(
    *,
    capital: CapitalState,
    positions: PositionBook,
    order_log: OrderLog,
    fill_log: FillLog,
    trade_ledger: TradeLedger,
    exec_result: dict[str, Any],
    timestamp: str | None = None,
    lineage: LineageRecord | None = None,
) -> dict[str, Any]:
    """Update accounting ledgers from execution result; return summary."""
    ts = timestamp or _utc_now()
    applied: list[dict[str, Any]] = []
    for o in exec_result.get("orders") or []:
        order_log.add(
            OrderRecord(
                order_id=str(o["order_id"]),
                timestamp=ts,
                instrument=str(o["instrument"]),
                side=str(o["side"]),
                quantity=float(o["quantity"]),
                order_type="MARKET",
                status="SUBMITTED",
                meta=dict(o.get("lineage") or {}),
            )
        )
    for f in exec_result.get("fills") or []:
        inst = str(f.get("instrument") or "")
        qty = float(f.get("quantity") or f.get("qty") or 0.0)
        px = float(f.get("price") or f.get("fill_price") or 0.0)
        # Infer side from matching order
        side = "buy"
        oid = str(f.get("order_id") or "")
        for o in exec_result.get("orders") or []:
            if o["order_id"] == oid:
                side = str(o["side"]).lower()
                break
        if side in {"sell", "short"}:
            side_l = "sell"
        else:
            side_l = "buy"
        fee = float(f.get("fee") or 0.0)
        realized = positions.apply_fill(inst, quantity=qty, price=px, side=side_l)
        # Cash: buys reduce cash by notional; sells increase
        notional = abs(qty) * px
        if side_l == "buy":
            capital.apply_cash_delta(-notional, reason=f"fill_buy:{oid}")
        else:
            capital.apply_cash_delta(notional, reason=f"fill_sell:{oid}")
        if fee:
            capital.record_fee(fee)
        if abs(realized) > 0:
            capital.realize(realized, settle_into_cash=False)
        fid = str(f.get("event_id") or f.get("fill_id") or f"fill_{uuid.uuid4().hex[:12]}")
        fill_log.add(
            FillRecord(
                fill_id=fid,
                order_id=oid,
                timestamp=ts,
                instrument=inst,
                side=side_l,
                quantity=qty,
                price=px,
                fee=fee,
                meta={"lineage": (lineage.to_dict() if lineage else {})},
            )
        )
        trade_ledger.record_fill_as_trade(
            trade_id=f"tr_{uuid.uuid4().hex[:12]}",
            instrument=inst,
            side=side_l,
            quantity=qty,
            timestamp=ts,
            price=px,
            realized_pnl=realized,
            fees=fee,
        )
        applied.append({"fill_id": fid, "instrument": inst, "qty": qty, "price": px, "side": side_l})

    # Mark portfolio
    # prices from last fill
    for f in exec_result.get("fills") or []:
        inst = str(f.get("instrument") or "")
        px = float(f.get("price") or f.get("fill_price") or 0.0)
        if inst and px > 0:
            positions.mark_all({inst: px})
    capital.mark_unrealized(
        positions.total_unrealized(), market_value=positions.total_market_value()
    )
    return {
        "fills_applied": applied,
        "positions": positions.quantities(),
        "equity": capital.equity,
        "cash": capital.cash,
    }


def reconcile_pipeline_state(
    *,
    capital: CapitalState,
    positions: PositionBook,
    fill_log: FillLog,
    snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    fills = [f.to_dict() for f in fill_log.fills]
    # enrich for replay helpers expecting side/qty/price
    for f in fills:
        f.setdefault("qty", f.get("quantity"))
        f.setdefault("symbol", f.get("instrument"))
    audit = full_accounting_audit(
        capital=capital,
        fills=fills,
        snapshots=snapshots or [],
        ending_equity=capital.equity,
        final_positions=positions.quantities(),
        tolerance=1e-4,
    )
    cap = reconcile_capital(capital, ending_equity=capital.equity, fail=False)
    audit["capital_identity"] = cap.to_dict()
    audit["outcome"] = (
        StageOutcome.RECONCILIATION_OK.value
        if audit.get("ok")
        else StageOutcome.RECONCILIATION_FAILED.value
    )
    return audit


__all__ = [
    "apply_fills_to_ledgers",
    "build_execution_engine",
    "plan_and_execute",
    "reconcile_pipeline_state",
]

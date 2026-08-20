"""Sequential paper-trading simulator (bar-by-bar, no lookahead)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from iqrp.app.paper_trading.fill_model import AssumedFillModel, FillRecordDetail
from iqrp.app.paper_trading.risk import KillSwitchState, PaperRiskLimits, check_risk


@dataclass
class PendingOrder:
    ready_bar: int
    side: str
    qty: float  # signed: +buy, -sell (base qty)
    mid_at_order: float
    signal_ts: str
    order_ts: str
    candidate_id: str
    target_weight: float


@dataclass
class PaperSession:
    capital: float
    cash: float
    qty: float = 0.0  # base asset qty
    avg_price: float = 0.0
    equity: float = 0.0
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    current_day: str | None = None
    weight: float = 0.0
    fills: list[FillRecordDetail] = field(default_factory=list)
    orders: list[dict[str, Any]] = field(default_factory=list)
    risk_events: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    pending: list[PendingOrder] = field(default_factory=list)
    kill: KillSwitchState = field(default_factory=KillSwitchState)
    n_rejects: int = 0
    n_partials: int = 0
    n_fills: int = 0
    fees_paid: float = 0.0
    realized_pnl: float = 0.0
    last_recon_ok: bool = True
    lookahead_violations: int = 0


def _mark_equity(cash: float, qty: float, price: float) -> float:
    return float(cash + qty * price)


def _apply_fill(session: PaperSession, fill: FillRecordDetail, *, price_for_mark: float) -> None:
    if fill.status == "REJECTED" or fill.filled_qty == 0:
        session.n_rejects += 1
        return
    if fill.status == "PARTIAL":
        session.n_partials += 1
    session.n_fills += 1
    session.fees_paid += float(fill.fees)
    session.cash -= float(fill.fees)

    signed = float(fill.filled_qty) if fill.side in {"BUY", "LONG"} else -float(fill.filled_qty)
    # cash: buy decreases cash, sell increases
    session.cash -= signed * float(fill.fill_price)

    # position avg / realized
    prev = session.qty
    new = prev + signed
    if prev == 0 or (prev > 0 and signed > 0) or (prev < 0 and signed < 0):
        # increasing same direction
        if abs(new) > 1e-15:
            session.avg_price = (abs(prev) * session.avg_price + abs(signed) * fill.fill_price) / abs(new)
        session.qty = new
    elif abs(signed) <= abs(prev) + 1e-15:
        # reducing / closing
        closed = min(abs(signed), abs(prev))
        if prev > 0:
            session.realized_pnl += closed * (fill.fill_price - session.avg_price)
        else:
            session.realized_pnl += closed * (session.avg_price - fill.fill_price)
        session.qty = new
        if abs(session.qty) < 1e-12:
            session.qty = 0.0
            session.avg_price = 0.0
    else:
        # flip
        closed = abs(prev)
        if prev > 0:
            session.realized_pnl += closed * (fill.fill_price - session.avg_price)
        else:
            session.realized_pnl += closed * (session.avg_price - fill.fill_price)
        session.qty = new
        session.avg_price = float(fill.fill_price)

    session.equity = _mark_equity(session.cash, session.qty, price_for_mark)
    session.peak_equity = max(session.peak_equity, session.equity)
    session.fills.append(fill)


def reconcile_session(session: PaperSession, price: float, *, tol: float = 1e-4) -> dict[str, Any]:
    marked = _mark_equity(session.cash, session.qty, price)
    drift = abs(marked - session.equity)
    # rebuild equity from cash+qty
    session.equity = marked
    ok = drift <= tol * max(abs(session.capital), 1.0) or drift < 1e-6
    # fee/fill consistency soft check
    session.last_recon_ok = bool(ok)
    return {
        "ok": ok,
        "drift": float(drift),
        "equity": float(session.equity),
        "cash": float(session.cash),
        "qty": float(session.qty),
        "fees_paid": float(session.fees_paid),
    }


def run_sequential_paper(
    *,
    timestamps: pd.Series,
    closes: pd.Series,
    target_weights: np.ndarray,  # desired portfolio weight per bar (signed)
    fill_model: AssumedFillModel,
    limits: PaperRiskLimits,
    initial_capital: float,
    latency_bars: int = 1,
    candidate_label: str = "combo",
    inject: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Process one bar at a time. target_weights[i] may only use info <= i."""
    inject = dict(inject or {})
    n = len(closes)
    assert len(timestamps) == n == len(target_weights)
    session = PaperSession(
        capital=float(initial_capital),
        cash=float(initial_capital),
        equity=float(initial_capital),
        peak_equity=float(initial_capital),
        day_start_equity=float(initial_capital),
    )
    ts = pd.to_datetime(timestamps, utc=True)
    px = closes.to_numpy(dtype=float)
    tw = np.asarray(target_weights, dtype=float)

    # Lookahead assertion helper: target at i must not be NaN from future — caller responsibility
    stale = 0
    last_ts = None
    seen_ts: set[str] = set()
    data_events: list[dict[str, Any]] = []

    for i in range(n):
        t = ts.iloc[i]
        t_str = str(t)
        price = float(px[i])
        if not np.isfinite(price) or price <= 0:
            stale += 1
            data_events.append({"i": i, "event": "missing_price"})
            if inject.get("missing_candles") and i in set(inject.get("missing_indices") or []):
                continue
            continue

        # duplicate / out-of-order detection
        if t_str in seen_ts:
            data_events.append({"i": i, "event": "duplicate_candle"})
            if inject.get("halt_on_duplicate"):
                session.kill.trip("DUPLICATE_CANDLE", meta={"ts": t_str})
            continue
        if last_ts is not None and t <= last_ts:
            data_events.append({"i": i, "event": "out_of_order"})
            if inject.get("halt_on_ooo"):
                session.kill.trip("OUT_OF_ORDER", meta={"ts": t_str})
            continue
        seen_ts.add(t_str)
        last_ts = t
        stale = 0

        day = str(t.floor("D"))
        if session.current_day is None:
            session.current_day = day
            session.day_start_equity = session.equity
        elif day != session.current_day:
            session.current_day = day
            session.day_start_equity = session.equity

        # Process due pending orders at this bar (latency)
        still_pending: list[PendingOrder] = []
        for po in session.pending:
            if i >= po.ready_bar:
                force = None
                if inject.get("force_reject_orders"):
                    force = "REJECTED"
                elif inject.get("force_partial_orders"):
                    force = "PARTIAL"
                fill = fill_model.simulate(
                    side=po.side,
                    qty=abs(po.qty),
                    mid=price,  # fill mid at fill time (no future beyond i)
                    signal_ts=po.signal_ts,
                    order_ts=po.order_ts,
                    fill_ts=t_str,
                    candidate_id=po.candidate_id,
                    force_status=force,
                )
                if inject.get("exchange_timeout") and i == int(inject.get("timeout_bar") or -1):
                    session.kill.trip("EXCHANGE_TIMEOUT")
                    fill.status = "REJECTED"
                    fill.filled_qty = 0.0
                _apply_fill(session, fill, price_for_mark=price)
                session.orders.append({"bar": i, "pending": True, "fill": fill.to_dict()})
            else:
                still_pending.append(po)
        session.pending = still_pending

        # Mark
        session.equity = _mark_equity(session.cash, session.qty, price)
        session.peak_equity = max(session.peak_equity, session.equity)
        session.weight = (session.qty * price / session.equity) if session.equity > 1e-12 else 0.0

        # Signal / target at i (frozen; no future)
        desired = float(tw[i]) if np.isfinite(tw[i]) else 0.0
        if inject.get("model_failure") and i >= int(inject.get("model_failure_bar") or 10**9):
            desired = 0.0
            model_failed = True
        else:
            model_failed = False
        if inject.get("signal_failure") and i == int(inject.get("signal_failure_bar") or -1):
            desired = 0.0
            model_failed = True

        recon = reconcile_session(session, price)
        if inject.get("force_recon_fail") and i == int(inject.get("recon_fail_bar") or -1):
            recon["ok"] = False
            session.last_recon_ok = False

        approved, reasons = check_risk(
            limits=limits,
            kill=session.kill,
            target_weight=desired,
            current_weight=float(session.weight),
            equity=float(session.equity),
            peak_equity=float(session.peak_equity),
            day_start_equity=float(session.day_start_equity),
            stale_bars=stale,
            model_failed=model_failed,
            recon_failed=not recon["ok"],
            exec_failed=bool(inject.get("exec_failure") and i >= int(inject.get("exec_failure_bar") or 10**9)),
        )
        if reasons:
            session.risk_events.append({"bar": i, "ts": t_str, "reasons": reasons, "desired": desired, "approved": approved})

        if session.kill.halted:
            approved = 0.0

        # Generate order if weight delta material
        delta_w = approved - float(session.weight)
        if abs(delta_w) > 1e-6 and not session.kill.halted:
            # convert weight delta to qty using current equity/price
            target_qty = (approved * session.equity) / price if price > 0 else 0.0
            delta_qty = target_qty - session.qty
            if abs(delta_qty) > 1e-12:
                side = "BUY" if delta_qty > 0 else "SELL"
                ready = i + max(int(latency_bars), 0)
                # If latency 0, fill same bar AFTER signal — still no future bars beyond i
                po = PendingOrder(
                    ready_bar=ready,
                    side=side,
                    qty=float(delta_qty),
                    mid_at_order=price,
                    signal_ts=t_str,
                    order_ts=t_str,
                    candidate_id=candidate_label,
                    target_weight=approved,
                )
                if ready <= i:
                    fill = fill_model.simulate(
                        side=side,
                        qty=abs(delta_qty),
                        mid=price,
                        signal_ts=t_str,
                        order_ts=t_str,
                        fill_ts=t_str,
                        candidate_id=candidate_label,
                    )
                    _apply_fill(session, fill, price_for_mark=price)
                    session.orders.append({"bar": i, "pending": False, "fill": fill.to_dict()})
                else:
                    session.pending.append(po)

        session.equity = _mark_equity(session.cash, session.qty, price)
        session.weight = (session.qty * price / session.equity) if session.equity > 1e-12 else 0.0
        session.equity_curve.append(
            {
                "bar": i,
                "ts": t_str,
                "equity": session.equity,
                "cash": session.cash,
                "qty": session.qty,
                "weight": session.weight,
                "price": price,
                "halted": session.kill.halted,
            }
        )

    # Final reconcile
    final_px = float(px[-1]) if n else 0.0
    final_recon = reconcile_session(session, final_px) if n else {"ok": True, "drift": 0.0}

    eq = np.array([e["equity"] for e in session.equity_curve], dtype=float)
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12) if len(eq) > 2 else np.array([])
    sharpe = float("nan")
    if rets.size > 5 and np.std(rets, ddof=1) > 1e-15:
        # assume crypto intraday; annualize by bars in curve
        sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(252 * 24))  # rough; overwritten by runner with TF

    peak = np.maximum.accumulate(eq) if len(eq) else np.array([initial_capital])
    dd = (peak - eq) / np.maximum(peak, 1e-12) if len(eq) else np.array([0.0])
    net_return = float(session.equity / initial_capital - 1.0) if initial_capital else 0.0

    return {
        "candidate_label": candidate_label,
        "n_bars": n,
        "net_return": net_return,
        "final_equity": float(session.equity),
        "max_drawdown": float(dd.max()) if len(dd) else 0.0,
        "sharpe_raw": sharpe,
        "n_fills": session.n_fills,
        "n_rejects": session.n_rejects,
        "n_partials": session.n_partials,
        "fees_paid": float(session.fees_paid),
        "realized_pnl": float(session.realized_pnl),
        "kill_switch": session.kill.to_dict(),
        "risk_events": session.risk_events,
        "data_events": data_events,
        "final_recon": final_recon,
        "fills": [f.to_dict() for f in session.fills],
        "equity_curve_tail": session.equity_curve[-5:],
        "equity_curve": session.equity_curve if n <= 2000 else session.equity_curve[:: max(n // 2000, 1)],
        "lookahead_violations": session.lookahead_violations,
        "cost_model_label": fill_model.label,
    }


__all__ = ["run_sequential_paper", "PaperSession", "reconcile_session"]

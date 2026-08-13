"""Event handlers implementing MARKET → … → RISK_UPDATE cascade."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Mapping

import numpy as np

from iqrp.app.backtesting.event_engine import (
    Event,
    EventDrivenEngine,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    PortfolioEvent,
    RiskEvent,
    RiskUpdateEvent,
    SignalEvent,
)
from iqrp.app.backtesting.event_engine.forecast_event import ForecastEvent
from iqrp.app.backtesting.pit import LookaheadViolation, assert_no_lookahead
from iqrp.app.backtesting.runner.context import PipelineContext
from iqrp.app.backtesting.accounting.snapshots import PortfolioSnapshot


def _aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise LookaheadViolation(f"naive timestamp rejected: {ts!r}")
    return ts


def _merge_strategy(payload: Mapping[str, Any] | None, extra: Mapping[str, Any] | None) -> dict[str, Any]:
    out = dict(payload or {})
    if extra:
        out.update(dict(extra))
    return out


class EventPipeline:
    """Registers cascade handlers on an :class:`EventDrivenEngine`."""

    def __init__(self, engine: EventDrivenEngine, context: PipelineContext) -> None:
        self.engine = engine
        self.ctx = context
        context.engine = engine
        self._register()

    def _register(self) -> None:
        e = self.engine
        e.register(EventType.MARKET, self.on_market)
        e.register(EventType.FEATURE, self.on_feature)
        e.register(EventType.SIGNAL, self.on_signal)
        e.register(EventType.FORECAST, self.on_forecast)
        e.register(EventType.RISK, self.on_risk)
        e.register(EventType.PORTFOLIO, self.on_portfolio)
        e.register(EventType.ORDER, self.on_order)
        e.register(EventType.EXECUTION, self.on_execution)
        e.register(EventType.FILL, self.on_fill)
        e.register(EventType.POSITION, self.on_position)
        e.register(EventType.PNL, self.on_pnl)
        e.register(EventType.RISK_UPDATE, self.on_risk_update)

    def _pit_check(self, data_ts: datetime, event: Event, *, context: str) -> None:
        if not self.ctx.config.enforce_pit:
            return
        try:
            assert_no_lookahead(data_ts, event.timestamp, context=context)
        except LookaheadViolation as exc:
            self.ctx.invalidated = True
            self.ctx.invalidation_reason = str(exc)
            self.engine.invalidate(str(exc))
            raise

    def on_market(self, event: Event) -> None:
        self.ctx.event_count += 1
        self.ctx.current_time = _aware(event.timestamp)
        payload = dict(event.payload or {})
        bars = dict(payload.get("bars") or {})
        if not bars:
            inst = payload.get("instrument") or payload.get("symbol")
            if inst is not None:
                bars = {str(inst): payload}
        for inst, bar in bars.items():
            if not isinstance(bar, Mapping):
                continue
            # Leakage boundary: bar timestamp must not be after event time
            bar_ts = bar.get("timestamp", event.timestamp)
            if isinstance(bar_ts, str):
                bar_ts = datetime.fromisoformat(bar_ts.replace("Z", "+00:00"))
            if isinstance(bar_ts, datetime):
                self._pit_check(bar_ts, event, context=f"market:{inst}")
            close = bar.get("close", bar.get("price"))
            if close is None:
                continue
            self.ctx.latest_prices[str(inst)] = float(close)
            self.ctx.latest_bars[str(inst)] = dict(bar)
        if self.ctx.universe:
            # Restrict to configured universe when provided
            for inst in list(self.ctx.latest_prices):
                if inst not in self.ctx.universe:
                    # Keep prices for open positions; skip new signals later
                    pass
        self.ctx.mark_prices()
        strat = self.ctx.strategy.on_market_data(event, self.ctx)
        bar_extra = self.ctx.strategy.on_bar(event, self.ctx)
        merged = _merge_strategy({"bars": bars}, strat)
        merged = _merge_strategy(merged, bar_extra)
        if merged.get("target_weights"):
            self.ctx.target_weights = {
                str(k): float(v) for k, v in dict(merged["target_weights"]).items()
            }
        self.engine.submit(
            Event(
                timestamp=event.timestamp,
                event_type=EventType.FEATURE,
                payload=merged,
            )
        )

    def on_feature(self, event: Event) -> None:
        self.ctx.event_count += 1
        bars = dict((event.payload or {}).get("bars") or self.ctx.latest_bars)
        # Simple features: returns vs prior close stored in diagnostics
        features: dict[str, dict[str, float]] = {}
        hist = self.ctx.diagnostics.setdefault("price_history", {})
        for inst, bar in bars.items():
            if not isinstance(bar, Mapping):
                continue
            close = float(bar.get("close", self.ctx.latest_prices.get(inst, 0.0)) or 0.0)
            prev = float(hist.get(inst, close) or close)
            ret = 0.0 if prev <= 0 else close / prev - 1.0
            features[str(inst)] = {"close": close, "ret_1": ret}
            hist[str(inst)] = close
        payload = _merge_strategy(
            {"bars": bars, "features": features},
            self.ctx.strategy.on_features(
                event.with_payload(features=features) if hasattr(event, "with_payload") else event,
                self.ctx,
            ),
        )
        if payload.get("target_weights"):
            self.ctx.target_weights = {
                str(k): float(v) for k, v in dict(payload["target_weights"]).items()
            }
        signals = dict(payload.get("signals") or payload.get("target_weights") or {})
        self.engine.submit(
            SignalEvent(
                timestamp=event.timestamp,
                payload={**payload, "signals": signals},
            )
        )

    def on_signal(self, event: Event) -> None:
        self.ctx.event_count += 1
        payload = _merge_strategy(event.payload, self.ctx.strategy.on_signal(event, self.ctx))
        signals = dict(payload.get("signals") or {})
        targets = dict(payload.get("target_weights") or self.ctx.target_weights or {})
        if not targets and signals:
            targets = self.ctx.portfolio_adapter.targets_from_signals(signals)
        if targets:
            self.ctx.target_weights = {str(k): float(v) for k, v in targets.items()}
        self.engine.submit(
            ForecastEvent(
                timestamp=event.timestamp,
                payload={**payload, "signals": signals, "target_weights": dict(self.ctx.target_weights)},
            )
        )

    def on_forecast(self, event: Event) -> None:
        self.ctx.event_count += 1
        payload = _merge_strategy(event.payload, self.ctx.strategy.on_forecast(event, self.ctx))
        # Forecast stage currently passes through strategy targets / signals
        self.engine.submit(
            RiskEvent(
                timestamp=event.timestamp,
                payload=payload,
            )
        )

    def on_risk(self, event: Event) -> None:
        self.ctx.event_count += 1
        payload = _merge_strategy(event.payload, self.ctx.strategy.on_risk(event, self.ctx))
        targets = dict(payload.get("target_weights") or self.ctx.target_weights or {})
        # Simple risk clamp: gross leverage cap
        max_gross = float((self.ctx.config.risk_config or {}).get("max_gross_leverage", 1.0))
        gross = sum(abs(float(v)) for v in targets.values())
        if gross > max_gross + 1e-12 and gross > 0:
            scale = max_gross / gross
            targets = {k: float(v) * scale for k, v in targets.items()}
        self.ctx.target_weights = targets
        self.ctx.risk_state = {
            "gross": sum(abs(v) for v in targets.values()),
            "net": sum(targets.values()),
            "max_gross_leverage": max_gross,
            "asof": event.timestamp.isoformat(),
        }
        self.engine.submit(
            PortfolioEvent(
                timestamp=event.timestamp,
                payload={**payload, "target_weights": targets, "risk": dict(self.ctx.risk_state)},
            )
        )

    def on_portfolio(self, event: Event) -> None:
        self.ctx.event_count += 1
        payload = _merge_strategy(event.payload, self.ctx.strategy.on_portfolio(event, self.ctx))
        targets = dict(payload.get("target_weights") or self.ctx.target_weights or {})
        if payload.get("signals") and not targets:
            targets = self.ctx.portfolio_adapter.targets_from_signals(dict(payload["signals"]))
        elif targets:
            targets = self.ctx.portfolio_adapter.targets_from_weights(targets)
        # Universe filter
        if self.ctx.universe:
            targets = {k: v for k, v in targets.items() if k in self.ctx.universe}
        self.ctx.target_weights = targets
        rebalance = bool(payload.get("rebalance", True))
        equity = self.ctx.current_equity()
        orders: list[dict[str, Any]] = []
        if rebalance and targets is not None:
            orders = self.ctx.execution_adapter.plan_from_targets(
                self.ctx.positions.quantities(),
                targets,
                equity=equity,
                prices=self.ctx.latest_prices,
            )
        self.ctx.pending_orders = list(orders)
        self.engine.submit(
            OrderEvent(
                timestamp=event.timestamp,
                payload={
                    **payload,
                    "target_weights": targets,
                    "orders": orders,
                    "portfolio_backend": self.ctx.portfolio_adapter.backend,
                },
            )
        )

    def on_order(self, event: Event) -> None:
        self.ctx.event_count += 1
        payload = _merge_strategy(event.payload, self.ctx.strategy.on_order(event, self.ctx))
        orders = list(payload.get("orders") or self.ctx.pending_orders or [])
        for o in orders:
            oid = str(o.get("order_id") or uuid.uuid4().hex)
            o["order_id"] = oid
            self.ctx.orders.add(
                {
                    "order_id": oid,
                    "timestamp": event.timestamp.isoformat(),
                    "instrument": str(o.get("instrument", "")),
                    "side": str(o.get("side", "buy")),
                    "quantity": float(o.get("quantity", 0.0)),
                    "order_type": str(o.get("order_type", "market")),
                    "limit_price": o.get("limit_price"),
                    "status": "submitted",
                    "meta": {},
                }
            )
        cost_est = self.ctx.execution_adapter.estimate_costs(
            orders,
            market_context=self._market_context(),
            commission_bps=self.ctx.config.commission_bps,
            spread_bps=self.ctx.config.spread_bps,
            slippage_bps=self.ctx.config.slippage_bps,
        )
        self.engine.submit(
            Event(
                timestamp=event.timestamp,
                event_type=EventType.EXECUTION,
                payload={**payload, "orders": orders, "cost_estimate": cost_est},
            )
        )

    def _market_context(self) -> dict[str, Any]:
        ctx: dict[str, Any] = {}
        spread_bps = float(self.ctx.config.spread_bps)
        for inst, px in self.ctx.latest_prices.items():
            mid = float(px)
            ctx[inst] = {
                "mid": mid,
                "price": mid,
                "spread": mid * spread_bps / 10_000.0,
                "adv": 1_000_000.0,
                "volatility": 0.02,
            }
        return ctx

    def on_execution(self, event: Event) -> None:
        self.ctx.event_count += 1
        orders = list((event.payload or {}).get("orders") or [])
        if not orders:
            # Still advance cascade for mark-to-market path
            self.engine.submit(
                Event(
                    timestamp=event.timestamp,
                    event_type=EventType.POSITION,
                    payload={"fills": []},
                )
            )
            return
        sim = self.ctx.execution_adapter.simulate_execution(
            orders,
            market_context=self._market_context(),
            spread_bps=self.ctx.config.spread_bps,
            slippage_bps=self.ctx.config.slippage_bps,
            commission_bps=self.ctx.config.commission_bps,
            seed=self.ctx.config.seed,
        )
        fill_events = []
        for row in list(sim.get("orders") or []):
            inst = str(row.get("instrument", ""))
            qty = float(row.get("filled_qty", row.get("quantity", 0.0)) or 0.0)
            px = float(row.get("exec_vwap", row.get("price", 0.0)) or 0.0)
            fee = float(row.get("fee", 0.0) or 0.0)
            side = str(row.get("side", "buy"))
            oid = ""
            for o in orders:
                if str(o.get("instrument")) == inst:
                    oid = str(o.get("order_id", ""))
                    side = str(o.get("side", side))
                    break
            if qty <= 0 or px <= 0:
                continue
            fid = uuid.uuid4().hex
            slip_raw = row.get("slippage", 0.0)
            if isinstance(slip_raw, Mapping):
                slip_val = float(
                    slip_raw.get("slippage", slip_raw.get("cost", slip_raw.get("total", 0.0)))
                    or 0.0
                )
            else:
                slip_val = float(slip_raw or 0.0)
            fill_payload = {
                "fill_id": fid,
                "order_id": oid,
                "instrument": inst,
                "symbol": inst,
                "side": side,
                "quantity": qty,
                "price": px,
                "fee": fee,
                "slippage": slip_val,
                "backend": row.get("backend", sim.get("backend")),
            }
            fill_events.append(fill_payload)
            self.engine.submit(FillEvent(timestamp=event.timestamp, payload=fill_payload))
        # Single POSITION event after fills (priority ensures FILLs run first).
        self.engine.submit(
            Event(
                timestamp=event.timestamp,
                event_type=EventType.POSITION,
                payload={"fills": fill_events},
            )
        )

    def on_fill(self, event: Event) -> None:
        self.ctx.event_count += 1
        payload = _merge_strategy(event.payload, self.ctx.strategy.on_fill(event, self.ctx))
        inst = str(payload.get("instrument") or payload.get("symbol") or "")
        side = str(payload.get("side", "buy"))
        qty = abs(float(payload.get("quantity", payload.get("qty", 0.0)) or 0.0))
        px = float(payload.get("price", 0.0) or 0.0)
        fee = abs(float(payload.get("fee", 0.0) or 0.0))
        if not inst or qty <= 0 or px <= 0:
            return
        self.ctx.fills.add(
            {
                "fill_id": str(payload.get("fill_id") or event.event_id),
                "order_id": str(payload.get("order_id", "")),
                "timestamp": event.timestamp.isoformat(),
                "instrument": inst,
                "side": side,
                "quantity": qty,
                "price": px,
                "fee": fee,
                "slippage": float(payload.get("slippage", 0.0) or 0.0),
                "meta": {"backend": payload.get("backend")},
            }
        )
        realized = self.ctx.positions.apply_fill(inst, quantity=qty, price=px, side=side)
        notional = qty * px
        if side.lower() in {"buy", "b", "cover", "long"}:
            self.ctx.capital.apply_cash_delta(-notional, reason=f"buy:{inst}")
        else:
            self.ctx.capital.apply_cash_delta(notional, reason=f"sell:{inst}")
        if fee > 0:
            self.ctx.capital.record_fee(fee)
        if abs(realized) > 1e-12:
            self.ctx.capital.realize(realized, settle_into_cash=False)
        self.ctx.trades.record_fill_as_trade(
            trade_id=str(payload.get("fill_id") or event.event_id),
            instrument=inst,
            side=side,
            quantity=qty,
            timestamp=event.timestamp.isoformat(),
            price=px,
            realized_pnl=realized,
            fees=fee,
        )
        self.ctx.last_costs += fee
        self.ctx.latest_prices[inst] = px
        self.ctx.mark_prices()

    def on_position(self, event: Event) -> None:
        self.ctx.event_count += 1
        self.ctx.mark_prices()
        self.engine.submit(
            Event(
                timestamp=event.timestamp,
                event_type=EventType.PNL,
                payload={
                    "positions": self.ctx.positions.snapshot(),
                    "equity": self.ctx.capital.equity,
                    "cash": self.ctx.capital.cash,
                },
            )
        )

    def on_pnl(self, event: Event) -> None:
        self.ctx.event_count += 1
        self.ctx.bar_count += 1
        equity = self.ctx.current_equity()
        prev = self.ctx.equity_curve[-1] if self.ctx.equity_curve else float(self.ctx.capital.initial_capital)
        ret = 0.0 if prev <= 0 else equity / prev - 1.0
        self.ctx.equity_curve.append(float(equity))
        self.ctx.returns.append(float(ret))
        self.ctx.timestamps.append(event.timestamp.isoformat())
        self.ctx.peak_equity = max(float(self.ctx.peak_equity or equity), equity)
        dd = 0.0 if self.ctx.peak_equity <= 0 else 1.0 - equity / self.ctx.peak_equity
        gross = self.ctx.positions.total_exposure()
        net = self.ctx.positions.total_market_value()
        lev = 0.0 if equity <= 0 else gross / equity
        # Rolling vol / VaR approximations
        r = np.asarray(self.ctx.returns[-21:], dtype=np.float64)
        vol = float(np.std(r) * np.sqrt(252.0)) if r.size > 1 else 0.0
        var = float(-np.quantile(r, 0.05)) if r.size >= 5 else 0.0
        cvar = float(-np.mean(r[r <= -var])) if r.size >= 5 and np.any(r <= -var) else var
        self.ctx.snapshots.add(
            PortfolioSnapshot(
                timestamp=event.timestamp.isoformat(),
                equity=float(equity),
                cash=float(self.ctx.capital.cash),
                gross_exposure=float(gross),
                net_exposure=float(net),
                leverage=float(lev),
                volatility=vol,
                drawdown=float(max(dd, 0.0)),
                var=var,
                cvar=cvar,
                turnover=float(self.ctx.last_turnover),
                costs=float(self.ctx.last_costs),
                realized_pnl=float(self.ctx.capital.realized_pnl),
                unrealized_pnl=float(self.ctx.capital.unrealized_pnl),
                positions=self.ctx.positions.quantities(),
            )
        )
        self.ctx.last_costs = 0.0
        self.engine.submit(
            RiskUpdateEvent(
                timestamp=event.timestamp,
                payload={
                    "equity": equity,
                    "drawdown": dd,
                    "leverage": lev,
                    "volatility": vol,
                    "var": var,
                    "cvar": cvar,
                },
            )
        )

    def on_risk_update(self, event: Event) -> None:
        self.ctx.event_count += 1
        self.ctx.risk_state.update(dict(event.payload or {}))
        # Hard risk breach → invalidate
        max_dd = float((self.ctx.config.risk_config or {}).get("max_drawdown", 1.0))
        dd = float((event.payload or {}).get("drawdown", 0.0) or 0.0)
        if dd > max_dd + 1e-12:
            reason = f"drawdown {dd:.6f} exceeded max_drawdown {max_dd:.6f}"
            self.ctx.invalidated = True
            self.ctx.invalidation_reason = reason
            self.engine.invalidate(reason)


__all__ = ["EventPipeline"]

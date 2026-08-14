"""Institutional ExecutionEngine orchestrator.

CRITICAL RULES
--------------
1. Never generate alpha / invent positions / exceed approved target residual.
2. Before submit: risk validation if risk_engine provided — Risk is authoritative.
3. Kill switches fail-safe: halt blocks new submits.
4. Urgency never overrides hard risk / kill switch.
5. Idempotent events.
6. On HALT: stop new orders; cancel open if configured.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from iqrp.app.core.exceptions import ExecutionError, ValidationError
from iqrp.app.execution.algorithms.base import ChildSlice, coerce_urgency
from iqrp.app.execution.analytics import execution_quality_report
from iqrp.app.execution.config import ExecutionSettings
from iqrp.app.execution.latency import LatencyTracker
from iqrp.app.execution.order_manager.child_order import create_child_order
from iqrp.app.execution.order_manager.execution_state import ExecutionState, transition_execution
from iqrp.app.execution.order_manager.fill_manager import Fill
from iqrp.app.execution.order_manager.order import Order, target_to_orders
from iqrp.app.execution.order_manager.order_manager import OrderManager
from iqrp.app.execution.order_manager.order_state import TERMINAL_STATES, OrderState
from iqrp.app.execution.order_manager.order_validator import ValidationResult
from iqrp.app.execution.order_manager.parent_order import ParentOrder
from iqrp.app.execution.order_manager.position_reconciliation import ReconciliationResult
from iqrp.app.execution.registry import get_algorithm
from iqrp.app.execution.serializer import ExecutionSerializer
from iqrp.app.execution.simulation import simulate_execution as sim_execution
from iqrp.app.execution.slippage.estimator import estimate_slippage as estimate_slippage_fn
from iqrp.app.execution.smart_routing.router import RoutingDecision, SmartRouter
from iqrp.app.execution.smart_routing.venue import (
    SimulatedVenue,
    Venue,
    VenueOrderRequest,
    VenueResponseStatus,
    as_venue,
)
from iqrp.app.execution.transaction_costs.total_cost import (
    post_trade_cost_analysis,
    pre_trade_cost_estimate,
)
from iqrp.app.execution.types import KillSwitch, OrderType, Urgency


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _instrument_ctx(
    market_context: Mapping[str, Any] | None,
    instrument: str,
) -> dict[str, Any]:
    """Resolve per-instrument market context from nested or flat maps."""
    ctx = dict(market_context or {})
    key = str(instrument).upper()
    nested = ctx.get(key) or ctx.get(instrument)
    if isinstance(nested, Mapping):
        base = {k: v for k, v in ctx.items() if not isinstance(v, Mapping)}
        base.update(dict(nested))
        return base
    return ctx


@dataclass
class ExecutionReport:
    """Result of ``ExecutionEngine.execute``."""

    execution_id: str
    status: str
    state: str
    algo: str
    parents: list[dict[str, Any]] = field(default_factory=list)
    children: list[dict[str, Any]] = field(default_factory=list)
    fills: list[dict[str, Any]] = field(default_factory=list)
    routing: list[dict[str, Any]] = field(default_factory=list)
    pre_trade: dict[str, Any] = field(default_factory=dict)
    post_trade: dict[str, Any] = field(default_factory=dict)
    analytics: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    audit: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "status": self.status,
            "state": self.state,
            "algo": self.algo,
            "parents": list(self.parents),
            "children": list(self.children),
            "fills": list(self.fills),
            "routing": list(self.routing),
            "pre_trade": dict(self.pre_trade),
            "post_trade": dict(self.post_trade),
            "analytics": dict(self.analytics),
            "latency": dict(self.latency),
            "audit": list(self.audit),
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }


class ExecutionEngine:
    """Orchestrate planning → validation → routing → submit → fill → analytics."""

    def __init__(
        self,
        settings: ExecutionSettings | None = None,
        order_manager: OrderManager | None = None,
        router: SmartRouter | None = None,
        kill_switch: KillSwitch | None = None,
        risk_engine: Any | None = None,
    ) -> None:
        self.settings = settings or ExecutionSettings.default()
        self.kill_switch = kill_switch or KillSwitch()
        self.risk_engine = risk_engine
        self.order_manager = order_manager or OrderManager(
            self.settings,
            kill_switch=self.kill_switch,
            validate_risk=self._make_risk_callback(),
        )
        # Share kill switch with OM if a custom OM was provided without one
        if kill_switch is not None:
            self.order_manager.kill_switch = self.kill_switch
        self.router = router or SmartRouter(kill_switch=self.kill_switch)
        if kill_switch is not None:
            self.router.kill_switch = self.kill_switch
        self.latency = LatencyTracker()
        self.serializer = ExecutionSerializer()
        self.state = ExecutionState.IDLE
        self._halted = False
        self._halt_reason = ""
        self._processed_events: set[str] = set()
        self._audit: list[dict[str, Any]] = []
        self._last_report: ExecutionReport | None = None

    # ------------------------------------------------------------------ helpers
    def _audit_event(self, action: str, **details: Any) -> None:
        self._audit.append({"action": action, "timestamp": _utc_now(), "details": dict(details)})

    def _make_risk_callback(self):
        if self.risk_engine is None:
            return None

        def _validate(order: Order) -> tuple[bool, str]:
            return self._check_risk(order)

        return _validate

    def _check_risk(self, order: Order) -> tuple[bool, str]:
        """Risk is authoritative when a risk_engine is provided."""
        eng = self.risk_engine
        if eng is None:
            return True, ""
        # Prefer validate_position / check_limits when available
        try:
            if hasattr(eng, "validate_position"):
                result = eng.validate_position(
                    {order.instrument: order.quantity * (1 if order.side.is_buy else -1)}
                )
                if isinstance(result, tuple) and len(result) >= 1:
                    ok = bool(result[0])
                    reason = str(result[1]) if len(result) > 1 else ""
                    return ok, reason
                if hasattr(result, "approved"):
                    return bool(result.approved), str(getattr(result, "reason", "") or "")
                if isinstance(result, dict):
                    ok = bool(result.get("ok", result.get("approved", True)))
                    return ok, str(result.get("reason", ""))
                if result is False:
                    return False, "risk_engine.validate_position rejected"
            if hasattr(eng, "check_limits"):
                result = eng.check_limits(
                    {order.instrument: order.quantity * (1 if order.side.is_buy else -1)}
                )
                if isinstance(result, tuple):
                    ok = bool(result[0])
                    return ok, str(result[1]) if len(result) > 1 else ""
                if isinstance(result, list) and result:
                    return False, f"limit breaches: {result}"
                if result is False:
                    return False, "risk_engine.check_limits rejected"
        except Exception as exc:
            if self.settings.risk.enforce_hard_limits:
                return False, f"risk_engine error: {exc}"
        return True, ""

    def _assert_not_halted(
        self,
        *,
        account_id: str | None = None,
        venue: str | None = None,
        strategy_id: str | None = None,
    ) -> None:
        if self._halted or self.state is ExecutionState.HALTED:
            raise ExecutionError(
                self._halt_reason or "execution halted",
                code="EXECUTION_HALTED",
            )
        blocked, reason = self.kill_switch.is_blocked(
            account_id=account_id, venue=venue, strategy_id=strategy_id
        )
        if blocked:
            raise ExecutionError(reason, code="KILL_SWITCH_ACTIVE")

    def _set_state(self, new_state: ExecutionState) -> None:
        self.state = transition_execution(self.state, new_state)

    def _seed_venue_quotes(
        self,
        venues: Sequence[Any],
        instrument: str,
        ctx: Mapping[str, Any],
    ) -> None:
        mid = float(ctx.get("mid", ctx.get("price", 0.0)) or 0.0)
        spread = float(ctx.get("spread", 0.02) or 0.02)
        adv = float(ctx.get("adv", 1e6) or 1e6)
        vol = float(ctx.get("volatility", 0.02) or 0.02)
        half = 0.5 * spread
        for v in venues:
            if isinstance(v, SimulatedVenue):
                st = v.get_state()
            elif isinstance(v, Venue):
                st = v.venue_state
            elif hasattr(v, "get_state"):
                st = v.get_state()
            else:
                continue
            if mid > 0:
                st.mid = mid
                st.bid = mid - half
                st.ask = mid + half
            st.adv = max(st.adv, adv)
            st.volatility = vol
            st.available_qty = max(st.available_qty, adv * 0.1, 1e6)
            st.instruments.add(str(instrument).upper())

    def _resolve_venues(
        self,
        venues: Sequence[Any] | None,
        instrument: str,
        ctx: Mapping[str, Any],
    ) -> list[Any]:
        if venues:
            out = list(venues)
        else:
            mid = float(ctx.get("mid", ctx.get("price", 100.0)) or 100.0)
            out = [
                SimulatedVenue(
                    venue_id=self.settings.default_venue,
                    instruments={str(instrument).upper()},
                    mode="fill",
                    mid=mid,
                    spread=float(ctx.get("spread", 0.02) or 0.02),
                    adv=float(ctx.get("adv", 1e6) or 1e6),
                )
            ]
        self._seed_venue_quotes(out, instrument, ctx)
        return out

    def _algo_order_type(self, algo: str) -> OrderType:
        key = str(algo).strip().lower()
        if key in {"market"}:
            return OrderType.MARKET
        return OrderType.LIMIT

    # --------------------------------------------------------------- planning
    def plan_from_targets(
        self,
        current_positions: dict[str, float],
        target_positions: dict[str, float],
        **kwargs: Any,
    ) -> list[Order]:
        """Convert approved current→target deltas into orders (no alpha)."""
        self._assert_not_halted()
        specs = target_to_orders(current_positions, target_positions, **kwargs)
        orders = [self.order_manager.create_from_spec(s) for s in specs]
        self._audit_event(
            "plan_from_targets",
            n_orders=len(orders),
            instruments=sorted(set(current_positions) | set(target_positions)),
        )
        return orders

    # -------------------------------------------------------- cost / slippage
    def estimate_costs(
        self,
        orders_or_delta: Sequence[Order] | Mapping[str, float] | Order,
        market_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Pre-trade cost estimate for orders or a delta position map."""
        orders = self._coerce_orders(orders_or_delta, market_context)
        estimates: list[dict[str, Any]] = []
        total = 0.0
        for o in orders:
            ctx = _instrument_ctx(market_context, o.instrument)
            mid = float(ctx.get("mid", ctx.get("price", o.price or 0.0)) or 0.0)
            est = pre_trade_cost_estimate(
                side=o.side.value,
                quantity=o.quantity,
                mid=mid if mid > 0 else float(o.price or 1.0),
                spread=float(ctx.get("spread", 0.0) or 0.0),
                adv=float(ctx.get("adv", 1e6) or 1e6),
                volatility=float(ctx.get("volatility", 0.02) or 0.02),
            )
            estimates.append({"instrument": o.instrument, "order_id": o.order_id, **est})
            total += float(est.get("total_cost", 0.0))
        return {"orders": estimates, "total_cost": total}

    def estimate_slippage(
        self,
        orders_or_delta: Sequence[Order] | Mapping[str, float] | Order | None = None,
        market_context: Mapping[str, Any] | None = None,
        *,
        side: str | None = None,
        quantity: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if orders_or_delta is None and side is not None and quantity is not None:
            ctx = dict(market_context or {})
            return estimate_slippage_fn(
                side=side,
                quantity=quantity,
                mid=float(ctx.get("mid", ctx.get("price", 100.0))),
                spread=float(ctx.get("spread", 0.0) or 0.0),
                adv=float(ctx.get("adv", 1e6) or 1e6),
                volatility=float(ctx.get("volatility", 0.02) or 0.02),
                **kwargs,
            )
        orders = self._coerce_orders(orders_or_delta or [], market_context)
        out: list[dict[str, Any]] = []
        for o in orders:
            ctx = _instrument_ctx(market_context, o.instrument)
            mid = float(ctx.get("mid", ctx.get("price", o.price or 100.0)) or 100.0)
            slip = estimate_slippage_fn(
                side=o.side.value,
                quantity=o.quantity,
                mid=mid,
                spread=float(ctx.get("spread", 0.0) or 0.0),
                adv=float(ctx.get("adv", 1e6) or 1e6),
                volatility=float(ctx.get("volatility", 0.02) or 0.02),
                **kwargs,
            )
            out.append({"instrument": o.instrument, "order_id": o.order_id, **slip})
        return {"orders": out}

    def _coerce_orders(
        self,
        orders_or_delta: Sequence[Order] | Mapping[str, float] | Order,
        market_context: Mapping[str, Any] | None,
    ) -> list[Order]:
        if isinstance(orders_or_delta, Order):
            return [orders_or_delta]
        if isinstance(orders_or_delta, Mapping):
            # Treat as target deltas from flat (current=0)
            prices = {}
            for inst, qty in orders_or_delta.items():
                ctx = _instrument_ctx(market_context, str(inst))
                if "mid" in ctx or "price" in ctx:
                    prices[str(inst)] = float(ctx.get("mid", ctx.get("price", 0.0)))
            specs = target_to_orders(
                dict.fromkeys(orders_or_delta, 0.0),
                {k: float(v) for k, v in orders_or_delta.items()},
                prices=prices or None,
            )
            return [
                Order(
                    instrument=s.instrument,
                    side=s.side,
                    quantity=s.quantity,
                    price=s.price,
                    order_type=s.order_type,
                    urgency=s.urgency,
                )
                for s in specs
            ]
        return list(orders_or_delta)

    # ----------------------------------------------------------------- execute
    def execute(
        self,
        parent_order_or_targets: ParentOrder | Order | Mapping[str, float] | Sequence[Order],
        *,
        algo: str = "twap",
        urgency: Urgency | str = Urgency.NORMAL,
        venues: Sequence[Any] | None = None,
        market_context: Mapping[str, Any] | None = None,
        current: Mapping[str, float] | None = None,
        simulation_mode: bool | None = None,
        account_id: str | None = None,
        strategy_id: str | None = None,
        **kwargs: Any,
    ) -> ExecutionReport:
        """Full execution flow: plan → validate → estimate → slice → route → submit → analyze."""
        execution_id = f"ex_{uuid4().hex[:16]}"
        urg = coerce_urgency(urgency, Urgency(self.settings.default_urgency))
        audit: list[dict[str, Any]] = []
        errors: list[str] = []

        def _a(action: str, **d: Any) -> None:
            audit.append({"action": action, "timestamp": _utc_now(), "details": dict(d)})

        # Kill / halt is fail-safe: raise so callers cannot proceed
        self._assert_not_halted(account_id=account_id, strategy_id=strategy_id)

        # Reset to IDLE if previous run completed
        if self.state in {
            ExecutionState.COMPLETED,
            ExecutionState.CANCELLED,
            ExecutionState.FAILED,
        }:
            self.state = ExecutionState.IDLE

        try:
            self._set_state(ExecutionState.PLANNING)
        except ExecutionError:
            self.state = ExecutionState.PLANNING

        # --- Resolve parent orders from targets / order / parent ---
        parents: list[ParentOrder] = []
        if isinstance(parent_order_or_targets, ParentOrder):
            parents = [parent_order_or_targets]
        elif isinstance(parent_order_or_targets, Order):
            po = ParentOrder(
                instrument=parent_order_or_targets.instrument,
                side=parent_order_or_targets.side,
                quantity=parent_order_or_targets.quantity,
                urgency=urg,
                algo=str(algo),
                strategy_id=strategy_id or parent_order_or_targets.strategy_id,
                portfolio_id=parent_order_or_targets.portfolio_id,
            )
            parents = [po]
        elif isinstance(parent_order_or_targets, Mapping):
            cur = dict(current or dict.fromkeys(parent_order_or_targets, 0.0))
            tgt = {str(k).upper(): float(v) for k, v in parent_order_or_targets.items()}
            cur = {str(k).upper(): float(v) for k, v in cur.items()}
            prices = {}
            for inst in set(cur) | set(tgt):
                ctx = _instrument_ctx(market_context, inst)
                px = ctx.get("mid", ctx.get("price"))
                if px is not None:
                    prices[inst] = float(px)
            specs = target_to_orders(
                cur,
                tgt,
                prices=prices or None,
                urgency=urg,
                **{k: kwargs[k] for k in ("lot_size", "min_qty", "round_lots") if k in kwargs},
            )
            for s in specs:
                parents.append(
                    ParentOrder(
                        instrument=s.instrument,
                        side=s.side,
                        quantity=s.quantity,
                        urgency=urg,
                        algo=str(algo),
                        strategy_id=strategy_id,
                        metadata={"current": s.tags.get("current"), "target": s.tags.get("target")},
                    )
                )
        else:
            for o in parent_order_or_targets:
                parents.append(
                    ParentOrder(
                        instrument=o.instrument,
                        side=o.side,
                        quantity=o.quantity,
                        urgency=urg,
                        algo=str(algo),
                        strategy_id=strategy_id or o.strategy_id,
                    )
                )

        if not parents:
            self.state = ExecutionState.COMPLETED
            report = ExecutionReport(
                execution_id=execution_id,
                status="EMPTY",
                state=self.state.value,
                algo=str(algo),
                audit=audit,
            )
            self._last_report = report
            return report

        for p in parents:
            self.order_manager.register_parent(p)
        _a("planned", n_parents=len(parents))

        # VALIDATING
        try:
            self._set_state(ExecutionState.VALIDATING)
        except ExecutionError:
            self.state = ExecutionState.VALIDATING

        algorithm = get_algorithm(str(algo))
        child_orders: list[Order] = []
        routing_decisions: list[dict[str, Any]] = []
        fill_dicts: list[dict[str, Any]] = []
        pre_trade: dict[str, Any] = {"by_parent": []}
        post_trade: dict[str, Any] = {"by_parent": []}
        analytics_out: dict[str, Any] = {"by_parent": []}
        status = "PENDING"

        sim_mode = simulation_mode
        if sim_mode is None:
            sim_mode = (
                bool(venues and all(isinstance(v, SimulatedVenue) for v in venues))
                or venues is None
            )

        try:
            for parent in parents:
                # Kill / halt re-check before each parent
                self._assert_not_halted(
                    account_id=account_id,
                    strategy_id=strategy_id or parent.strategy_id,
                )

                ctx = _instrument_ctx(market_context, parent.instrument)
                ctx = dict(ctx)
                ctx.setdefault("side", "buy" if parent.side.is_buy else "sell")
                ctx.setdefault("urgency", urg.value)
                ctx.setdefault("residual", parent.residual_qty)
                ctx.setdefault("approved_quantity", parent.residual_qty)

                venue_list = self._resolve_venues(venues, parent.instrument, ctx)

                # Pre-trade estimates
                mid = float(ctx.get("mid", ctx.get("price", 100.0)) or 100.0)
                cost_est = pre_trade_cost_estimate(
                    side=parent.side.value,
                    quantity=parent.quantity,
                    mid=mid,
                    spread=float(ctx.get("spread", 0.0) or 0.0),
                    adv=float(ctx.get("adv", 1e6) or 1e6),
                    volatility=float(ctx.get("volatility", 0.02) or 0.02),
                )
                slip_est = estimate_slippage_fn(
                    side=parent.side.value,
                    quantity=parent.quantity,
                    mid=mid,
                    spread=float(ctx.get("spread", 0.0) or 0.0),
                    adv=float(ctx.get("adv", 1e6) or 1e6),
                    volatility=float(ctx.get("volatility", 0.02) or 0.02),
                )
                pre_trade["by_parent"].append(
                    {
                        "parent_id": parent.parent_id,
                        "instrument": parent.instrument,
                        "costs": cost_est,
                        "slippage": slip_est,
                    }
                )
                _a("pre_trade", parent_id=parent.parent_id, total_cost=cost_est.get("total_cost"))

                # Risk gate on synthetic parent order
                probe = Order(
                    instrument=parent.instrument,
                    side=parent.side,
                    quantity=parent.quantity,
                    order_type=self._algo_order_type(algo),
                    price=mid,
                    urgency=urg,
                    strategy_id=strategy_id or parent.strategy_id,
                    account_id=account_id,
                    algo=str(algo),
                )
                ok, reason = self._check_risk(probe)
                if not ok:
                    raise ExecutionError(reason or "risk reject", code="HARD_RISK_REJECT")

                vres = self.order_manager.validator.validate(probe)
                if not vres.ok:
                    raise ValidationError(
                        "; ".join(vres.errors),
                        code="ORDER_VALIDATION_FAILED",
                        details={"errors": vres.errors},
                    )

                # Algorithm plan → child slices (never exceed residual)
                slices: list[ChildSlice] = algorithm.plan(parent.residual_qty, ctx)
                total_planned = float(sum(s.quantity for s in slices))
                if total_planned > parent.residual_qty + 1e-6:
                    raise ExecutionError(
                        f"algo planned {total_planned} exceeds residual {parent.residual_qty}",
                        code="RESIDUAL_EXCEEDED",
                    )
                _a("algo_plan", parent_id=parent.parent_id, n_slices=len(slices), qty=total_planned)

                try:
                    self._set_state(ExecutionState.EXECUTING)
                except ExecutionError:
                    self.state = ExecutionState.EXECUTING

                parent_fills: list[dict[str, Any]] = []
                otype = self._algo_order_type(algo)

                for idx, sl in enumerate(slices):
                    self._assert_not_halted(
                        account_id=account_id,
                        strategy_id=strategy_id or parent.strategy_id,
                    )
                    # Hard residual clip
                    qty = min(float(sl.quantity), parent.residual_qty)
                    if qty <= 0:
                        continue

                    child = create_child_order(
                        parent,
                        quantity=qty,
                        order_type=otype,
                        price=sl.limit_price_hint if otype is OrderType.LIMIT else None,
                        urgency=sl.urgency,
                        algo=str(algo),
                        tags={"slice": idx, "not_before_offset": sl.not_before_offset},
                    )
                    # Register in OM
                    self.order_manager._orders[child.order_id] = child
                    if child.idempotency_key:
                        self.order_manager._by_idempotency[child.idempotency_key] = child.order_id

                    self.latency.start(child.order_id)
                    self.order_manager.validate_and_approve(child.order_id)

                    # Route
                    decision = self.router.route(child, venue_list)
                    routing_decisions.append(decision.to_dict())
                    if not decision.accepted:
                        reasons = [r.message for r in decision.rejections]
                        errors.append(f"routing rejected {child.order_id}: {reasons}")
                        _a("route_reject", order_id=child.order_id, reasons=reasons)
                        continue

                    primary = decision.primary_venue_id or self.settings.default_venue
                    child.venue = primary

                    # Submit
                    self.latency.mark_submit(child.order_id)
                    self.order_manager.submit(child.order_id, venue=primary)
                    _a("submit", order_id=child.order_id, venue=primary)

                    # Simulate fill via SimulatedVenue when applicable
                    venue_obj = self._find_venue(venue_list, primary)
                    if sim_mode or isinstance(venue_obj, SimulatedVenue):
                        self._simulate_venue_fill(
                            child,
                            venue_obj,
                            venue_list,
                            primary,
                            fill_dicts,
                            parent_fills,
                            _a,
                        )
                    child_orders.append(child)

                parent.sync_fills_from_children(child_orders)

                # Post-trade
                post = (
                    post_trade_cost_analysis(
                        parent_fills,
                        side=parent.side.value,
                        arrival_price=mid,
                    )
                    if parent_fills
                    else {}
                )
                post_trade["by_parent"].append(
                    {"parent_id": parent.parent_id, "instrument": parent.instrument, **post}
                )
                lat_sum = self.latency.summary(
                    [c.order_id for c in child_orders if c.parent_id == parent.parent_id]
                )
                aq = execution_quality_report(
                    side=parent.side.value,
                    ordered_qty=parent.quantity,
                    fills=parent_fills,
                    arrival_price=mid,
                    vwap_benchmark=float(ctx.get("vwap", mid)),
                    twap_benchmark=float(ctx.get("twap", mid)),
                    latency=lat_sum,
                    pre_trade_estimate=cost_est,
                    post_trade_costs=post,
                )
                analytics_out["by_parent"].append(
                    {"parent_id": parent.parent_id, "instrument": parent.instrument, **aq}
                )

            # Final state
            any_fill = bool(fill_dicts)
            all_done = all(p.residual_qty <= 1e-9 for p in parents) if parents else True
            if errors and not any_fill:
                try:
                    self._set_state(ExecutionState.FAILED)
                except ExecutionError:
                    self.state = ExecutionState.FAILED
                status = "FAILED"
            elif all_done and any_fill:
                try:
                    self._set_state(ExecutionState.COMPLETED)
                except ExecutionError:
                    self.state = ExecutionState.COMPLETED
                status = "FILLED"
            elif any_fill:
                try:
                    self._set_state(ExecutionState.PARTIALLY_EXECUTED)
                except ExecutionError:
                    self.state = ExecutionState.PARTIALLY_EXECUTED
                status = "PARTIAL"
            else:
                try:
                    self._set_state(ExecutionState.COMPLETED)
                except ExecutionError:
                    self.state = ExecutionState.COMPLETED
                status = "COMPLETED"

        except (ExecutionError, ValidationError) as exc:
            errors.append(str(exc))
            try:
                if self.state not in {ExecutionState.HALTED, ExecutionState.FAILED}:
                    if can_fail(self.state):
                        self.state = ExecutionState.FAILED
            except Exception:
                self.state = ExecutionState.FAILED
            status = "BLOCKED" if getattr(exc, "code", "") == "KILL_SWITCH_ACTIVE" else "FAILED"
            # Re-raise kill switch so smoke test can catch it
            if getattr(exc, "code", "") in {"KILL_SWITCH_ACTIVE", "EXECUTION_HALTED"}:
                raise

        lat = self.latency.summary([c.order_id for c in child_orders])
        report = ExecutionReport(
            execution_id=execution_id,
            status=status,
            state=self.state.value,
            algo=str(algo),
            parents=[p.to_dict() for p in parents],
            children=[c.to_dict() for c in child_orders],
            fills=fill_dicts,
            routing=routing_decisions,
            pre_trade=pre_trade,
            post_trade=post_trade,
            analytics=analytics_out,
            latency=lat,
            audit=audit,
            errors=errors,
            metadata={"urgency": urg.value, "simulation_mode": bool(sim_mode)},
        )
        self._last_report = report
        self._audit.extend(audit)
        return report

    def _find_venue(self, venues: Sequence[Any], venue_id: str) -> Any | None:
        for v in venues:
            vid = getattr(v, "venue_id", None)
            if vid is None and isinstance(v, dict):
                vid = v.get("venue_id")
            if str(vid) == str(venue_id):
                return v
            if isinstance(v, Venue) and v.venue_id == venue_id:
                return v
        return venues[0] if venues else None

    def _simulate_venue_fill(
        self,
        child: Order,
        venue_obj: Any,
        venue_list: Sequence[Any],
        primary: str,
        fill_dicts: list[dict[str, Any]],
        parent_fills: list[dict[str, Any]],
        audit_fn,
    ) -> None:
        req = VenueOrderRequest(
            instrument=child.instrument,
            side=child.side,
            quantity=child.quantity,
            order_type=child.order_type,
            price=child.price,
            client_order_id=child.client_order_id or child.order_id,
            order_id=child.order_id,
        )
        if isinstance(venue_obj, SimulatedVenue) or hasattr(venue_obj, "submit"):
            resp = venue_obj.submit(req)
        else:
            # Synthetic fill at mid
            st = as_venue(venue_obj).venue_state if venue_obj else None
            px = (st.mid if st and st.mid else child.price) or 0.0
            from iqrp.app.execution.smart_routing.venue import VenueResponse

            resp = VenueResponse(
                status=VenueResponseStatus.FILL,
                venue_id=primary,
                venue_order_id=f"{primary}-syn",
                client_order_id=child.order_id,
                filled_qty=child.quantity,
                fill_price=float(px),
            )

        event_ack = f"ack|{child.order_id}|{resp.venue_order_id}"
        self.latency.mark_ack(child.order_id)
        if child.state is OrderState.SUBMITTED:
            self.order_manager.acknowledge(
                child.order_id, venue_order_id=resp.venue_order_id, event_id=event_ack
            )

        if resp.status in {VenueResponseStatus.FILL, VenueResponseStatus.PARTIAL_FILL}:
            event_fill = f"fill|{child.order_id}|{resp.venue_order_id}|{resp.filled_qty}"
            self.latency.mark_fill(child.order_id)
            self.order_manager.apply_fill(
                child.order_id,
                fill_qty=float(resp.filled_qty),
                fill_price=float(resp.fill_price or child.price or 0.0),
                event_id=event_fill,
                venue_exec_id=resp.venue_order_id,
            )
            fd = {
                "order_id": child.order_id,
                "instrument": child.instrument,
                "quantity": float(resp.filled_qty),
                "qty": float(resp.filled_qty),
                "price": float(resp.fill_price or 0.0),
                "fill_price": float(resp.fill_price or 0.0),
                "venue": primary,
                "event_id": event_fill,
            }
            fill_dicts.append(fd)
            parent_fills.append(fd)
            audit_fn("fill", order_id=child.order_id, qty=resp.filled_qty, price=resp.fill_price)
        elif resp.status is VenueResponseStatus.REJECT:
            audit_fn("venue_reject", order_id=child.order_id, reason=resp.reason)

    # ------------------------------------------------------------------- route
    def route(self, order: Order, venues: Sequence[Any]) -> RoutingDecision:
        self._assert_not_halted(
            account_id=order.account_id,
            strategy_id=order.strategy_id,
            venue=order.venue,
        )
        return self.router.route(order, venues)

    def validate_order(self, order: Order) -> ValidationResult:
        return self.order_manager.validator.validate(order)

    # ----------------------------------------------------------- events / recon
    def apply_event(self, event_id: str, **kwargs: Any) -> Order | dict[str, Any]:
        """Idempotent event application."""
        if event_id in self._processed_events:
            self._audit_event("event_idempotent_skip", event_id=event_id)
            order_id = kwargs.get("order_id")
            if order_id:
                return self.order_manager.get(str(order_id))
            return {"event_id": event_id, "status": "duplicate"}
        event_type = str(kwargs.get("event_type", kwargs.get("type", "fill")))
        order_id = str(kwargs.get("order_id", ""))
        payload = dict(kwargs.get("payload") or {})
        for k in ("fill_qty", "fill_price", "venue_order_id", "venue_exec_id", "reason"):
            if k in kwargs and k not in payload:
                payload[k] = kwargs[k]
        order = self.order_manager.process_event(
            event_id, event_type, order_id=order_id, payload=payload
        )
        self._processed_events.add(event_id)
        self._audit_event(
            "apply_event", event_id=event_id, event_type=event_type, order_id=order_id
        )
        return order

    def reconcile(
        self,
        expected: Mapping[str, float],
        executed: Mapping[str, float],
        broker: Mapping[str, float] | None = None,
    ) -> ReconciliationResult:
        return self.order_manager.reconcile_positions(
            expected=dict(expected),
            executed=dict(executed),
            broker=dict(broker or executed),
        )

    # ----------------------------------------------------------- halt / kill
    def halt(self, reason: str, *, cancel_open: bool = True) -> None:
        self._halted = True
        self._halt_reason = str(reason)
        self.kill_switch.engage_global(reason)
        try:
            if self.state not in {
                ExecutionState.COMPLETED,
                ExecutionState.CANCELLED,
                ExecutionState.FAILED,
                ExecutionState.HALTED,
            }:
                self.state = ExecutionState.HALTED
            else:
                self.state = ExecutionState.HALTED
        except Exception:
            self.state = ExecutionState.HALTED
        self.router.kill_switch = self.kill_switch
        self.order_manager.kill_switch = self.kill_switch
        if cancel_open:
            for order in self.order_manager.list_orders():
                if order.state not in TERMINAL_STATES and order.state not in {
                    OrderState.CREATED,
                    OrderState.REJECTED,
                }:
                    try:
                        self.order_manager.cancel(order.order_id, reason=f"halt: {reason}")
                    except Exception:
                        pass
        self._audit_event("halt", reason=reason, cancel_open=cancel_open)

    def kill(
        self,
        scope: str = "global",
        key: str | None = None,
        reason: str = "kill",
    ) -> None:
        scope_l = str(scope).strip().lower()
        if scope_l == "global":
            self.kill_switch.engage_global(reason)
            self._halted = True
            self._halt_reason = reason
            self.state = ExecutionState.HALTED
        elif scope_l == "account":
            if not key:
                raise ValueError("account kill requires key")
            self.kill_switch.engage_account(key, reason)
        elif scope_l == "venue":
            if not key:
                raise ValueError("venue kill requires key")
            self.kill_switch.engage_venue(key, reason)
        elif scope_l == "strategy":
            if not key:
                raise ValueError("strategy kill requires key")
            self.kill_switch.engage_strategy(key, reason)
        else:
            raise ValueError(f"unknown kill scope: {scope}")
        self.router.kill_switch = self.kill_switch
        self.order_manager.kill_switch = self.kill_switch
        self._audit_event("kill", scope=scope_l, key=key, reason=reason)

    # -------------------------------------------------------------- analytics
    def analytics(
        self,
        fills: Sequence[Mapping[str, Any]] | Sequence[Fill],
        arrival_price: float,
        *,
        side: str = "buy",
        ordered_qty: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        fill_maps: list[dict[str, Any]] = []
        for f in fills:
            if isinstance(f, Fill):
                fill_maps.append(f.to_dict())
            else:
                fill_maps.append(dict(f))
        qty = ordered_qty
        if qty is None:
            qty = float(
                sum(
                    abs(float(f.get("fill_qty", f.get("quantity", f.get("qty", 0.0)))))
                    for f in fill_maps
                )
            )
        return execution_quality_report(
            side=side,
            ordered_qty=float(qty),
            fills=fill_maps,
            arrival_price=float(arrival_price),
            **{
                k: kwargs[k]
                for k in (
                    "vwap_benchmark",
                    "twap_benchmark",
                    "latency",
                    "pre_trade_estimate",
                    "post_trade_costs",
                )
                if k in kwargs
            },
        )

    def simulate_execution(self, **kwargs: Any) -> dict[str, Any]:
        return sim_execution(**kwargs)

    # --------------------------------------------------------------- save/load
    def save(self, path: str | Path) -> Path:
        payload = {
            "settings": self.settings.model_dump(),
            "kill_switch": self.kill_switch.to_dict(),
            "state": self.state.value,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "processed_events": sorted(self._processed_events),
            "audit": list(self._audit),
            "orders": [o.to_dict() for o in self.order_manager.list_orders()],
            "last_report": self._last_report.to_dict() if self._last_report else None,
            "latency": self.latency.to_dict(),
        }
        return self.serializer.save(payload, path)

    def load(self, path: str | Path) -> dict[str, Any]:
        data = self.serializer.load(path)
        if "kill_switch" in data:
            ks = data["kill_switch"]
            self.kill_switch.global_halt = bool(ks.get("global_halt", False))
            self.kill_switch.reason = str(ks.get("reason", ""))
            self.kill_switch.accounts = set(ks.get("accounts") or [])
            self.kill_switch.venues = set(ks.get("venues") or [])
            self.kill_switch.strategies = set(ks.get("strategies") or [])
        if "state" in data:
            try:
                self.state = ExecutionState(data["state"])
            except Exception:
                pass
        self._halted = bool(data.get("halted", False))
        self._halt_reason = str(data.get("halt_reason", ""))
        self._processed_events = set(data.get("processed_events") or [])
        self._audit = list(data.get("audit") or [])
        for od in data.get("orders") or []:
            order = Order.from_dict(od)
            self.order_manager._orders[order.order_id] = order
        return data


def can_fail(state: ExecutionState) -> bool:
    return state in {
        ExecutionState.IDLE,
        ExecutionState.PLANNING,
        ExecutionState.VALIDATING,
        ExecutionState.EXECUTING,
        ExecutionState.PARTIALLY_EXECUTED,
        ExecutionState.COMPLETING,
        ExecutionState.HALTED,
    }


__all__ = ["ExecutionEngine", "ExecutionReport"]

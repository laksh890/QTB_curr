"""Institutional Execution Order Manager.

CRITICAL RULES
--------------
- Execution never generates alpha.
- Never override hard risk limits.
- Urgency influences aggressiveness but NEVER overrides hard risk.
- Idempotent fills/events via event_id / idempotency_key.
- No future information.
- Kill-switch flags (global/account/venue/strategy) are checked before submit
  and cannot be overridden by urgency or confidence.
"""

from __future__ import annotations

from typing import Any, Callable
from uuid import uuid4

from iqrp.app.core.exceptions import ExecutionError, ValidationError
from iqrp.app.execution.config import ExecutionSettings
from iqrp.app.execution.order_manager.audit import AuditLog
from iqrp.app.execution.order_manager.cancel_replace import (
    CancelRequest,
    ReplaceRequest,
    begin_cancel,
    build_replacement,
)
from iqrp.app.execution.order_manager.fill_manager import Fill, FillManager
from iqrp.app.execution.order_manager.order import Order, OrderSpec, target_to_orders
from iqrp.app.execution.order_manager.order_group import OrderGroup
from iqrp.app.execution.order_manager.order_lifecycle import (
    approve,
    begin_validation,
    mark_acknowledged,
    mark_cancelled,
    mark_failed,
    mark_rejected,
    mark_submitted,
)
from iqrp.app.execution.order_manager.order_state import OrderState
from iqrp.app.execution.order_manager.order_validator import OrderValidator
from iqrp.app.execution.order_manager.parent_order import ParentOrder
from iqrp.app.execution.order_manager.position_reconciliation import PositionReconciler
from iqrp.app.execution.types import KillSwitch, OrderType, Side, TimeInForce, Urgency


class OrderManager:
    """Create, validate, submit, acknowledge, fill, cancel, and replace orders.

    Event processing is idempotent: duplicate ``event_id`` / ``idempotency_key``
    values are ignored. Kill-switch and hard risk gates are never overridden.
    """

    def __init__(
        self,
        settings: ExecutionSettings | None = None,
        *,
        validator: OrderValidator | None = None,
        kill_switch: KillSwitch | None = None,
        audit: AuditLog | None = None,
        validate_risk: Callable[[Order], tuple[bool, str]] | None = None,
    ) -> None:
        self.settings = settings or ExecutionSettings.default()
        self.audit = audit or AuditLog()
        self.kill_switch = kill_switch or KillSwitch()
        self.validator = validator or OrderValidator(
            self.settings, validate_risk=validate_risk
        )
        if validate_risk is not None and self.validator.validate_risk is None:
            self.validator.validate_risk = validate_risk
        self.fills = FillManager(allow_overfill=self.settings.fills.allow_overfill)
        self.fill = self.fills  # alias
        self.reconciler = PositionReconciler(
            qty_tolerance=self.settings.reconciliation.qty_tolerance,
            notional_tolerance=self.settings.reconciliation.notional_tolerance,
            alert_on_diff=self.settings.reconciliation.alert_on_diff,
        )
        self._orders: dict[str, Order] = {}
        self._by_idempotency: dict[str, str] = {}
        self._processed_events: set[str] = set()
        self._parents: dict[str, ParentOrder] = {}
        self._groups: dict[str, OrderGroup] = {}

    # ------------------------------------------------------------------ create
    def create_order(
        self,
        *,
        instrument: str,
        side: Side | str,
        quantity: float,
        order_type: OrderType | str = OrderType.LIMIT,
        price: float | None = None,
        stop_price: float | None = None,
        time_in_force: TimeInForce | str | None = None,
        venue: str | None = None,
        algo: str | None = None,
        urgency: Urgency | str | None = None,
        strategy_id: str | None = None,
        portfolio_id: str | None = None,
        parent_id: str | None = None,
        account_id: str | None = None,
        client_order_id: str | None = None,
        idempotency_key: str | None = None,
        tags: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Order:
        """Create an order in CREATED state (idempotent on idempotency_key)."""
        if isinstance(side, str):
            side = Side(side)
        if isinstance(order_type, str):
            order_type = OrderType(order_type)
        tif = time_in_force or TimeInForce(self.settings.default_time_in_force)
        if isinstance(tif, str):
            tif = TimeInForce(tif)
        urg = urgency or Urgency(self.settings.default_urgency)
        if isinstance(urg, str):
            urg = Urgency(urg)

        order = Order(
            instrument=instrument,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
            stop_price=stop_price,
            time_in_force=tif,
            venue=venue or self.settings.default_venue,
            algo=algo,
            urgency=urg,
            strategy_id=strategy_id,
            portfolio_id=portfolio_id,
            parent_id=parent_id,
            account_id=account_id,
            client_order_id=client_order_id,
            idempotency_key=idempotency_key,
            tags=dict(tags or {}),
            metadata=dict(metadata or {}),
        )

        if order.idempotency_key and order.idempotency_key in self._by_idempotency:
            existing_id = self._by_idempotency[order.idempotency_key]
            self.audit.append(
                "create_idempotent_hit",
                f"returning existing order {existing_id}",
                order_id=existing_id,
                details={"idempotency_key": order.idempotency_key},
            )
            return self._orders[existing_id]

        self._orders[order.order_id] = order
        if order.idempotency_key:
            self._by_idempotency[order.idempotency_key] = order.order_id
        order.audit.append({"event": "created", "state": order.state.value})
        self.audit.append(
            "created",
            f"created order {order.order_id}",
            order_id=order.order_id,
            details={"instrument": order.instrument, "side": order.side.value, "qty": order.quantity},
        )
        return order

    def create_from_spec(self, spec: OrderSpec) -> Order:
        return self.create_order(**spec.to_order_kwargs())

    def create_from_target(
        self,
        current: dict[str, float],
        target: dict[str, float],
        **kwargs: Any,
    ) -> list[Order]:
        specs = target_to_orders(current, target, **kwargs)
        return [self.create_from_spec(s) for s in specs]

    # --------------------------------------------------------------- validation
    def validate_and_approve(self, order_id: str) -> Order:
        order = self.get(order_id)
        begin_validation(order, audit=self.audit)
        try:
            result = self.validator.validate(order)
            if not result.ok:
                mark_rejected(order, reason="; ".join(result.errors), audit=self.audit)
                raise ValidationError(
                    "; ".join(result.errors),
                    code="ORDER_VALIDATION_FAILED",
                    details={"errors": result.errors, "order_id": order_id},
                )
            approve(order, audit=self.audit)
        except ValidationError:
            if order.state is not OrderState.REJECTED:
                mark_rejected(order, reason="validation error", audit=self.audit)
            raise
        except Exception as exc:  # noqa: BLE001
            mark_failed(order, reason=str(exc), audit=self.audit)
            raise ExecutionError(
                f"validation failed: {exc}",
                code="ORDER_VALIDATION_ERROR",
                details={"order_id": order_id},
            ) from exc
        return order

    # ------------------------------------------------------------------- submit
    def submit(self, order_id: str, *, venue: str | None = None, event_id: str | None = None) -> Order:
        """Submit an APPROVED order. Kill-switches are hard gates."""
        if event_id and self._is_duplicate_event(event_id, order_id, "submit"):
            return self.get(order_id)

        order = self.get(order_id)
        if order.state is not OrderState.APPROVED:
            raise ExecutionError(
                f"submit requires APPROVED state, got {order.state.value}",
                code="ORDER_SUBMIT_BAD_STATE",
                details={"order_id": order_id, "state": order.state.value},
            )

        venue = venue or order.venue or self.settings.default_venue
        if self.settings.kill_switch.check_on_submit:
            blocked, reason = self.kill_switch.is_blocked(
                account_id=order.account_id if self.settings.kill_switch.check_account else None,
                venue=venue if self.settings.kill_switch.check_venue else None,
                strategy_id=order.strategy_id if self.settings.kill_switch.check_strategy else None,
            )
            # global always checked when check_global
            if self.settings.kill_switch.check_global and self.kill_switch.global_halt:
                blocked, reason = True, self.kill_switch.reason or "global kill-switch active"
            if blocked:
                mark_failed(order, reason=reason, audit=self.audit)
                raise ExecutionError(
                    reason,
                    code="KILL_SWITCH_ACTIVE",
                    details={
                        "order_id": order_id,
                        "venue": venue,
                        "account_id": order.account_id,
                        "strategy_id": order.strategy_id,
                    },
                )

        # Re-check hard risk immediately before submit — urgency never overrides
        if self.validator.validate_risk is not None and self.settings.risk.enforce_hard_limits:
            ok, reason = self.validator.validate_risk(order)
            if not ok:
                mark_rejected(order, reason=reason or "hard risk reject on submit", audit=self.audit)
                raise ExecutionError(
                    reason or "hard risk reject on submit",
                    code="HARD_RISK_REJECT",
                    details={"order_id": order_id},
                )

        mark_submitted(order, venue=venue, audit=self.audit)
        if event_id:
            self._processed_events.add(event_id)
        return order

    def acknowledge(
        self,
        order_id: str,
        *,
        venue_order_id: str | None = None,
        event_id: str | None = None,
    ) -> Order:
        if event_id and self._is_duplicate_event(event_id, order_id, "acknowledge"):
            return self.get(order_id)
        order = self.get(order_id)
        if order.state is not OrderState.SUBMITTED:
            raise ExecutionError(
                f"acknowledge requires SUBMITTED, got {order.state.value}",
                code="ORDER_ACK_BAD_STATE",
                details={"order_id": order_id, "state": order.state.value},
            )
        mark_acknowledged(order, venue_order_id=venue_order_id, audit=self.audit)
        if event_id:
            self._processed_events.add(event_id)
        return order

    # -------------------------------------------------------------------- fills
    def apply_fill(
        self,
        order_id: str,
        *,
        fill_qty: float,
        fill_price: float,
        event_id: str,
        venue_exec_id: str | None = None,
        liquidity_flag: str | None = None,
        fees: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> Order:
        """Apply a fill idempotently. Duplicate ``event_id`` is a no-op."""
        order = self.get(order_id)
        if order.state not in {
            OrderState.SUBMITTED,
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIALLY_FILLED,
            OrderState.CANCEL_PENDING,
        }:
            # Idempotent re-delivery after fill still returns current order
            if self.fills.seen(event_id):
                return order
            raise ExecutionError(
                f"cannot fill order in state {order.state.value}",
                code="ORDER_FILL_BAD_STATE",
                details={"order_id": order_id, "state": order.state.value},
            )

        self.fills.apply_fill(
            order,
            fill_qty=fill_qty,
            fill_price=fill_price,
            event_id=event_id,
            audit=self.audit,
            venue_exec_id=venue_exec_id,
            liquidity_flag=liquidity_flag,
            fees=fees,
            metadata=metadata,
        )
        self._processed_events.add(event_id)
        return order

    # ----------------------------------------------------------- cancel/replace
    def cancel(
        self,
        order_id: str,
        *,
        reason: str = "",
        event_id: str | None = None,
        confirm: bool = True,
    ) -> Order:
        if event_id and self._is_duplicate_event(event_id, order_id, "cancel"):
            return self.get(order_id)
        order = self.get(order_id)
        begin_cancel(order, audit=self.audit, reason=reason)
        if confirm and order.state is OrderState.CANCEL_PENDING:
            mark_cancelled(order, audit=self.audit, reason=reason or "cancelled")
        if event_id:
            self._processed_events.add(event_id)
        return order

    def replace(
        self,
        order_id: str,
        *,
        quantity: float | None = None,
        price: float | None = None,
        stop_price: float | None = None,
        order_type: OrderType | None = None,
        time_in_force: TimeInForce | None = None,
        urgency: Urgency | None = None,
        reason: str = "",
        request_id: str | None = None,
        auto_approve: bool = False,
    ) -> Order:
        """Replace an open order. Returns the new (replacement) order."""
        rid = request_id or f"rp_{uuid4().hex[:12]}"
        if rid in self._processed_events:
            # Find replacement by idempotency key
            key = f"replace|{order_id}|{rid}"
            if key in self._by_idempotency:
                return self.get(self._by_idempotency[key])
        original = self.get(order_id)
        req = ReplaceRequest(
            order_id=order_id,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            order_type=order_type,
            time_in_force=time_in_force,
            urgency=urgency,
            request_id=rid,
            reason=reason,
        )
        replacement = build_replacement(original, req, audit=self.audit)
        self._orders[replacement.order_id] = replacement
        if replacement.idempotency_key:
            self._by_idempotency[replacement.idempotency_key] = replacement.order_id
        self._processed_events.add(rid)
        if auto_approve:
            self.validate_and_approve(replacement.order_id)
        return replacement

    # ----------------------------------------------------------------- registry
    def get(self, order_id: str) -> Order:
        try:
            return self._orders[order_id]
        except KeyError as exc:
            raise ExecutionError(
                f"unknown order_id: {order_id}",
                code="ORDER_NOT_FOUND",
                details={"order_id": order_id},
            ) from exc

    def list_orders(self, *, state: OrderState | None = None) -> list[Order]:
        orders = list(self._orders.values())
        if state is not None:
            orders = [o for o in orders if o.state is state]
        return orders

    def register_parent(self, parent: ParentOrder) -> ParentOrder:
        self._parents[parent.parent_id] = parent
        return parent

    def register_group(self, group: OrderGroup) -> OrderGroup:
        self._groups[group.group_id] = group
        return group

    def process_event(
        self,
        event_id: str,
        event_type: str,
        *,
        order_id: str,
        payload: dict[str, Any] | None = None,
    ) -> Order:
        """Generic idempotent event dispatcher."""
        if event_id in self._processed_events:
            self.audit.append(
                "event_idempotent_skip",
                f"duplicate event {event_id}",
                order_id=order_id,
                details={"event_type": event_type, "event_id": event_id},
            )
            return self.get(order_id)
        payload = dict(payload or {})
        et = event_type.lower()
        if et == "acknowledge":
            return self.acknowledge(
                order_id, venue_order_id=payload.get("venue_order_id"), event_id=event_id
            )
        if et == "fill":
            return self.apply_fill(
                order_id,
                fill_qty=float(payload["fill_qty"]),
                fill_price=float(payload["fill_price"]),
                event_id=event_id,
                venue_exec_id=payload.get("venue_exec_id"),
            )
        if et == "cancel":
            return self.cancel(order_id, reason=payload.get("reason", ""), event_id=event_id)
        if et == "reject":
            order = self.get(order_id)
            mark_rejected(order, reason=payload.get("reason", "rejected"), audit=self.audit)
            self._processed_events.add(event_id)
            return order
        raise ExecutionError(
            f"unknown event_type: {event_type}",
            code="UNKNOWN_EVENT_TYPE",
            details={"event_type": event_type, "event_id": event_id},
        )

    def reconcile_positions(
        self,
        *,
        expected: dict[str, float],
        executed: dict[str, float],
        broker: dict[str, float],
    ):
        return self.reconciler.reconcile(expected=expected, executed=executed, broker=broker)

    def _is_duplicate_event(self, event_id: str, order_id: str, action: str) -> bool:
        if event_id in self._processed_events:
            self.audit.append(
                "event_idempotent_skip",
                f"duplicate {action} event {event_id}",
                order_id=order_id,
                details={"event_id": event_id, "action": action},
            )
            return True
        return False

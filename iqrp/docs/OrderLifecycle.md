# Order Lifecycle

Order and execution state machines for institutional order management: allowed transitions, lifecycle helpers, and idempotent event processing.

**Modules:**  
`iqrp.app.execution.order_manager.order_state` · `order_lifecycle` · `execution_state`

Related: [OrderManager](OrderManager.md) · [ExecutionRisk](ExecutionRisk.md) · [ExecutionPlatform](ExecutionPlatform.md)

---

## Order states

```text
CREATED
   │
   ▼
VALIDATING ──► APPROVED ──► SUBMITTED ──► ACKNOWLEDGED
   │              │             │              │
   │              │             │              ├──► PARTIALLY_FILLED ──► FILLED
   │              │             │              │           │
   │              │             │              │           └──► CANCEL_PENDING ──► CANCELLED
   │              │             │              ├──► CANCEL_PENDING / CANCELLED / EXPIRED / REPLACED / FAILED
   │              │             ├──► PARTIALLY_FILLED / FILLED / REJECTED / CANCELLED / EXPIRED / FAILED
   │              ├──► CANCELLED / REJECTED / FAILED
   └──► REJECTED / CANCELLED / FAILED
```

### Terminal states

`FILLED` · `CANCELLED` · `REJECTED` · `EXPIRED` · `REPLACED` · `FAILED`

Terminal states have **empty** outgoing sets — no further transitions.

---

## Allowed transitions

| From | To |
|------|----|
| `CREATED` | `VALIDATING`, `REJECTED`, `CANCELLED`, `FAILED` |
| `VALIDATING` | `APPROVED`, `REJECTED`, `FAILED`, `CANCELLED` |
| `APPROVED` | `SUBMITTED`, `CANCELLED`, `REJECTED`, `FAILED` |
| `SUBMITTED` | `ACKNOWLEDGED`, `REJECTED`, `CANCEL_PENDING`, `CANCELLED`, `EXPIRED`, `FAILED`, `PARTIALLY_FILLED`, `FILLED` |
| `ACKNOWLEDGED` | `PARTIALLY_FILLED`, `FILLED`, `CANCEL_PENDING`, `CANCELLED`, `EXPIRED`, `REPLACED`, `FAILED` |
| `PARTIALLY_FILLED` | `PARTIALLY_FILLED` (self), `FILLED`, `CANCEL_PENDING`, `CANCELLED`, `EXPIRED`, `REPLACED`, `FAILED` |
| `CANCEL_PENDING` | `CANCELLED`, `PARTIALLY_FILLED`, `FILLED`, `FAILED`, `REJECTED` |
| Terminal | ∅ |

Illegal transitions raise `ExecutionError(code="ORDER_STATE_TRANSITION_ILLEGAL")` — never silently coerced.

```python
from iqrp.app.execution.order_manager.order_state import (
    OrderState,
    can_transition,
    transition_order,
    TERMINAL_STATES,
)

assert can_transition(OrderState.APPROVED, OrderState.SUBMITTED)
assert not can_transition(OrderState.FILLED, OrderState.SUBMITTED)
```

---

## Lifecycle helpers

```python
from iqrp.app.execution.order_manager.order_lifecycle import (
    begin_validation,
    approve,
    mark_submitted,
    mark_acknowledged,
    mark_partial,
    mark_filled,
    request_cancel,
    mark_cancelled,
    mark_rejected,
    mark_expired,
    mark_replaced,
    mark_failed,
    apply_fill_state,
    is_cancellable,
    is_replaceable,
)

begin_validation(order)          # → VALIDATING
approve(order)                   # → APPROVED
mark_submitted(order, venue="SIM")
mark_acknowledged(order, venue_order_id="V-1")
apply_fill_state(order)          # residual → PARTIALLY_FILLED or FILLED
```

| Helper | Effect |
|--------|--------|
| `state_after_fill` | Derive fill state from residual (no future info) |
| `is_cancellable` | False for terminal / `CANCEL_PENDING` / early CREATED-VALIDATING |
| `is_replaceable` | `SUBMITTED`, `ACKNOWLEDGED`, `PARTIALLY_FILLED` |

Each transition appends to `order.audit` and optional `AuditLog`.

---

## Idempotent events

`OrderManager` and `ExecutionEngine` treat `event_id` as a once-only key:

| Event | Manager method | Typical transition |
|-------|----------------|--------------------|
| create (idempotency_key) | `create_order` | return existing |
| submit | `submit(..., event_id=)` | `APPROVED` → `SUBMITTED` |
| acknowledge | `acknowledge(..., event_id=)` | → `ACKNOWLEDGED` |
| fill | `apply_fill(..., event_id=)` | → partial / filled |
| cancel | `cancel(..., event_id=)` | → `CANCEL_PENDING` / `CANCELLED` |
| reject | `process_event(..., "reject")` | → `REJECTED` |

```python
om.apply_fill(oid, fill_qty=100, fill_price=10.0, event_id="F1")
om.apply_fill(oid, fill_qty=100, fill_price=10.0, event_id="F1")  # no-op
```

Duplicate events audit `event_idempotent_skip` / `fill_idempotent_skip` and return the current order unchanged. Fills also require non-empty `event_id`.

Engine: `engine.apply_event(event_id, event_type="fill", order_id=..., fill_qty=..., fill_price=...)`.

---

## Execution-level state machine

Parent / engine workflow (`ExecutionState`):

```text
IDLE → PLANNING → VALIDATING → EXECUTING ⇄ PARTIALLY_EXECUTED → COMPLETING → COMPLETED
                      │              │
                      │              ├──► CANCELLED / FAILED / HALTED
                      └──► CANCELLED / FAILED / HALTED
HALTED → IDLE | PLANNING | CANCELLED | FAILED
```

| State | Meaning |
|-------|---------|
| `IDLE` | Ready |
| `PLANNING` | Target → parent/order materialization |
| `VALIDATING` | Risk + order validation |
| `EXECUTING` | Children live |
| `PARTIALLY_EXECUTED` | Some fills, residual remains |
| `COMPLETING` / `COMPLETED` | Wind-down / done |
| `CANCELLED` / `FAILED` | Terminal failure paths |
| `HALTED` | Kill / operator halt — blocks new work |

```python
from iqrp.app.execution.order_manager.execution_state import (
    ExecutionState,
    transition_execution,
)

new = transition_execution(ExecutionState.IDLE, ExecutionState.PLANNING)
```

Illegal edges raise `ExecutionError(code="EXECUTION_STATE_TRANSITION_ILLEGAL")`.

---

## Happy path (OrderManager)

```python
from iqrp.app.execution import OrderManager, Side, OrderType

om = OrderManager()
o = om.create_order(instrument="AAPL", side=Side.BUY, quantity=100, order_type=OrderType.LIMIT, price=190.0)
om.validate_and_approve(o.order_id)                 # CREATED → VALIDATING → APPROVED
om.submit(o.order_id, venue="SIM", event_id="s1")   # → SUBMITTED
om.acknowledge(o.order_id, venue_order_id="1", event_id="a1")
om.apply_fill(o.order_id, fill_qty=100, fill_price=190.01, event_id="f1")  # → FILLED
assert o.state.value == "FILLED"
```

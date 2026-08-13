# Order Manager

Institutional order book for create → validate → approve → submit → acknowledge → fill → cancel/replace, with parent/child hierarchy, groups, idempotent events, and position reconciliation.

**Package:** `iqrp.app.execution.order_manager`  
**Primary type:** `OrderManager`  
**Settings:** `ExecutionSettings` / `iqrp/configs/execution/default.yaml`

Related: [OrderLifecycle](OrderLifecycle.md) · [ExecutionPlatform](ExecutionPlatform.md) · [ExecutionRisk](ExecutionRisk.md) · [PositionReconciliation](PositionReconciliation.md)

---

## Role

`OrderManager` owns working orders. It does **not** choose targets or generate alpha. Inputs are explicit order fields or current→target deltas from Portfolio. Hard risk and kill switches gate every submit; urgency never bypasses them.

```python
from iqrp.app.execution import OrderManager, ExecutionSettings, Side, OrderType, Urgency

om = OrderManager(ExecutionSettings.default())
order = om.create_order(
    instrument="AAPL",
    side=Side.BUY,
    quantity=500,
    order_type=OrderType.LIMIT,
    price=190.0,
    urgency=Urgency.NORMAL,
    strategy_id="s1",
)
om.validate_and_approve(order.order_id)
om.submit(order.order_id, venue="SIM")
om.acknowledge(order.order_id, venue_order_id="SIM-1", event_id="ack-1")
om.apply_fill(
    order.order_id,
    fill_qty=500,
    fill_price=190.01,
    event_id="fill-1",
)
```

---

## `Order`

| Field | Notes |
|-------|-------|
| `instrument`, `side`, `quantity` | Required; instrument uppercased |
| `order_type`, `price`, `stop_price` | Microstructure fields |
| `time_in_force` | Default from settings (`DAY`) |
| `venue`, `algo`, `urgency` | Routing / planner hints |
| `strategy_id`, `portfolio_id`, `account_id` | Scopes for kill / audit |
| `parent_id` | Link to `ParentOrder` |
| `client_order_id`, `idempotency_key` | Client identity; auto-derived if omitted |
| `filled_qty`, `avg_fill_price`, `residual_qty` | Fill tracking |
| `state` | `OrderState` (see [OrderLifecycle](OrderLifecycle.md)) |
| `audit`, `tags`, `metadata` | Append-only / free-form |

Properties: `residual_qty`, `is_terminal`, `notional`. Serialization: `to_dict()` / `from_dict()`.

### Idempotency key

If omitted, derived as:

```text
{strategy_id}|{instrument}|{side}|{quantity}|{order_type}|{price}|{client_order_id}
```

`create_order` with a duplicate key returns the **existing** order (idempotent create).

---

## Order states

`CREATED` → `VALIDATING` → `APPROVED` → `SUBMITTED` → `ACKNOWLEDGED` → `PARTIALLY_FILLED` / `FILLED`, plus `CANCEL_PENDING`, `CANCELLED`, `REJECTED`, `EXPIRED`, `REPLACED`, `FAILED`.

Terminal: `FILLED`, `CANCELLED`, `REJECTED`, `EXPIRED`, `REPLACED`, `FAILED`.

Full transition table: [OrderLifecycle](OrderLifecycle.md).

---

## Parent / child

### `ParentOrder`

Program-level quantity owning child working orders.

```python
from iqrp.app.execution import ParentOrder, Side, Urgency

parent = ParentOrder(
    instrument="MSFT",
    side=Side.BUY,
    quantity=10_000,
    urgency=Urgency.HIGH,
    algo="vwap",
    strategy_id="s1",
)
om.register_parent(parent)
```

- `residual_qty = quantity - filled_qty`
- `attach_child(child)` sets `child.parent_id` and records `child_ids`
- `sync_fills_from_children(children)` aggregates fills and sets `PARTIALLY_EXECUTED` / `COMPLETED`

`ExecutionEngine` creates children via `create_child_order` and clips each slice to `parent.residual_qty`. Algorithms must not plan beyond approved residual.

### Child orders

Children are ordinary `Order` instances with `parent_id` set. Slice metadata lives in `tags` (e.g. `slice`, `not_before_offset`).

---

## Order groups

```python
from iqrp.app.execution.order_manager.order_group import OrderGroup, GroupType

group = OrderGroup(name="pair_ab", group_type=GroupType.PAIR, strategy_id="s1")
group.add_order(leg_a, leg="A", ratio=1.0)
group.add_order(leg_b, leg="B", ratio=-1.0)
om.register_group(group)
group.sync_state(om.list_orders())
```

Types: `BASKET`, `PAIR`, `SPREAD`, `LIST`. Group urgency is advisory only — member hard risk still applies.

---

## `target_to_orders`

Convert approved current→target maps into delta `OrderSpec`s (no alpha):

```python
from iqrp.app.execution.order_manager.order import target_to_orders

specs = target_to_orders(
    current={"AAPL": 100.0, "MSFT": 50.0},
    target={"AAPL": 150.0, "MSFT": 0.0},
    prices={"AAPL": 190.0, "MSFT": 420.0},
    lot_size=1.0,
    min_qty=1.0,
    urgency=Urgency.NORMAL,
)
# AAPL BUY 50, MSFT SELL 50
orders = om.create_from_target(
    {"AAPL": 100.0}, {"AAPL": 150.0}, prices={"AAPL": 190.0}
)
```

Rules:

- `delta = target - current`; zero deltas skipped
- Lot-round when `round_lots=True`; drop below `min_qty`
- Side `BUY` if delta > 0 else `SELL`
- Tags record `current`, `target`, `delta`

Engine shortcut: `engine.plan_from_targets(current, target)`.

---

## Validation

`OrderValidator` checks instrument meta, quantity bounds, tick/lot, price bands, capital/notional caps, and optional `validate_risk(order) -> (ok, reason)`.

```python
result = om.validator.validate(order)
# ValidationResult(ok=..., errors=[...], warnings=[...])
om.validate_and_approve(order.order_id)  # CREATED → VALIDATING → APPROVED | REJECTED
```

Hard risk rejects set `REJECTED` and raise `ValidationError` / `ExecutionError` — never overridden by urgency.

---

## Fills

`FillManager` applies fills keyed by `event_id`:

```python
om.apply_fill(
    order_id,
    fill_qty=100,
    fill_price=190.05,
    event_id="venue-exec-42",
    venue_exec_id="42",
    liquidity_flag="REMOVE",
    fees=0.5,
)
# Duplicate event_id → no-op, returns current order
```

- Updates `filled_qty`, VWAP `avg_fill_price`, residual
- Transitions to `PARTIALLY_FILLED` / `FILLED` via lifecycle helpers
- Overfill raises unless `settings.fills.allow_overfill`

`Fill` fields: `order_id`, `fill_qty`, `fill_price`, `event_id`, `timestamp`, `venue_exec_id`, `liquidity_flag`, `fees`, `metadata`.

---

## Submit gates

Before `SUBMITTED`:

1. State must be `APPROVED`
2. Kill switch scopes (global / account / venue / strategy) per config
3. Re-check `validate_risk` when `risk.enforce_hard_limits`

```python
om.submit(order.order_id, venue="SIM", event_id="sub-1")
```

---

## Cancel / replace

```python
om.cancel(order_id, reason="user", event_id="c1", confirm=True)
replacement = om.replace(
    order_id,
    quantity=400,
    price=189.5,
    reason="improve",
    auto_approve=True,
)
```

Replace marks the original `REPLACED` and returns a new `Order` with a distinct idempotency key (`replace|{order_id}|{request_id}`).

---

## Events

Generic idempotent dispatcher:

```python
om.process_event(
    "evt-9",
    "fill",
    order_id=order.order_id,
    payload={"fill_qty": 50, "fill_price": 190.0},
)
```

Supported types: `acknowledge`, `fill`, `cancel`, `reject`. Duplicate `event_id` → audit skip + return current order.

Engine wrapper: `engine.apply_event(event_id, event_type="fill", order_id=..., fill_qty=..., fill_price=...)`.

---

## Reconciliation

```python
result = om.reconcile_positions(
    expected={"AAPL": 1000},
    executed={"AAPL": 1000},
    broker={"AAPL": 998},
)
# result.matched, result.alerts, result.per_instrument
```

See [PositionReconciliation](PositionReconciliation.md).

---

## Registry helpers

| Method | Description |
|--------|-------------|
| `get(order_id)` | Lookup or `ORDER_NOT_FOUND` |
| `list_orders(state=None)` | Filter by `OrderState` |
| `register_parent` / `register_group` | Attach parent / group objects |
| `create_from_spec` / `create_from_target` | Spec / delta helpers |

Audit trail: `om.audit` (`AuditLog`) plus per-order `order.audit`.

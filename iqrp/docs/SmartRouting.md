# Smart Routing

Multi-venue order routing: score venues, allocate quantity, enforce safety gates, and fall back when primaries fail.

**Package:** `iqrp.app.execution.smart_routing`  
**Primary types:** `SmartRouter`, `RoutingDecision`, `SimulatedVenue`, `Venue`, `VenueState`

Related: [ExecutionPlatform](ExecutionPlatform.md) · [ExecutionRisk](ExecutionRisk.md) · [OrderManager](OrderManager.md)

---

## Safety rules (hard)

Never route when any of the following hold:

1. Venue unavailable / halted / trading disabled  
2. Instrument not listed on venue  
3. Order type unsupported  
4. Invalid price or quantity (tick/lot)  
5. Kill switch engaged (global / account / venue / strategy)  
6. Risk check callback rejects `(order, venue)`

Urgency and alpha confidence never override these gates. Execution never generates alpha.

---

## Venues

### `Venue` / `VenueState`

Logical venue with microstructure state: bid/ask/mid, ADV, volatility, available qty, supported order types, reliability / latency scores, routable flags.

### `SimulatedVenue`

In-process venue for tests and paper fills:

```python
from iqrp.app.execution import SimulatedVenue, SmartRouter, Order, Side, OrderType

venue_a = SimulatedVenue(
    venue_id="SIM_A",
    instruments={"AAPL"},
    mode="fill",
    mid=190.0,
    spread=0.02,
    adv=5e7,
)
venue_b = SimulatedVenue(
    venue_id="SIM_B",
    instruments={"AAPL"},
    mode="fill",
    mid=190.01,
    spread=0.03,
    adv=2e7,
)
```

`mode` controls response behavior (`fill`, reject, partial, etc.). Engine defaults to a single `SimulatedVenue` when `venues=None`.

---

## Scoring

Weighted composite — higher score is better:

| Component | Default weight | Interpretation |
|-----------|----------------|----------------|
| `price` | 0.25 | Side-aware expected price quality |
| `fees` | 0.10 | Lower fee → higher score |
| `spread` | 0.10 | Tighter → better |
| `liquidity` | 0.15 | Depth / available qty |
| `impact` | 0.15 | Lower expected impact |
| `fill_prob` | 0.10 | Historical / modelled fill odds |
| `latency` | 0.05 | Lower latency → better |
| `reliability` | 0.10 | Uptime / reject rate |

```python
from iqrp.app.execution.smart_routing.scoring import ScoreWeights, rank_venues, score_venue

weights = ScoreWeights(price=0.3, impact=0.2, liquidity=0.2)
# SmartRouter(weights=weights)
```

Cost model: `estimate_venue_cost` → `VenueCostEstimate`.  
Liquidity: `assess_liquidity` → `LiquiditySnapshot`.

---

## `SmartRouter`

```python
from iqrp.app.execution import SmartRouter, KillSwitch

router = SmartRouter(
    weights=None,
    mode="single",          # or "multi"
    impact_coeff=0.1,
    kill_switch=KillSwitch(),
    risk_check=None,        # (order, venue) -> bool | (bool, reason)
    max_fallbacks=5,
    max_venues=3,
)

order = Order(instrument="AAPL", side=Side.BUY, quantity=1000, order_type=OrderType.LIMIT, price=190.0)
decision = router.route(order, [venue_a, venue_b])
print(decision.accepted, decision.primary_venue_id, decision.to_dict())
```

### `RoutingDecision`

| Field | Meaning |
|-------|---------|
| `accepted` | At least one eligible venue |
| `primary_venue_id` | Top choice |
| `allocations` | `VenueAllocation` list (`single` or split) |
| `scores` / `costs` / `liquidity` | Diagnostics |
| `fallback` | `FallbackChain` after primary |
| `rejections` | `RejectionReason` list (`code`, `message`, `venue_id`) |
| `residual_qty` | Unallocated quantity |

---

## Allocation

| Mode | Behavior |
|------|----------|
| `single` | 100% to best eligible venue |
| `multi` | Split across top venues up to `max_venues` by score / liquidity |

`allocate_quantity` produces an `AllocationPlan` with residual if capacity binds. Residual is never silently inflated.

---

## Fallback

```python
from iqrp.app.execution.smart_routing.fallback import (
    build_fallback_chain,
    select_fallback,
)

chain = build_fallback_chain(decision.scores, primary_venue_id=decision.primary_venue_id, max_fallbacks=3)
# On primary failure:
next_venue = select_fallback(
    chain,
    {"SIM_A": venue_a, "SIM_B": venue_b},
    failed_venue_id="SIM_A",
    failure_reason="timeout",
)
```

`FallbackChain.next_venue()` advances the cursor; skips non-routable venues.

---

## Engine integration

```python
from iqrp.app.execution import ExecutionEngine

engine = ExecutionEngine()
report = engine.execute(
    {"AAPL": 2000},
    current={"AAPL": 0},
    algo="twap",
    venues=[venue_a, venue_b],
    market_context={"AAPL": {"mid": 190.0, "spread": 0.02, "adv": 5e7}},
)
# report.routing[*] == RoutingDecision.to_dict()
```

Manual: `engine.route(order, venues)`. Shared `KillSwitch` instance is propagated to router and order manager on `halt` / `kill`.

---

## Rejection codes (examples)

Router emits structured `RejectionReason` entries such as venue not routable, instrument missing, unsupported type, kill switch active, risk reject, invalid qty/price. When all venues fail, `accepted=False` and the engine records routing errors without inventing fills.
